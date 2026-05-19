//! Fix `MemoryAccess::Aligned` on `OpLoad`/`OpStore`/`OpCopyMemory` to
//! match the pointer's storage class (only fully known after the
//! storage-class specializer): strip the bit on logical-addressing
//! pointers (spec-forbidden there) and add it with the pointee's natural
//! alignment on `PhysicalStorageBuffer` (spec-required, and
//! `PhysicalPtr<T>::get` results arrive here as `Inferred` and only get
//! promoted to physical by the specializer).
//
// Limited to the `Aligned` bit; rust-gpu never emits
// `MakePointerAvailable`/`Visible` (which would carry id operands), so we
// don't need to worry about shuffling unrelated operands.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use spirt::func_at::FuncAtMut;
use spirt::transform::{InnerInPlaceTransform, InnerTransform, Transformed, Transformer};
use spirt::{
    Context, DataInst, DataInstForm, DataInstFormDef, DataInstKind, Func, GlobalVar, Module, Type,
    TypeKind, TypeOrConst, Value, spv,
};
use std::collections::VecDeque;

use super::SpvWellKnownWithExtras;

/// `MemoryAccess::Aligned` bit value (SPIR-V spec, OperandKind `MemoryAccess`).
const MEMORY_ACCESS_ALIGNED_BIT: u32 = 0x2;

/// Used when we can't compute a natural alignment (opaque/unmodelled types).
/// The literal is just a guarantee hint; the producer's actual alignment wins.
const FALLBACK_ALIGN: u32 = 1;

pub fn strip_when_invalid(module: &mut Module) {
    let cx = &module.cx();
    let wk = &super::SpvSpecWithExtras::get().well_known;

    let mut fixer = AlignmentFixer {
        cx,
        wk,
        transformed_data_inst_forms: FxHashMap::default(),
        seen_global_vars: FxHashSet::default(),
        global_var_queue: VecDeque::new(),
        seen_funcs: FxHashSet::default(),
        func_queue: VecDeque::new(),
    };

    fixer.in_place_transform_module(module);

    while !fixer.global_var_queue.is_empty() || !fixer.func_queue.is_empty() {
        while let Some(gv) = fixer.global_var_queue.pop_front() {
            fixer.in_place_transform_global_var_decl(&mut module.global_vars[gv]);
        }
        while let Some(func) = fixer.func_queue.pop_front() {
            fixer.in_place_transform_func_decl(&mut module.funcs[func]);
        }
    }
}

struct AlignmentFixer<'a> {
    cx: &'a Context,
    wk: &'static super::SpvWellKnownWithExtras,

    transformed_data_inst_forms: FxHashMap<DataInstForm, Transformed<DataInstForm>>,
    seen_global_vars: FxHashSet<GlobalVar>,
    global_var_queue: VecDeque<GlobalVar>,
    seen_funcs: FxHashSet<Func>,
    func_queue: VecDeque<Func>,
}

impl Transformer for AlignmentFixer<'_> {
    fn transform_data_inst_form_use(
        &mut self,
        data_inst_form: DataInstForm,
    ) -> Transformed<DataInstForm> {
        // Recurse into the form so that `FuncCall` (and other) inner uses are
        // visited — the default impl returns `Unchanged` without descending,
        // which would leave callee funcs unqueued.
        if let Some(&cached) = self.transformed_data_inst_forms.get(&data_inst_form) {
            return cached;
        }
        let transformed = self.cx[data_inst_form]
            .inner_transform_with(self)
            .map(|def| self.cx.intern(def));
        self.transformed_data_inst_forms
            .insert(data_inst_form, transformed);
        transformed
    }

    fn transform_global_var_use(&mut self, gv: GlobalVar) -> Transformed<GlobalVar> {
        if self.seen_global_vars.insert(gv) {
            self.global_var_queue.push_back(gv);
        }
        Transformed::Unchanged
    }
    fn transform_func_use(&mut self, func: Func) -> Transformed<Func> {
        if self.seen_funcs.insert(func) {
            self.func_queue.push_back(func);
        }
        Transformed::Unchanged
    }

    fn in_place_transform_data_inst_def(&mut self, mut func_at_data_inst: FuncAtMut<'_, DataInst>) {
        let cx = self.cx;
        let wk = self.wk;

        func_at_data_inst
            .reborrow()
            .inner_in_place_transform_with(self);

        let func_at_data_inst_frozen = func_at_data_inst.reborrow().freeze();
        let data_inst_def = func_at_data_inst_frozen.def();
        let inst_form_def = &cx[data_inst_def.form];

        let DataInstKind::SpvInst(spv_inst) = &inst_form_def.kind else {
            return;
        };

        let opcode = spv_inst.opcode;
        let is_load = opcode == wk.OpLoad;
        let is_store = opcode == wk.OpStore;
        let is_copy = opcode == wk.OpCopyMemory;
        if !(is_load || is_store || is_copy) {
            return;
        }

        let func = func_at_data_inst_frozen.at(());
        // Returns (storage_class, pointee_type) for a pointer-typed value.
        let pointer_info = |ptr: Value| -> Option<(u32, Type)> {
            let ptr_ty = func.at(ptr).type_of(cx);
            let TypeKind::SpvInst {
                spv_inst: ptr_spv_inst,
                type_and_const_inputs,
            } = &cx[ptr_ty].kind
            else {
                return None;
            };
            if ptr_spv_inst.opcode != wk.OpTypePointer {
                return None;
            }
            let sc = match ptr_spv_inst.imms[..] {
                [spv::Imm::Short(_, sc)] => sc,
                _ => return None,
            };
            let pointee = type_and_const_inputs.first().and_then(|toc| match toc {
                &TypeOrConst::Type(t) => Some(t),
                _ => None,
            })?;
            Some((sc, pointee))
        };

        let dst_info = pointer_info(data_inst_def.inputs[0]);
        let src_info = if is_copy {
            pointer_info(data_inst_def.inputs[1])
        } else {
            dst_info
        };

        // Pick alignment from the value type (load output, store value, or
        // pointee for copy).
        let access_align = |info: Option<(u32, Type)>, output_type: Option<Type>| -> u32 {
            if let Some(ty) = output_type {
                natural_alignment(cx, wk, ty)
            } else if let Some((_, pointee)) = info {
                natural_alignment(cx, wk, pointee)
            } else {
                FALLBACK_ALIGN
            }
        };

        let dst_align = if is_load {
            access_align(dst_info, inst_form_def.output_type)
        } else if is_store {
            access_align(
                dst_info,
                Some(func.at(data_inst_def.inputs[1]).type_of(cx)),
            )
        } else {
            access_align(dst_info, None)
        };
        let src_align = if is_copy { access_align(src_info, None) } else { dst_align };

        let dst_physical = dst_info.is_some_and(|(sc, _)| sc == wk.PhysicalStorageBuffer);
        let src_physical = src_info.is_some_and(|(sc, _)| sc == wk.PhysicalStorageBuffer);

        // Walk the imms, ensuring Aligned on physical pointers and removing
        // it on non-physical. Slot 0 = dst, slot 1 = src (`OpCopyMemory`).
        let mut new_imms: SmallVec<[spv::Imm; 4]> = SmallVec::new();
        let mut imm_iter = spv_inst.imms.iter().copied().peekable();
        let mut mem_access_seen = [false; 2];

        while let Some(imm) = imm_iter.next() {
            match imm {
                spv::Imm::Short(kind, bits) if kind == wk.MemoryAccess => {
                    let slot = if is_copy && mem_access_seen[0] { 1 } else { 0 };
                    mem_access_seen[slot] = true;

                    let target_physical = if slot == 0 { dst_physical } else { src_physical };
                    let has_aligned = (bits & MEMORY_ACCESS_ALIGNED_BIT) != 0;
                    let align = if slot == 0 { dst_align } else { src_align };

                    let new_bits = if target_physical {
                        bits | MEMORY_ACCESS_ALIGNED_BIT
                    } else {
                        bits & !MEMORY_ACCESS_ALIGNED_BIT
                    };
                    new_imms.push(spv::Imm::Short(kind, new_bits));

                    if has_aligned {
                        let existing = imm_iter.next(); // consume the literal
                        if target_physical {
                            new_imms.push(existing.unwrap_or(spv::Imm::Short(
                                wk.MemoryAccess,
                                align,
                            )));
                        }
                    } else if target_physical {
                        new_imms.push(spv::Imm::Short(wk.LiteralInteger, align));
                    }
                }
                _ => new_imms.push(imm),
            }
        }

        // Add MemoryAccess slots for any physical pointer that didn't have one.
        if is_load || is_store {
            if dst_physical && !mem_access_seen[0] {
                new_imms.push(spv::Imm::Short(wk.MemoryAccess, MEMORY_ACCESS_ALIGNED_BIT));
                new_imms.push(spv::Imm::Short(wk.LiteralInteger, dst_align));
            }
        } else if is_copy {
            match (mem_access_seen[0], mem_access_seen[1]) {
                (false, false) => {
                    if dst_physical || src_physical {
                        let dst_bits = if dst_physical {
                            MEMORY_ACCESS_ALIGNED_BIT
                        } else {
                            0
                        };
                        let src_bits = if src_physical {
                            MEMORY_ACCESS_ALIGNED_BIT
                        } else {
                            0
                        };
                        // Two-operand form (1.4+) so per-side alignment is independent.
                        new_imms.push(spv::Imm::Short(wk.MemoryAccess, dst_bits));
                        if dst_physical {
                            new_imms.push(spv::Imm::Short(wk.LiteralInteger, dst_align));
                        }
                        if dst_physical != src_physical || src_physical {
                            new_imms.push(spv::Imm::Short(wk.MemoryAccess, src_bits));
                            if src_physical {
                                new_imms.push(spv::Imm::Short(wk.LiteralInteger, src_align));
                            }
                        }
                    }
                }
                (true, false) => {
                    // Single MA covered both; split off a src MA if src is physical.
                    if src_physical {
                        new_imms.push(spv::Imm::Short(
                            wk.MemoryAccess,
                            MEMORY_ACCESS_ALIGNED_BIT,
                        ));
                        new_imms.push(spv::Imm::Short(wk.LiteralInteger, src_align));
                    }
                }
                // Other shapes already cover both sides; leave alone.
                (false, true) | (true, true) => {}
            }
        }

        // `MemoryAccess(None)` ≡ absent. For `OpCopyMemory` drop both slots
        // together — dropping just one would change which side the other
        // applies to.
        let is_none_ma =
            |imm: &spv::Imm| matches!(imm, spv::Imm::Short(k, 0) if *k == wk.MemoryAccess);
        if is_load || is_store {
            while new_imms.last().is_some_and(is_none_ma) {
                new_imms.pop();
            }
        } else if is_copy {
            let all_mas_none = new_imms
                .iter()
                .filter(|imm| matches!(imm, spv::Imm::Short(k, _) if *k == wk.MemoryAccess))
                .all(is_none_ma);
            if all_mas_none {
                new_imms.retain(
                    |imm| !matches!(imm, spv::Imm::Short(k, _) if *k == wk.MemoryAccess),
                );
            }
        }

        // Bail out early if nothing changed.
        if new_imms.as_slice() == spv_inst.imms.as_slice() {
            return;
        }

        let new_form = cx.intern(DataInstFormDef {
            kind: DataInstKind::SpvInst(spv::Inst {
                opcode: spv_inst.opcode,
                imms: new_imms.into_iter().collect(),
            }),
            output_type: inst_form_def.output_type,
        });

        func_at_data_inst.def().form = new_form;
    }
}

/// Natural alignment in bytes of a SPIR-V type. Falls back to
/// `FALLBACK_ALIGN` for opaque/unmodelled types.
fn natural_alignment(cx: &Context, wk: &SpvWellKnownWithExtras, ty: Type) -> u32 {
    let TypeKind::SpvInst {
        spv_inst,
        type_and_const_inputs,
    } = &cx[ty].kind
    else {
        return FALLBACK_ALIGN;
    };
    let op = spv_inst.opcode;
    if op == wk.OpTypeInt || op == wk.OpTypeFloat {
        if let Some(&spv::Imm::Short(_, bit_width)) = spv_inst.imms.first() {
            return ((bit_width / 8).max(1)).next_power_of_two();
        }
        return FALLBACK_ALIGN;
    }
    if op == wk.OpTypeVector || op == wk.OpTypeMatrix || op == wk.OpTypeArray
        || op == wk.OpTypeRuntimeArray
    {
        if let Some(&TypeOrConst::Type(elem)) = type_and_const_inputs.first() {
            return natural_alignment(cx, wk, elem);
        }
        return FALLBACK_ALIGN;
    }
    if op == wk.OpTypeStruct {
        return type_and_const_inputs
            .iter()
            .filter_map(|toc| match toc {
                &TypeOrConst::Type(field) => Some(natural_alignment(cx, wk, field)),
                _ => None,
            })
            .max()
            .unwrap_or(FALLBACK_ALIGN);
    }
    FALLBACK_ALIGN
}
