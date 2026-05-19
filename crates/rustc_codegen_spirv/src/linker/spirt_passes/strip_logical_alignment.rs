//! Strip `Aligned` memory-access decorations from `OpLoad`/`OpStore`/
//! `OpCopyMemory` whose pointer's storage class doesn't permit them.
//!
//! `rustc_codegen_spirv` unconditionally emits `MemoryAccess::Aligned` on every
//! load/store/memcpy because `PhysicalStorageBuffer` accesses *require* it; the
//! SPIR-V spec forbids `Aligned` on logical-addressing storage classes, so the
//! same blanket emission would otherwise produce a `spirv-val` failure on any
//! shader that touches a non-physical pointer. This pass walks the module and
//! drops the `Aligned` bit (plus its trailing alignment literal) wherever the
//! pointer is not in `PhysicalStorageBuffer`.
//!
//! For `OpCopyMemory` the first `MemoryAccess` operand applies to the dst
//! pointer, the second (1.4+) to the src.
//
// NOTE: this is intentionally limited to the `Aligned` bit. `rustc_codegen_spirv`
// does not emit `MakePointerAvailable` / `MakePointerVisible` (which would carry
// id operands trailing the literal), so encountering them is a bug we leave
// alone rather than risk shuffling unrelated operands.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use spirt::func_at::FuncAtMut;
use spirt::transform::{InnerInPlaceTransform, InnerTransform, Transformed, Transformer};
use spirt::{
    Context, DataInst, DataInstForm, DataInstFormDef, DataInstKind, Func, GlobalVar, Module,
    TypeKind, Value, spv,
};
use std::collections::VecDeque;

/// `MemoryAccess::Aligned` bit value (SPIR-V spec, OperandKind `MemoryAccess`).
const MEMORY_ACCESS_ALIGNED_BIT: u32 = 0x2;

pub fn strip_when_invalid(module: &mut Module) {
    let cx = &module.cx();
    let wk = &super::SpvSpecWithExtras::get().well_known;

    let mut stripper = Stripper {
        cx,
        wk,
        transformed_data_inst_forms: FxHashMap::default(),
        seen_global_vars: FxHashSet::default(),
        global_var_queue: VecDeque::new(),
        seen_funcs: FxHashSet::default(),
        func_queue: VecDeque::new(),
    };

    stripper.in_place_transform_module(module);

    while !stripper.global_var_queue.is_empty() || !stripper.func_queue.is_empty() {
        while let Some(gv) = stripper.global_var_queue.pop_front() {
            stripper.in_place_transform_global_var_decl(&mut module.global_vars[gv]);
        }
        while let Some(func) = stripper.func_queue.pop_front() {
            stripper.in_place_transform_func_decl(&mut module.funcs[func]);
        }
    }
}

struct Stripper<'a> {
    cx: &'a Context,
    wk: &'static super::SpvWellKnownWithExtras,

    transformed_data_inst_forms: FxHashMap<DataInstForm, Transformed<DataInstForm>>,
    seen_global_vars: FxHashSet<GlobalVar>,
    global_var_queue: VecDeque<GlobalVar>,
    seen_funcs: FxHashSet<Func>,
    func_queue: VecDeque<Func>,
}

impl Transformer for Stripper<'_> {
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
        let pointee_storage_class = |ptr: Value| -> Option<u32> {
            let ptr_ty = func.at(ptr).type_of(cx);
            let TypeKind::SpvInst {
                spv_inst: ptr_spv_inst,
                ..
            } = &cx[ptr_ty].kind
            else {
                return None;
            };
            if ptr_spv_inst.opcode != wk.OpTypePointer {
                return None;
            }
            match ptr_spv_inst.imms[..] {
                [spv::Imm::Short(_, sc)] => Some(sc),
                _ => None,
            }
        };

        // First pointer applies to the first `MemoryAccess`; the second (only on
        // `OpCopyMemory`) to the second `MemoryAccess`.
        let dst_sc = pointee_storage_class(data_inst_def.inputs[0]);
        let src_sc = if is_copy {
            pointee_storage_class(data_inst_def.inputs[1])
        } else {
            dst_sc
        };

        let mut new_imms: SmallVec<[spv::Imm; 4]> = SmallVec::new();
        let mut imm_iter = spv_inst.imms.iter().copied().peekable();
        let mut mem_access_count = 0u32;
        let mut changed = false;

        while let Some(imm) = imm_iter.next() {
            match imm {
                spv::Imm::Short(kind, bits) if kind == wk.MemoryAccess => {
                    let target_sc = if is_copy && mem_access_count == 1 {
                        src_sc
                    } else {
                        dst_sc
                    };
                    mem_access_count += 1;

                    let is_physical = target_sc == Some(wk.PhysicalStorageBuffer);
                    let has_aligned = (bits & MEMORY_ACCESS_ALIGNED_BIT) != 0;

                    if has_aligned && !is_physical {
                        new_imms.push(spv::Imm::Short(kind, bits & !MEMORY_ACCESS_ALIGNED_BIT));
                        // Drop the alignment literal that follows.
                        imm_iter.next();
                        changed = true;
                    } else {
                        new_imms.push(imm);
                    }
                }
                _ => new_imms.push(imm),
            }
        }

        if !changed {
            return;
        }

        // An entirely-`None` `MemoryAccess` operand is semantically equivalent
        // to omitting it. For `OpLoad`/`OpStore` (single trailing operand) drop
        // it outright. For `OpCopyMemory` only drop the *whole* `MemoryAccess`
        // section when every operand is `None` — otherwise the remaining one
        // would silently change which pointer it applies to.
        let is_none_ma = |imm: &spv::Imm| {
            matches!(imm, spv::Imm::Short(k, 0) if *k == wk.MemoryAccess)
        };
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
                new_imms.retain(|imm| !matches!(imm, spv::Imm::Short(k, _) if *k == wk.MemoryAccess));
            }
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
