//! Patch two `PhysicalStorageBuffer`-specific quirks so idiomatic
//! `PhysicalPtr<T>` use lowers to valid SPIR-V:
//!
//! 1. `OpConstantNull` of a physical pointer (spec only allows it for
//!    logical or `VariablePointers` pointers; Rust's NPO `Option<&T>`
//!    emits one for `None`). Rewrite each use to a function-local
//!    `OpConvertUToPtr` of a `u64` zero.
//! 2. `OpConvertPtrToU` to a non-64-bit int on a physical pointer
//!    (Vulkan requires 64-bit under `PhysicalStorageBuffer64`; rustc emits
//!    u32 because the target's logical pointer width is 32). Widen to u64
//!    and patch the comparison consumer's zero-constant operand.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use spirt::func_at::FuncAtMut;
use spirt::transform::{InnerInPlaceTransform, InnerTransform, Transformed, Transformer};
use spirt::{
    Const, ConstDef, ConstKind, Context, ControlNode, ControlNodeKind, DataInst, DataInstDef,
    DataInstForm, DataInstFormDef, DataInstKind, Func, GlobalVar, Module, Type, TypeDef, TypeKind,
    Value, spv,
};
use std::collections::VecDeque;
use std::rc::Rc;

use super::SpvWellKnownWithExtras;

pub fn lower_physical_nulls(module: &mut Module) {
    let cx_rc = module.cx();
    let cx = &*cx_rc;
    let wk = &super::SpvSpecWithExtras::get().well_known;

    // Intern a `u64` type and a `u64` zero constant (`OpConstantNull` of an
    // integer scalar is permitted). The synthesised `OpConvertUToPtr` will
    // take this constant as its input; the `OpConvertPtrToU` widening
    // path reuses the same type and constant for comparison operands.
    let u64_ty = cx.intern(TypeDef {
        attrs: Default::default(),
        kind: TypeKind::SpvInst {
            spv_inst: spv::Inst {
                opcode: wk.OpTypeInt,
                imms: [
                    spv::Imm::Short(wk.LiteralInteger, 64),
                    spv::Imm::Short(wk.LiteralInteger, 0),
                ]
                .into_iter()
                .collect(),
            },
            type_and_const_inputs: SmallVec::new(),
        },
    });
    let u64_zero = cx.intern(ConstDef {
        attrs: Default::default(),
        ty: u64_ty,
        kind: ConstKind::SpvInst {
            spv_inst_and_const_inputs: Rc::new((
                spv::Inst {
                    opcode: wk.OpConstantNull,
                    imms: SmallVec::new(),
                },
                SmallVec::new(),
            )),
        },
    });

    let mut fixer = Lowerer {
        cx,
        wk,
        u64_ty,
        u64_zero,
        offending: FxHashMap::default(),
        seen_global_vars: FxHashSet::default(),
        global_var_queue: VecDeque::new(),
        seen_funcs: FxHashSet::default(),
        func_queue: VecDeque::new(),
        transformed_data_inst_forms: FxHashMap::default(),
        parent_block: None,
        widened_ptr_to_u: FxHashSet::default(),
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

/// Check whether `c` is an `OpConstantNull` whose result type is an
/// `OpTypePointer` in the `PhysicalStorageBuffer` storage class. Returns
/// the result type if so.
fn offending_null_pointer_type(
    cx: &Context,
    wk: &SpvWellKnownWithExtras,
    c: Const,
) -> Option<Type> {
    let const_def = &cx[c];
    let ConstKind::SpvInst {
        spv_inst_and_const_inputs,
    } = &const_def.kind
    else {
        return None;
    };
    let (spv_inst, _const_inputs) = &**spv_inst_and_const_inputs;
    if spv_inst.opcode != wk.OpConstantNull {
        return None;
    }
    let TypeKind::SpvInst {
        spv_inst: ty_inst, ..
    } = &cx[const_def.ty].kind
    else {
        return None;
    };
    if ty_inst.opcode != wk.OpTypePointer {
        return None;
    }
    let storage_class = match ty_inst.imms[..] {
        [spv::Imm::Short(_, sc)] => sc,
        _ => return None,
    };
    if storage_class != wk.PhysicalStorageBuffer {
        return None;
    }
    Some(const_def.ty)
}

struct Lowerer<'a> {
    cx: &'a Context,
    wk: &'static SpvWellKnownWithExtras,
    u64_ty: Type,
    u64_zero: Const,

    /// Cache of `OpConstantNull` constants we've classified, mapping to their
    /// (physical pointer) result type.
    offending: FxHashMap<Const, Option<Type>>,

    seen_global_vars: FxHashSet<GlobalVar>,
    global_var_queue: VecDeque<GlobalVar>,
    seen_funcs: FxHashSet<Func>,
    func_queue: VecDeque<Func>,
    transformed_data_inst_forms: FxHashMap<DataInstForm, Transformed<DataInstForm>>,

    /// Set while walking inside a `ControlNodeKind::Block`, so the inner
    /// `DataInst` visitor can locate the parent block to insert into.
    parent_block: Option<ControlNode>,

    /// `OpConvertPtrToU` data insts we've already widened to `u64` output.
    /// Comparison consumers reaching such an inst know its operand is `u64`.
    widened_ptr_to_u: FxHashSet<DataInst>,
}

impl Lowerer<'_> {
    fn classify(&mut self, c: Const) -> Option<Type> {
        if let Some(&cached) = self.offending.get(&c) {
            return cached;
        }
        let result = offending_null_pointer_type(self.cx, self.wk, c);
        self.offending.insert(c, result);
        result
    }
}

impl Transformer for Lowerer<'_> {
    fn transform_data_inst_form_use(
        &mut self,
        data_inst_form: DataInstForm,
    ) -> Transformed<DataInstForm> {
        // Recurse so that `FuncCall` inner uses queue their callees — the
        // default returns `Unchanged` without descending.
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

    fn in_place_transform_control_node_def(
        &mut self,
        mut func_at_control_node: FuncAtMut<'_, ControlNode>,
    ) {
        let old_parent = self.parent_block.take();
        if let ControlNodeKind::Block { .. } = func_at_control_node.reborrow().def().kind {
            self.parent_block = Some(func_at_control_node.position);
        }
        func_at_control_node.inner_in_place_transform_with(self);
        self.parent_block = old_parent;
    }

    fn in_place_transform_data_inst_def(&mut self, mut func_at_data_inst: FuncAtMut<'_, DataInst>) {
        let cx = self.cx;
        let wk = self.wk;

        func_at_data_inst
            .reborrow()
            .inner_in_place_transform_with(self);

        let target_inst = func_at_data_inst.position;

        // ── (1) Replace `Value::Const(offending_null)` inputs with a
        // function-local `OpConvertUToPtr 0`.
        let mut to_rewrite: SmallVec<[(usize, Type); 2]> = SmallVec::new();
        {
            let def = func_at_data_inst.reborrow().freeze().def();
            for (i, v) in def.inputs.iter().enumerate() {
                if let &Value::Const(c) = v
                    && let Some(ty) = self.classify(c)
                {
                    to_rewrite.push((i, ty));
                }
            }
        }
        if !to_rewrite.is_empty()
            && let Some(parent_block) = self.parent_block
        {
            let func = func_at_data_inst.reborrow().at(());
            for (idx, ptr_ty) in to_rewrite {
                let new_inst = func.data_insts.define(
                    cx,
                    DataInstDef {
                        attrs: Default::default(),
                        form: cx.intern(DataInstFormDef {
                            kind: DataInstKind::SpvInst(spv::Inst {
                                opcode: wk.OpConvertUToPtr,
                                imms: SmallVec::new(),
                            }),
                            output_type: Some(ptr_ty),
                        }),
                        inputs: [Value::Const(self.u64_zero)].into_iter().collect(),
                    }
                    .into(),
                );
                let ControlNodeKind::Block { insts } =
                    &mut func.control_nodes[parent_block].kind
                else {
                    unreachable!()
                };
                insts.insert_before(new_inst, target_inst, func.data_insts);
                func.data_insts[target_inst].inputs[idx] = Value::DataInstOutput(new_inst);
            }
        }

        // ── (2) Widen `OpConvertPtrToU` of a physical pointer to `u64`,
        // and patch the immediate comparison consumer's other operand if it
        // is a constant in a narrower integer type.
        let widen_this: Option<DataInstKind> = {
            let frozen = func_at_data_inst.reborrow().freeze();
            let def = frozen.def();
            let inst_form_def = &cx[def.form];
            match &inst_form_def.kind {
                DataInstKind::SpvInst(spv_inst) if spv_inst.opcode == wk.OpConvertPtrToU => {
                    let already_u64 = inst_form_def.output_type == Some(self.u64_ty);
                    let src_is_physical = def
                        .inputs
                        .first()
                        .map(|&v| {
                            let ty = frozen.at(()).at(v).type_of(cx);
                            match &cx[ty].kind {
                                TypeKind::SpvInst {
                                    spv_inst: pti, ..
                                } if pti.opcode == wk.OpTypePointer => matches!(
                                    pti.imms[..],
                                    [spv::Imm::Short(_, sc)] if sc == wk.PhysicalStorageBuffer
                                ),
                                _ => false,
                            }
                        })
                        .unwrap_or(false);
                    if !already_u64 && src_is_physical {
                        Some(inst_form_def.kind.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        if let Some(kind) = widen_this {
            // Widen the result type to `u64`.
            let new_form = cx.intern(DataInstFormDef {
                kind,
                output_type: Some(self.u64_ty),
            });
            func_at_data_inst.reborrow().def().form = new_form;
            self.widened_ptr_to_u.insert(target_inst);
        }

        // ── (3) If this is a comparison whose operands reference an
        // already-widened `OpConvertPtrToU`, widen any other constant
        // operands to `u64` so the operand types match.
        let is_compare = {
            let def = func_at_data_inst.reborrow().freeze().def();
            let inst_form_def = &cx[def.form];
            if let DataInstKind::SpvInst(spv_inst) = &inst_form_def.kind {
                [
                    wk.OpIEqual,
                    wk.OpINotEqual,
                    wk.OpUGreaterThan,
                    wk.OpUGreaterThanEqual,
                    wk.OpULessThan,
                    wk.OpULessThanEqual,
                ]
                .contains(&spv_inst.opcode)
            } else {
                false
            }
        };
        if !is_compare {
            return;
        }

        let any_widened_operand = {
            let def = func_at_data_inst.reborrow().freeze().def();
            def.inputs.iter().any(|v| {
                matches!(v, Value::DataInstOutput(d) if self.widened_ptr_to_u.contains(d))
            })
        };
        if !any_widened_operand {
            return;
        }

        // For each constant operand, intern a u64 zero replacement. We only
        // handle integer-zero constants (the null-check pattern emitted by
        // `Option<&T>` NPO matching); anything else stays untouched and
        // `spirv-val` will complain — which is what we want, since silent
        // truncation of arbitrary values would be incorrect.
        let to_patch: SmallVec<[usize; 2]> = {
            let def = func_at_data_inst.reborrow().freeze().def();
            def.inputs
                .iter()
                .enumerate()
                .filter_map(|(i, v)| match v {
                    &Value::Const(c) => {
                        let const_def = &cx[c];
                        if const_def.ty == self.u64_ty {
                            return None;
                        }
                        // Only patch zero constants of integer type.
                        let TypeKind::SpvInst {
                            spv_inst: ty_inst, ..
                        } = &cx[const_def.ty].kind
                        else {
                            return None;
                        };
                        if ty_inst.opcode != wk.OpTypeInt {
                            return None;
                        }
                        let ConstKind::SpvInst {
                            spv_inst_and_const_inputs,
                        } = &const_def.kind
                        else {
                            return None;
                        };
                        let (c_inst, _) = &**spv_inst_and_const_inputs;
                        let is_zero = match c_inst.opcode {
                            op if op == wk.OpConstantNull => true,
                            op if op == wk.OpConstant => {
                                c_inst.imms.iter().all(|imm| matches!(imm, spv::Imm::Short(_, 0)))
                            }
                            _ => false,
                        };
                        is_zero.then_some(i)
                    }
                    _ => None,
                })
                .collect()
        };

        for idx in to_patch {
            func_at_data_inst.reborrow().def().inputs[idx] = Value::Const(self.u64_zero);
        }
    }
}
