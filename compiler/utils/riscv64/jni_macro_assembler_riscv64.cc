/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "jni_macro_assembler_riscv64.h"

#include "entrypoints/quick/quick_entrypoints.h"
#include "managed_register_riscv64.h"
#include "offsets.h"
#include "thread.h"

namespace art {
namespace riscv64 {

#define __ asm_.

Riscv64JNIMacroAssembler::~Riscv64JNIMacroAssembler() {
}

void Riscv64JNIMacroAssembler::FinalizeCode() {
  __ FinalizeCode();
}

void Riscv64JNIMacroAssembler::GetCurrentThread(ManagedRegister tr) {
  __ GetCurrentThread(tr);
}

void Riscv64JNIMacroAssembler::GetCurrentThread(FrameOffset offset) {
  __ GetCurrentThread(offset, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

// See Riscv64 PCS Section 5.2.2.1.
void Riscv64JNIMacroAssembler::IncreaseFrameSize(size_t adjust) {
  __ IncreaseFrameSize(adjust);
}

// See Riscv64 PCS Section 5.2.2.1.
void Riscv64JNIMacroAssembler::DecreaseFrameSize(size_t adjust) {
  __ DecreaseFrameSize(adjust);
}

void Riscv64JNIMacroAssembler::Store(FrameOffset offs, ManagedRegister m_src, size_t size) {
  __ Store(offs, m_src, size);
}

void Riscv64JNIMacroAssembler::StoreRef(FrameOffset offs, ManagedRegister m_src) {
  __ StoreRef(offs, m_src);
}

void Riscv64JNIMacroAssembler::StoreRawPtr(FrameOffset offs, ManagedRegister m_src) {
  __ StoreRawPtr(offs, m_src);
}

void Riscv64JNIMacroAssembler::StoreImmediateToFrame(FrameOffset offs,
                                                   uint32_t imm) {
  __ StoreImmediateToFrame(offs, imm, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

void Riscv64JNIMacroAssembler::StoreStackOffsetToThread(ThreadOffset64 tr_offs,
                                                      FrameOffset fr_offs) {
  __ StoreStackOffsetToThread(tr_offs, fr_offs, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

void Riscv64JNIMacroAssembler::StoreStackPointerToThread(ThreadOffset64 tr_offs) {
  __ StoreStackPointerToThread(tr_offs);
}

void Riscv64JNIMacroAssembler::StoreSpanning(FrameOffset dest_off,
                                           ManagedRegister m_source,
                                           FrameOffset in_off) {
  __ StoreSpanning(dest_off, m_source, in_off, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

void Riscv64JNIMacroAssembler::Load(ManagedRegister m_dst, FrameOffset src, size_t size) {
  __ Load(m_dst, src, size);
}

void Riscv64JNIMacroAssembler::LoadFromThread(ManagedRegister m_dst,
                                            ThreadOffset64 src,
                                            size_t size) {
  __ LoadFromThread(m_dst, src, size);
}

void Riscv64JNIMacroAssembler::LoadRef(ManagedRegister m_dst, FrameOffset offs) {
  __ LoadRef(m_dst, offs);
}

void Riscv64JNIMacroAssembler::LoadRef(ManagedRegister m_dst,
                                     ManagedRegister m_base,
                                     MemberOffset offs,
                                     bool unpoison_reference) {
  __ LoadRef(m_dst, m_base, offs, unpoison_reference);
}

void Riscv64JNIMacroAssembler::LoadRawPtr(ManagedRegister m_dst,
                                        ManagedRegister m_base,
                                        Offset offs) {
  __ LoadRawPtr(m_dst, m_base, offs);
}

void Riscv64JNIMacroAssembler::LoadRawPtrFromThread(ManagedRegister m_dst, ThreadOffset64 offs) {
  __ LoadRawPtrFromThread(m_dst, offs);
}

// Copying routines.
void Riscv64JNIMacroAssembler::MoveArguments(ArrayRef<ArgumentLocation> dests,
                                           ArrayRef<ArgumentLocation> srcs) {
  DCHECK_EQ(dests.size(), srcs.size());
  auto get_mask = [](ManagedRegister reg) -> uint64_t {
    Riscv64ManagedRegister riscv64_reg = reg.AsRiscv64();
    if (riscv64_reg.IsGpuRegister()) {
      size_t core_reg_number = static_cast<size_t>(riscv64_reg.AsGpuRegister());
      DCHECK_LT(core_reg_number, 32u);
      return UINT64_C(1) << core_reg_number;
    } else {
      DCHECK(riscv64_reg.IsFpuRegister());
      size_t fp_reg_number = static_cast<size_t>(riscv64_reg.AsFpuRegister());
      DCHECK_LT(fp_reg_number, 32u);
      return (UINT64_C(1) << 32u) << fp_reg_number;
    }
  };

  // Collect registers to move while storing/copying args to stack slots.
  uint64_t src_regs = 0u;
  uint64_t dest_regs = 0u;
  for (size_t i = 0, arg_count = srcs.size(); i != arg_count; ++i) {
    const ArgumentLocation& src = srcs[i];
    const ArgumentLocation& dest = dests[i];
    DCHECK_EQ(src.GetSize(), dest.GetSize());
    if (dest.IsRegister()) {
      if (src.IsRegister() && src.GetRegister().Equals(dest.GetRegister())) {
        // Nothing to do.
      } else {
        if (src.IsRegister()) {
          src_regs |= get_mask(src.GetRegister());
        }
        dest_regs |= get_mask(dest.GetRegister());
      }
    } else {
      if (src.IsRegister()) {
        Store(dest.GetFrameOffset(), src.GetRegister(), dest.GetSize());
      } else {
        Copy(dest.GetFrameOffset(), src.GetFrameOffset(), dest.GetSize());
      }
    }
  }
  // Fill destination registers.
  // There should be no cycles, so this simple algorithm should make progress.
  while (dest_regs != 0u) {
    uint64_t old_dest_regs = dest_regs;
    for (size_t i = 0, arg_count = srcs.size(); i != arg_count; ++i) {
      const ArgumentLocation& src = srcs[i];
      const ArgumentLocation& dest = dests[i];
      if (!dest.IsRegister()) {
        continue;  // Stored in first loop above.
      }
      uint64_t dest_reg_mask = get_mask(dest.GetRegister());
      if ((dest_reg_mask & dest_regs) == 0u) {
        continue;  // Equals source, or already filled in one of previous iterations.
      }
      if ((dest_reg_mask & src_regs) != 0u) {
        continue;  // Cannot clobber this register yet.
      }
      if (src.IsRegister()) {
        Move(dest.GetRegister(), src.GetRegister(), dest.GetSize());
        src_regs &= ~get_mask(src.GetRegister());  // Allow clobbering source register.
      } else {
        Load(dest.GetRegister(), src.GetFrameOffset(), dest.GetSize());
      }
      dest_regs &= ~get_mask(dest.GetRegister());  // Destination register was filled.
    }
    CHECK_NE(old_dest_regs, dest_regs);
    DCHECK_EQ(0u, dest_regs & ~old_dest_regs);
  }
}

void Riscv64JNIMacroAssembler::Move(ManagedRegister m_dst, ManagedRegister m_src, size_t size) {
  __ Move(m_dst, m_src, size);
}

void Riscv64JNIMacroAssembler::CopyRawPtrFromThread(FrameOffset fr_offs,
                                                  ThreadOffset64 tr_offs) {
  __ CopyRawPtrFromThread(fr_offs, tr_offs, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

void Riscv64JNIMacroAssembler::CopyRawPtrToThread(ThreadOffset64 tr_offs,
                                                FrameOffset fr_offs,
                                                ManagedRegister m_scratch) {
  __ CopyRawPtrToThread(tr_offs, fr_offs, m_scratch);
}

void Riscv64JNIMacroAssembler::CopyRef(FrameOffset dest, FrameOffset src) {
  __ CopyRef(dest, src, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

void Riscv64JNIMacroAssembler::CopyRef(FrameOffset dest,
                                     ManagedRegister base,
                                     MemberOffset offs,
                                     bool unpoison_reference) {
  GpuRegister scratch = TMP;
  __ Addiu64(scratch, base.AsRiscv64().AsGpuRegister(), offs.Int32Value());
  __ Ld(scratch, scratch, 0);
  if (unpoison_reference) {
    asm_.MaybeUnpoisonHeapReference(scratch);
  }
  __ Addiu64(T6, SP, dest.Int32Value());
  __ Sd(scratch, T6, 0);
}

void Riscv64JNIMacroAssembler::Copy(FrameOffset dest,
                                  FrameOffset src,
                                  size_t size) {
  __ Copy(dest, src, Riscv64ManagedRegister::FromGpuRegister(TMP2), size);
}

void Riscv64JNIMacroAssembler::Copy(FrameOffset dest,
                                  ManagedRegister src_base,
                                  Offset src_offset,
                                  ManagedRegister m_scratch,
                                  size_t size) {
  __ Copy(dest, src_base, src_offset, m_scratch, size);
}

void Riscv64JNIMacroAssembler::Copy(ManagedRegister m_dest_base,
                                  Offset dest_offs,
                                  FrameOffset src,
                                  ManagedRegister m_scratch,
                                  size_t size) {
  __ Copy(m_dest_base, dest_offs, src, m_scratch, size);
}

void Riscv64JNIMacroAssembler::Copy(FrameOffset dst,
                                  FrameOffset src_base,
                                  Offset src_offset,
                                  ManagedRegister mscratch,
                                  size_t size) {
  __ Copy(dst, src_base, src_offset, mscratch, size);
}

void Riscv64JNIMacroAssembler::Copy(ManagedRegister m_dest,
                                  Offset dest_offset,
                                  ManagedRegister m_src,
                                  Offset src_offset,
                                  ManagedRegister m_scratch,
                                  size_t size) {
  __ Copy(m_dest, dest_offset, m_src, src_offset, m_scratch, size);
}

void Riscv64JNIMacroAssembler::Copy(FrameOffset dst,
                                  Offset dest_offset,
                                  FrameOffset src,
                                  Offset src_offset,
                                  ManagedRegister scratch,
                                  size_t size) {
  __ Copy(dst, dest_offset, src, src_offset, scratch, size);
}

void Riscv64JNIMacroAssembler::MemoryBarrier(ManagedRegister m_scratch) {
  __ MemoryBarrier(m_scratch);
}

void Riscv64JNIMacroAssembler::SignExtend(ManagedRegister mreg, size_t size) {
  __ SignExtend(mreg, size);
}

void Riscv64JNIMacroAssembler::ZeroExtend(ManagedRegister mreg, size_t size) {
  __ ZeroExtend(mreg, size);
}

void Riscv64JNIMacroAssembler::VerifyObject(ManagedRegister m_src, bool could_be_null) {
  // not validating references?
  __ VerifyObject(m_src, could_be_null);
}

void Riscv64JNIMacroAssembler::VerifyObject(FrameOffset src, bool could_be_null) {
  // not validating references?
  __ VerifyObject(src, could_be_null);
}

void Riscv64JNIMacroAssembler::Call(ManagedRegister m_base, Offset offs) {
  __ Call(m_base, offs, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

void Riscv64JNIMacroAssembler::Call(FrameOffset base, Offset offs) {
  __ Call(base, offs, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

void Riscv64JNIMacroAssembler::CallFromThread(ThreadOffset64 offset) {
  __ CallFromThread(offset, Riscv64ManagedRegister::FromGpuRegister(TMP2));
}

void Riscv64JNIMacroAssembler::CreateHandleScopeEntry(ManagedRegister m_out_reg,
                                                    FrameOffset handle_scope_offs,
                                                    ManagedRegister m_in_reg,
                                                    bool null_allowed) {
  __ CreateHandleScopeEntry(m_out_reg, handle_scope_offs, m_in_reg, null_allowed);
}

void Riscv64JNIMacroAssembler::CreateHandleScopeEntry(FrameOffset out_off,
                                                    FrameOffset handle_scope_offset,
                                                    ManagedRegister m_scratch,
                                                    bool null_allowed) {
  __ CreateHandleScopeEntry(out_off, handle_scope_offset, m_scratch, null_allowed);
}

void Riscv64JNIMacroAssembler::LoadReferenceFromHandleScope(ManagedRegister m_out_reg,
                                                          ManagedRegister m_in_reg) {
  __ LoadReferenceFromHandleScope(m_out_reg, m_in_reg);
}

void Riscv64JNIMacroAssembler::CreateJObject(ManagedRegister m_out_reg,
                                           FrameOffset spilled_reference_offset,
                                           ManagedRegister m_in_reg,
                                           bool null_allowed) {
  Riscv64ManagedRegister out_reg = m_out_reg.AsRiscv64();
  Riscv64ManagedRegister in_reg = m_in_reg.AsRiscv64();
  // For now we only hold stale handle scope entries in x registers.
  CHECK(in_reg.IsNoRegister() || in_reg.IsGpuRegister()) << in_reg;
  CHECK(out_reg.IsGpuRegister()) << out_reg;
  if (null_allowed) {
    // Null values get a jobject value null. Otherwise, the jobject is
    // the address of the spilled reference.
    // e.g. out_reg = (in == 0) ? 0 : (SP+spilled_reference_offset)
    if (in_reg.IsNoRegister()) {
      __ Addiu64(out_reg.AsGpuRegister(), SP, spilled_reference_offset.Int32Value());
      __ Lw(out_reg.AsGpuRegister(), out_reg.AsGpuRegister(), 0);

      in_reg = out_reg;
    }
    // ___ Cmp(reg_w(in_reg.AsOverlappingWRegister()), 0);
    if (!out_reg.Equals(in_reg)) {
      riscv64::Riscv64Label non_null_arg;
      __ Bnezc(in_reg.AsGpuRegister(), &non_null_arg);
      __ Move(out_reg.AsGpuRegister(), ZERO);;
      __ Bind(&non_null_arg);
    }
    // AddConstant(out_reg.AsXRegister(), SP, spilled_reference_offset.Int32Value(), ne);
    riscv64::Riscv64Label null_arg;
    __ Beqzc(in_reg.AsGpuRegister(), &null_arg);
    __ Addiu64(out_reg.AsGpuRegister(), SP, spilled_reference_offset.Int32Value());
    __ Bind(&null_arg);
  } else {
    // AddConstant(out_reg.AsXRegister(), SP, spilled_reference_offset.Int32Value(), al);
    __ Addiu64(out_reg.AsGpuRegister(), SP, spilled_reference_offset.Int32Value());
  }
}

void Riscv64JNIMacroAssembler::CreateJObject(FrameOffset out_off,
                                           FrameOffset spilled_reference_offset,
                                           bool null_allowed) {
  GpuRegister scratch = TMP;
  if (null_allowed) {
    // Null values get a jobject value null. Otherwise, the jobject is
    // the address of the spilled reference.
    // e.g. scratch = (scratch == 0) ? 0 : (SP+spilled_reference_offset)
    riscv64::Riscv64Label null_arg;
    __ Addiu64(scratch, SP, spilled_reference_offset.Int32Value());
    __ Lw(scratch, scratch, 0);
    __ Beqzc(scratch, &null_arg);
    __ Addiu64(scratch, SP, spilled_reference_offset.Int32Value());
    __ Bind(&null_arg);
  } else {
    __ Addiu64(scratch, SP, spilled_reference_offset.Int32Value());
  }
  __ Addiu64(T6, SP, out_off.Int32Value());
  __ Sd(scratch, T6, 0);
}

void Riscv64JNIMacroAssembler::ExceptionPoll(size_t stack_adjust) {
  __ ExceptionPoll(Riscv64ManagedRegister::FromGpuRegister(TMP2), stack_adjust);
}

std::unique_ptr<JNIMacroLabel> Riscv64JNIMacroAssembler::CreateLabel() {
  return std::unique_ptr<JNIMacroLabel>(new Riscv64JNIMacroLabel());
}

void Riscv64JNIMacroAssembler::Jump(JNIMacroLabel* label) {
  CHECK(label != nullptr);
  __ Bc(down_cast<Riscv64Label*>(Riscv64JNIMacroLabel::Cast(label)->AsRiscv64()));
}

void Riscv64JNIMacroAssembler::Jump(ManagedRegister m_base, Offset offs) {
  Riscv64ManagedRegister base = m_base.AsRiscv64();
  CHECK(base.IsGpuRegister()) << base;
  GpuRegister scratch = TMP;
  __ Addiu64(scratch, base.AsGpuRegister(), offs.Int32Value());
  __ Ld(scratch, scratch, 0);
  __ Jr(scratch);
}

void Riscv64JNIMacroAssembler::Bind(JNIMacroLabel* label) {
  CHECK(label != nullptr);
  __ Bind(Riscv64JNIMacroLabel::Cast(label)->AsRiscv64());
}

void Riscv64JNIMacroAssembler::BuildFrame(size_t frame_size,
                                        ManagedRegister method_reg,
                                        ArrayRef<const ManagedRegister> callee_save_regs) {
  __ BuildFrame(frame_size, method_reg, callee_save_regs);
}

void Riscv64JNIMacroAssembler::RemoveFrame(size_t frame_size,
                                         ArrayRef<const ManagedRegister> callee_save_regs,
                                         bool may_suspend) {
  __ RemoveFrame(frame_size, callee_save_regs, may_suspend);
}

void Riscv64JNIMacroAssembler::TestGcMarking(JNIMacroLabel* label, JNIMacroUnaryCondition cond) {
  CHECK(label != nullptr);

  DCHECK_EQ(Thread::IsGcMarkingSize(), 4u);
  DCHECK(kUseReadBarrier);

  GpuRegister test_reg = TMP;

  int32_t is_gc_marking_offset = Thread::IsGcMarkingOffset<kArm64PointerSize>().Int32Value();
  __ Addiu64(test_reg, TR, is_gc_marking_offset);
  __ Ld(test_reg, test_reg, 0);

  switch (cond) {
    case JNIMacroUnaryCondition::kZero:
      __ Beqzc(test_reg, down_cast<Riscv64Label*>(Riscv64JNIMacroLabel::Cast(label)->AsRiscv64()));
      break;
    case JNIMacroUnaryCondition::kNotZero:
      __ Bnezc(test_reg, down_cast<Riscv64Label*>(Riscv64JNIMacroLabel::Cast(label)->AsRiscv64()));
      break;
    default:
      LOG(FATAL) << "Not implemented unary condition: " << static_cast<int>(cond);
      UNREACHABLE();
  }
}

#undef ___

}  // namespace riscv64
}  // namespace art
