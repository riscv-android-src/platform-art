/*
 * Copyright (C) 2015 The Android Open Source Project
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

#include "calling_convention_riscv64.h"

#include <android-base/logging.h>

#include "arch/riscv64/jni_frame_riscv64.h"
#include "arch/instruction_set.h"
#include "handle_scope-inl.h"
#include "utils/riscv64/managed_register_riscv64.h"

namespace art {
namespace riscv64 {

static const GpuRegister kGpuArgumentRegisters[] = {
  A0, A1, A2, A3, A4, A5, A6, A7
};

static const FpuRegister kFpuArgumentRegisters[] = {
  FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7
};

static constexpr ManagedRegister kCalleeSaveRegisters[] = {
    // Hard float callee saves.
    Riscv64ManagedRegister::FromFpuRegister(FS0),
    Riscv64ManagedRegister::FromFpuRegister(FS1),
    Riscv64ManagedRegister::FromFpuRegister(FS2),
    Riscv64ManagedRegister::FromFpuRegister(FS3),
    Riscv64ManagedRegister::FromFpuRegister(FS4),
    Riscv64ManagedRegister::FromFpuRegister(FS5),
    Riscv64ManagedRegister::FromFpuRegister(FS6),
    Riscv64ManagedRegister::FromFpuRegister(FS7),
    Riscv64ManagedRegister::FromFpuRegister(FS8),
    Riscv64ManagedRegister::FromFpuRegister(FS9),
    Riscv64ManagedRegister::FromFpuRegister(FS10),
    Riscv64ManagedRegister::FromFpuRegister(FS11),

    // Core registers.
    Riscv64ManagedRegister::FromGpuRegister(S2),
    Riscv64ManagedRegister::FromGpuRegister(S3),
    Riscv64ManagedRegister::FromGpuRegister(S4),
    Riscv64ManagedRegister::FromGpuRegister(S5),
    Riscv64ManagedRegister::FromGpuRegister(S6),
    Riscv64ManagedRegister::FromGpuRegister(S7),
    Riscv64ManagedRegister::FromGpuRegister(S8),
    Riscv64ManagedRegister::FromGpuRegister(S9),
    Riscv64ManagedRegister::FromGpuRegister(S10),
    Riscv64ManagedRegister::FromGpuRegister(S0),
};

static constexpr ManagedRegister kCriticalCalleeSaveRegisters[] = {
    Riscv64ManagedRegister::FromGpuRegister(RA)
};

static constexpr uint32_t CalculateCoreCalleeSpillMask() {
  // RA is a special callee save which is not reported by CalleeSaveRegisters().
  uint32_t result = 1 << RA;
  for (auto&& r : kCalleeSaveRegisters) {
    if (r.AsRiscv64().IsGpuRegister()) {
      result |= (1 << r.AsRiscv64().AsGpuRegister());
    }
  }
  return result;
}

static constexpr uint32_t CalculateFpCalleeSpillMask() {
  uint32_t result = 0;
  for (auto&& r : kCalleeSaveRegisters) {
    if (r.AsRiscv64().IsFpuRegister()) {
      result |= (1 << r.AsRiscv64().AsFpuRegister());
    }
  }
  return result;
}

static constexpr uint32_t kCoreCalleeSpillMask = CalculateCoreCalleeSpillMask();
static constexpr uint32_t kFpCalleeSpillMask = CalculateFpCalleeSpillMask();

static ManagedRegister ReturnRegisterForShorty(const char* shorty) {
  if (shorty[0] == 'F' || shorty[0] == 'D') {
    return Riscv64ManagedRegister::FromFpuRegister(FA0);
  } else if (shorty[0] == 'V') {
    return Riscv64ManagedRegister::NoRegister();
  } else {
    return Riscv64ManagedRegister::FromGpuRegister(A0);
  }
}

ManagedRegister Riscv64ManagedRuntimeCallingConvention::ReturnRegister() {
  return ReturnRegisterForShorty(GetShorty());
}

ManagedRegister Riscv64JniCallingConvention::ReturnRegister() {
  return ReturnRegisterForShorty(GetShorty());
}

ManagedRegister Riscv64JniCallingConvention::IntReturnRegister() {
  return Riscv64ManagedRegister::FromGpuRegister(A0);
}

// Managed runtime calling convention

ManagedRegister Riscv64ManagedRuntimeCallingConvention::MethodRegister() {
  return Riscv64ManagedRegister::FromGpuRegister(A0);
}

bool Riscv64ManagedRuntimeCallingConvention::IsCurrentParamInRegister() {
  unsigned int gp_args = itr_args_ - itr_float_and_doubles_;
  if (IsCurrentParamAFloatOrDouble()) {
    unsigned int can_spilled_to_gp = 0;
    if (gp_args < (8 - 1u /* method */))
      can_spilled_to_gp += (8 - gp_args - 1u /* method */);
    return (itr_float_and_doubles_ < (kMaxFloatOrDoubleRegisterArguments + can_spilled_to_gp));
  } else {
    unsigned int spilled_to_gp = 0;
    if (gp_args < (8 - 1u /* method */))
      spilled_to_gp += (itr_float_and_doubles_ >= kMaxFloatOrDoubleRegisterArguments) ?
                       (itr_float_and_doubles_ - kMaxFloatOrDoubleRegisterArguments) : 0;
    return ((/* method */ 1u + gp_args + spilled_to_gp) < kMaxIntLikeRegisterArguments);
  }
}

bool Riscv64ManagedRuntimeCallingConvention::IsCurrentParamOnStack() {
  return !IsCurrentParamInRegister();
}

ManagedRegister Riscv64ManagedRuntimeCallingConvention::CurrentParamRegister() {
  CHECK(IsCurrentParamInRegister());
  int gp_reg = itr_args_ - itr_float_and_doubles_;

  if (IsCurrentParamAFloatOrDouble()) {
    // CHECK_LT(itr_float_and_doubles_, kMaxFloatOrDoubleRegisterArguments);
    if (itr_float_and_doubles_ < kMaxFloatOrDoubleRegisterArguments)
      return Riscv64ManagedRegister::FromFpuRegister(kFpuArgumentRegisters[itr_float_and_doubles_]);
    else {
      unsigned int spilled_to_gp = itr_float_and_doubles_ - kMaxFloatOrDoubleRegisterArguments;
      CHECK_LT(static_cast<unsigned int>(/* method */ 1u + gp_reg+spilled_to_gp), kMaxIntLikeRegisterArguments);
      return Riscv64ManagedRegister::FromGpuRegister(kGpuArgumentRegisters[/* method */ 1u + gp_reg + spilled_to_gp]);
    }
  } else {
    unsigned int spilled_to_gp = 0;
    spilled_to_gp += (itr_float_and_doubles_ >= kMaxFloatOrDoubleRegisterArguments) ?
                     (itr_float_and_doubles_ - kMaxFloatOrDoubleRegisterArguments) : 0;
    CHECK_LT(static_cast<unsigned int>(/* method */ 1u + gp_reg+spilled_to_gp), kMaxIntLikeRegisterArguments);
    return Riscv64ManagedRegister::FromGpuRegister(kGpuArgumentRegisters[/* method */ 1u + gp_reg+spilled_to_gp]);
  }
}

FrameOffset Riscv64ManagedRuntimeCallingConvention::CurrentParamStackOffset() {
  return FrameOffset(displacement_.Int32Value() +  // displacement
                     kFramePointerSize +  // Method ref
                     (itr_slots_ * sizeof(uint32_t)));  // offset into in args
}

// JNI calling convention

Riscv64JniCallingConvention::Riscv64JniCallingConvention(bool is_static,
                                                       bool is_synchronized,
                                                       bool is_critical_native,
                                                       const char* shorty)
    : JniCallingConvention(is_static,
                           is_synchronized,
                           is_critical_native,
                           shorty,
                           kRiscv64PointerSize) {
}

uint32_t Riscv64JniCallingConvention::CoreSpillMask() const {
  return is_critical_native_ ? 0u : kCoreCalleeSpillMask;
}

uint32_t Riscv64JniCallingConvention::FpSpillMask() const {
  return is_critical_native_ ? 0u : kFpCalleeSpillMask;
}

ManagedRegister Riscv64JniCallingConvention::ReturnScratchRegister() const {
  return ManagedRegister::NoRegister();
}

size_t Riscv64JniCallingConvention::FrameSize() const {
  if (is_critical_native_) {
    CHECK(!SpillsMethod());
    CHECK(!HasLocalReferenceSegmentState());
    CHECK(!SpillsReturnValue());
    return 0u;  // There is no managed frame for @CriticalNative.
  }

  // Method*, callee save area size, local reference segment state
  DCHECK(SpillsMethod());
  // ArtMethod*, RA and callee save area size, local reference segment state.
  size_t method_ptr_size = static_cast<size_t>(kFramePointerSize);
  size_t ra_and_callee_save_area_size = (CalleeSaveRegisters().size() + 1) * kFramePointerSize;

  if (IsCriticalNative())
    ra_and_callee_save_area_size -= kFramePointerSize;

  size_t total_size = method_ptr_size + ra_and_callee_save_area_size;

  DCHECK(HasLocalReferenceSegmentState());
  // Cookie is saved in one of the spilled registers.

  // Plus return value spill area size
  if (SpillsReturnValue()) {
    // No padding between the method pointer and the return value on arm64.
    DCHECK_EQ(ReturnValueSaveLocation().SizeValue(), method_ptr_size);
    total_size += SizeOfReturnValue();
  }

  return RoundUp(total_size, kStackAlignment);
}

size_t Riscv64JniCallingConvention::OutFrameSize() const {
  // Count param args, including JNIEnv* and jclass*.
  size_t all_args = NumberOfExtraArgumentsForJni() + NumArgs();
  size_t num_fp_args = NumFloatOrDoubleArgs();
  DCHECK_GE(all_args, num_fp_args);
  size_t num_non_fp_args = all_args - num_fp_args;
  // The size of outgoing arguments.
  size_t size = GetNativeOutArgsSize(num_fp_args, num_non_fp_args);

  // For @CriticalNative, we can make a tail call if there are no stack args and
  // we do not need to extend the result. Otherwise, add space for return PC.
  // if (is_critical_native_ && (size != 0u || RequiresSmallResultTypeExtension())) {
  //  size += kFramePointerSize;  // We need to spill RA with the args.
  // }

  if (UNLIKELY(IsCriticalNative())) {
    size += kFramePointerSize;  // We need to spill RA with the args.
  }

  size_t out_args_size = RoundUp(size, kRiscv64StackAlignment);
  if (UNLIKELY(IsCriticalNative())) {
    DCHECK_EQ(out_args_size, GetCriticalNativeStubFrameSize(GetShorty(), NumArgs() + 1u));
  }
  return out_args_size;
}

ArrayRef<const ManagedRegister> Riscv64JniCallingConvention::CalleeSaveRegisters() const {
  if (UNLIKELY(IsCriticalNative())) {
    if (UseTailCall()) {
      return ArrayRef<const ManagedRegister>();  // Do not spill anything.
    } else {
      // Spill RA with out args.
      return ArrayRef<const ManagedRegister>(kCriticalCalleeSaveRegisters);
    }
  } else {
    return ArrayRef<const ManagedRegister>(kCalleeSaveRegisters);
  }
}

bool Riscv64JniCallingConvention::IsCurrentParamInRegister() {
  unsigned int gp_args = itr_args_ - itr_float_and_doubles_;
  if (IsCurrentParamAFloatOrDouble()) {
    unsigned int can_spilled_to_gp = 0;
    if (gp_args < 8)
      can_spilled_to_gp += (8 - gp_args);
    return (itr_float_and_doubles_ < (kMaxFloatOrDoubleRegisterArguments + can_spilled_to_gp));
  } else {
    unsigned int spilled_to_gp = 0;
    if (gp_args < 8)
      spilled_to_gp += (itr_float_and_doubles_ >= kMaxFloatOrDoubleRegisterArguments) ?
                       (itr_float_and_doubles_ - kMaxFloatOrDoubleRegisterArguments) : 0;
    return ((gp_args + spilled_to_gp) < kMaxIntLikeRegisterArguments);
  }
  // TODO: Can we just call CurrentParamRegister to figure this out?
}

bool Riscv64JniCallingConvention::IsCurrentParamOnStack() {
  return !IsCurrentParamInRegister();
}

ManagedRegister Riscv64JniCallingConvention::CurrentParamRegister() {
  CHECK(IsCurrentParamInRegister());
  int gp_reg = itr_args_ - itr_float_and_doubles_;

  if (IsCurrentParamAFloatOrDouble()) {
    // CHECK_LT(itr_float_and_doubles_, kMaxFloatOrDoubleRegisterArguments);
    if (itr_float_and_doubles_ < kMaxFloatOrDoubleRegisterArguments)
      return Riscv64ManagedRegister::FromFpuRegister(kFpuArgumentRegisters[itr_float_and_doubles_]);
    else {
      unsigned int spilled_to_gp = itr_float_and_doubles_ - kMaxFloatOrDoubleRegisterArguments;
      CHECK_LT(static_cast<unsigned int>(gp_reg+spilled_to_gp), kMaxIntLikeRegisterArguments);
      return Riscv64ManagedRegister::FromGpuRegister(kGpuArgumentRegisters[gp_reg + spilled_to_gp]);
    }
  } else {
    unsigned int spilled_to_gp = 0;
    spilled_to_gp += (itr_float_and_doubles_ >= kMaxFloatOrDoubleRegisterArguments) ?
                     (itr_float_and_doubles_ - kMaxFloatOrDoubleRegisterArguments) : 0;
    CHECK_LT(static_cast<unsigned int>(gp_reg+spilled_to_gp), kMaxIntLikeRegisterArguments);
    return Riscv64ManagedRegister::FromGpuRegister(kGpuArgumentRegisters[gp_reg+spilled_to_gp]);
  }
}

FrameOffset Riscv64JniCallingConvention::CurrentParamStackOffset() {
  CHECK(IsCurrentParamOnStack());

  unsigned int gp_args = itr_args_ - itr_float_and_doubles_;
  unsigned int can_spilled_to_gp = 0;
  size_t args_on_stack = itr_args_ - std::min(kMaxIntLikeRegisterArguments, static_cast<size_t>(gp_args));
  if (gp_args < 8)
    can_spilled_to_gp += (8 - gp_args);
  
  args_on_stack -= std::min(static_cast<size_t>(kMaxFloatOrDoubleRegisterArguments + can_spilled_to_gp),
                            static_cast<size_t>(itr_float_and_doubles_));

  size_t offset = displacement_.Int32Value() - OutFrameSize() + (args_on_stack * kFramePointerSize);
  CHECK_LT(offset, OutFrameSize());

  return FrameOffset(offset);
  // XC-ART-TODO: Seems identical to X86_64 code.
}

ManagedRegister Riscv64JniCallingConvention::SavedLocalReferenceCookieRegister() const {
  // XC-ART-TBD: double check selected S10 in the future. 

  // The S10 is callee-save register in both managed and native ABIs.
  // It is saved in the stack frame and it has no special purpose like `tr`.
  static_assert((kCoreCalleeSpillMask & (1u << S10)) != 0u);  // Managed callee save register.
  return Riscv64ManagedRegister::FromGpuRegister(S10);
}

ManagedRegister Riscv64JniCallingConvention::HiddenArgumentRegister() const {
  CHECK(IsCriticalNative());
  // XC-ART-TBD: double check selected T0 in the future.
 
  // T0 is neither managed callee-save, nor argument register, nor scratch register.
  //XC-ART- TODO: Change to static_assert; std::none_of should be constexpr since C++20.

  return Riscv64ManagedRegister::FromGpuRegister(T0);
}

// Whether to use tail call (used only for @CriticalNative).
bool Riscv64JniCallingConvention::UseTailCall() const {
  CHECK(IsCriticalNative());
  // return OutFrameSize() == 0u;
  return false;
}

}  // namespace riscv64
}  // namespace art
