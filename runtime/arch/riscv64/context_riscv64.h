/*
 * Copyright (C) 2014 The Android Open Source Project
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

#ifndef ART_RUNTIME_ARCH_RISCV64_CONTEXT_RISCV64_H_
#define ART_RUNTIME_ARCH_RISCV64_CONTEXT_RISCV64_H_

#include <android-base/logging.h>

#include "arch/context.h"
#include "base/macros.h"
#include "registers_riscv64.h"

namespace art {
namespace riscv64 {

class Riscv64Context : public Context {
 public:
  Riscv64Context() {
    Reset();
  }
  virtual ~Riscv64Context() {}

  void Reset() override;

  void FillCalleeSaves(uint8_t* frame, const QuickMethodFrameInfo& fr) override;

  void SetSP(uintptr_t new_sp) override {
    SetGPR(SP, new_sp);
  }

  // Wendong: TBD performance issue
  void SetPC(uintptr_t new_pc) override {
    SetGPR(T6, new_pc);
  }

  void SetNterpDexPC(uintptr_t dex_pc_ptr) override {
    DCHECK(0);  // Wendong: TBD reg？
    SetGPR(T3, dex_pc_ptr);
  }

  void SetArg0(uintptr_t new_arg0_value) override {
    SetGPR(A0, new_arg0_value);
  }

  bool IsAccessibleGPR(uint32_t reg) override {
    DCHECK_LT(reg, arraysize(gprs_));
    return gprs_[reg] != nullptr;
  }

  uintptr_t* GetGPRAddress(uint32_t reg) override {
    DCHECK_LT(reg, arraysize(gprs_));
    return gprs_[reg];
  }

  uintptr_t GetGPR(uint32_t reg) override {
    CHECK_LT(reg, static_cast<uint32_t>(kNumberOfGpuRegisters));
    DCHECK(IsAccessibleGPR(reg));
    return *gprs_[reg];
  }

  void SetGPR(uint32_t reg, uintptr_t value) override;

  bool IsAccessibleFPR(uint32_t reg) override {
    CHECK_LT(reg, static_cast<uint32_t>(kNumberOfFpuRegisters));
    return fprs_[reg] != nullptr;
  }

  uintptr_t GetFPR(uint32_t reg) override {
    CHECK_LT(reg, static_cast<uint32_t>(kNumberOfFpuRegisters));
    DCHECK(IsAccessibleFPR(reg));
    return *fprs_[reg];
  }

  void SetFPR(uint32_t reg, uintptr_t value) override;

  void SmashCallerSaves() override;
  NO_RETURN void DoLongJump() override;


  static constexpr size_t rPC = kNumberOfGpuRegisters;
 private:
  // Pointers to register stack locations, initialized to null or the specific registers below. We need
  // an additional one for the PC.
  uintptr_t* gprs_[kNumberOfGpuRegisters + 1];
  uint64_t* fprs_[kNumberOfFpuRegisters];
  // Hold values for sp and pc if they are not located within a stack frame. 
  // We use t9 for the PC (as ra is required to be valid for single-frame deopt and must not be clobbered). 
  // We also need the first argument for single-frame deopt.
  uintptr_t sp_, pc_, arg0_;
};

}  // namespace riscv64
}  // namespace art

#endif  // ART_RUNTIME_ARCH_RISCV64_CONTEXT_RISCV64_H_
