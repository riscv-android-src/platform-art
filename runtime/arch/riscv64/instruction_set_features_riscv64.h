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

#ifndef ART_RUNTIME_ARCH_RISCV64_INSTRUCTION_SET_FEATURES_RISCV64_H_
#define ART_RUNTIME_ARCH_RISCV64_INSTRUCTION_SET_FEATURES_RISCV64_H_

#include "arch/instruction_set_features.h"

namespace art {

class Riscv64InstructionSetFeatures;
using Riscv64FeaturesUniquePtr = std::unique_ptr<const Riscv64InstructionSetFeatures>;

// Instruction set features relevant to the RISCV64 architecture.
class Riscv64InstructionSetFeatures final : public InstructionSetFeatures {
 public:
  static Riscv64FeaturesUniquePtr FromVariant(const std::string& variant,
                                             std::string* error_msg);

  // Parse a bitmap and create an InstructionSetFeatures.
  static Riscv64FeaturesUniquePtr FromBitmap(uint32_t bitmap);

  // Turn C pre-processor #defines into the equivalent instruction set features.
  static Riscv64FeaturesUniquePtr FromCppDefines();

  // Process /proc/cpuinfo and use kRuntimeISA to produce InstructionSetFeatures.
  static Riscv64FeaturesUniquePtr FromCpuInfo();

  // Process the auxiliary vector AT_HWCAP entry and use kRuntimeISA to produce
  // InstructionSetFeatures.
  static Riscv64FeaturesUniquePtr FromHwcap();

  // Use assembly tests of the current runtime (ie kRuntimeISA) to determine the
  // InstructionSetFeatures. This works around kernel bugs in AT_HWCAP and /proc/cpuinfo.
  static Riscv64FeaturesUniquePtr FromAssembly();

  // Use external cpu_features library.
  static Riscv64FeaturesUniquePtr FromCpuFeatures();

  bool Equals(const InstructionSetFeatures* other) const override;

  InstructionSet GetInstructionSet() const override {
    return InstructionSet::kRiscv64;
  }

  uint32_t AsBitmap() const override;

  std::string GetFeatureString() const override;

  virtual ~Riscv64InstructionSetFeatures() {}

 protected:
  std::unique_ptr<const InstructionSetFeatures>
      AddFeaturesFromSplitString(const std::vector<std::string>& features,
                                 std::string* error_msg) const override;

 private:
  explicit Riscv64InstructionSetFeatures(uint32_t bits) : InstructionSetFeatures(), bits_(bits) {
  }

  // Bitmap positions for encoding features as a bitmap: same order as /proc/cpuinfo
  enum {
    kIBitfield = (1 << 0),
    kMBitfield = (1 << 1),
    kABitfield = (1 << 2),
    kFBitfield = (1 << 3),
    kDBitfield = (1 << 4),
    kCBitfield = (1 << 5),
    kVBitfield = (1 << 6),
    kSBitfield = (1 << 7),
    kUBitfield = (1 << 8),
  };

  // Bits for AFDCM etc, will process late
  const uint32_t bits_;

  DISALLOW_COPY_AND_ASSIGN(Riscv64InstructionSetFeatures);
};

}  // namespace art

#endif  // ART_RUNTIME_ARCH_RISCV64_INSTRUCTION_SET_FEATURES_RISCV64_H_
