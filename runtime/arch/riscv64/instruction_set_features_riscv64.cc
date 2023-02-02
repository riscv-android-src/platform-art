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

#include "instruction_set_features_riscv64.h"

#include <fstream>
#include <sstream>

#include "android-base/stringprintf.h"
#include "android-base/strings.h"
#include "base/logging.h"

namespace art {

using android::base::StringPrintf;

Riscv64FeaturesUniquePtr Riscv64InstructionSetFeatures::FromVariant(
    const std::string& variant, std::string* error_msg ATTRIBUTE_UNUSED) {
  if (variant != "generic") {
    LOG(WARNING) << "Unexpected CPU variant for Riscv64 using defaults: " << variant;
  }
  uint32_t bits = kIBitfield | kMBitfield | kABitfield | kFBitfield | kDBitfield | kCBitfield;
  return Riscv64FeaturesUniquePtr(new Riscv64InstructionSetFeatures(bits));
}

Riscv64FeaturesUniquePtr Riscv64InstructionSetFeatures::FromBitmap(uint32_t bitmap) {
  return Riscv64FeaturesUniquePtr(new Riscv64InstructionSetFeatures(bitmap));
}

Riscv64FeaturesUniquePtr Riscv64InstructionSetFeatures::FromCppDefines() {
  uint32_t bits = kIBitfield | kMBitfield | kABitfield | kFBitfield | kDBitfield | kCBitfield;
  return Riscv64FeaturesUniquePtr(new Riscv64InstructionSetFeatures(bits));
}

Riscv64FeaturesUniquePtr Riscv64InstructionSetFeatures::FromCpuInfo() {
  // Look in /proc/cpuinfo for features we need.  Only use this when we can guarantee that
  // the kernel puts the appropriate feature flags in here.  Sometimes it doesn't.
  uint32_t bits = 0;
  bits = kIBitfield | kMBitfield | kABitfield | kFBitfield | kDBitfield | kCBitfield;  // basic implmentation for RV64
  // XC-TODO
  #if 0
  std::ifstream in("/proc/cpuinfo");
  if (!in.fail()) {
    while (!in.eof()) {
      std::string line;
      std::getline(in, line);
      if (!in.eof()) {
        LOG(INFO) << "cpuinfo line: " << line;
        if (line.find("isa") != std::string::npos) {
          LOG(INFO) << "found isa line: " << line;
          if (line.find("imafdcv") != std::string::npos) {
            bits = bits | kVBitfield;
          }
          break;
        }
      }
    }
    in.close();
  } else {
    LOG(ERROR) << "Failed to open /proc/cpuinfo";
  }
  #endif
  return Riscv64FeaturesUniquePtr(new Riscv64InstructionSetFeatures(bits));
}

Riscv64FeaturesUniquePtr Riscv64InstructionSetFeatures::FromHwcap() {
  UNIMPLEMENTED(WARNING);
  return FromCppDefines();
}

Riscv64FeaturesUniquePtr Riscv64InstructionSetFeatures::FromAssembly() {
  UNIMPLEMENTED(WARNING);
  return FromCppDefines();
}

Riscv64FeaturesUniquePtr Riscv64InstructionSetFeatures::FromCpuFeatures() {
  UNIMPLEMENTED(WARNING);
  return FromCppDefines();
}

bool Riscv64InstructionSetFeatures::Equals(const InstructionSetFeatures* other) const {
  if (InstructionSet::kRiscv64 != other->GetInstructionSet()) {
    return false;
  }
  const Riscv64InstructionSetFeatures* other_as_riscv64 = other->AsRiscv64InstructionSetFeatures();
  return bits_ == other_as_riscv64->bits_;
}

uint32_t Riscv64InstructionSetFeatures::AsBitmap() const {
  return bits_;
}

std::string Riscv64InstructionSetFeatures::GetFeatureString() const {
  std::string result = "rv64imaf";
  if (bits_ & kDBitfield) {
    result += "d";
  }
  if (bits_ & kCBitfield) {
    result += "c";
  }
  if (bits_ & kVBitfield) {
    result += "v";
  }
  return result;
}

std::unique_ptr<const InstructionSetFeatures>
Riscv64InstructionSetFeatures::AddFeaturesFromSplitString(
    const std::vector<std::string>& features ATTRIBUTE_UNUSED, std::string* error_msg ATTRIBUTE_UNUSED) const {
  uint32_t bits = kIBitfield | kMBitfield | kABitfield | kFBitfield | kDBitfield | kCBitfield;
  return std::unique_ptr<const InstructionSetFeatures>(new Riscv64InstructionSetFeatures(bits));
}

}  // namespace art
