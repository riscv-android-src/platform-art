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

#ifndef ART_COMPILER_UTILS_RISCV64_ASSEMBLER_RISCV64_H_
#define ART_COMPILER_UTILS_RISCV64_ASSEMBLER_RISCV64_H_

#include <deque>
#include <utility>
#include <vector>

#include "arch/riscv64/instruction_set_features_riscv64.h"
#include "base/arena_containers.h"
#include "base/enums.h"
#include "base/globals.h"
#include "base/macros.h"
#include "base/stl_util_identity.h"
#include "constants_riscv64.h"
#include "heap_poisoning.h"
#include "managed_register_riscv64.h"
#include "offsets.h"
#include "utils/assembler.h"
#include "utils/jni_macro_assembler.h"
#include "utils/label.h"

namespace art {
namespace riscv64 {

enum FPRoundingMode {
  kFPRoundingModeRNE             = 0x0,  // Round to Nearest, ties to Even
  kFPRoundingModeRTZ             = 0x1,  // Round towards Zero
  kFPRoundingModeRDN             = 0x2,  // Round Down (towards −∞)
  kFPRoundingModeRUP             = 0x3,  // Round Up (towards +∞)
  kFPRoundingModeRMM             = 0x4,  // Round to Nearest, ties to Max Magnitude
  kFPRoundingModeDYN             = 0x7,  // Dynamic rounding mode
};

#define FRM                     (kFPRoundingModeDYN)

enum LoadConst64Path {
  kLoadConst64PathZero           = 0x0,
  kLoadConst64PathOri            = 0x1,
  kLoadConst64PathDaddiu         = 0x2,
  kLoadConst64PathLui            = 0x4,
  kLoadConst64PathLuiOri         = 0x8,
  kLoadConst64PathOriDahi        = 0x10,
  kLoadConst64PathOriDati        = 0x20,
  kLoadConst64PathLuiDahi        = 0x40,
  kLoadConst64PathLuiDati        = 0x80,
  kLoadConst64PathDaddiuDsrlX    = 0x100,
  kLoadConst64PathOriDsllX       = 0x200,
  kLoadConst64PathDaddiuDsllX    = 0x400,
  kLoadConst64PathLuiOriDsllX    = 0x800,
  kLoadConst64PathOriDsllXOri    = 0x1000,
  kLoadConst64PathDaddiuDsllXOri = 0x2000,
  kLoadConst64PathDaddiuDahi     = 0x4000,
  kLoadConst64PathDaddiuDati     = 0x8000,
  kLoadConst64PathDinsu1         = 0x10000,
  kLoadConst64PathDinsu2         = 0x20000,
  kLoadConst64PathCatchAll       = 0x40000,
  kLoadConst64PathAllPaths       = 0x7ffff,
};

inline uint16_t Low12Bits(uint32_t value) {
  return static_cast<uint16_t>(value & 0xFFF);
}

inline uint32_t High20Bits(uint32_t value) {
  return static_cast<uint32_t>(value >> 12);
}

template <typename Asm>
void TemplateLoadConst32(Asm* a, GpuRegister rd, int32_t value) {
  if (IsUint<16>(value)) {
    // Use OR with (unsigned) immediate to encode 16b unsigned int.
    a->Ori(rd, ZERO, value);
  } else if (IsInt<16>(value)) {
    // Use ADD with (signed) immediate to encode 16b signed int.
    a->Addiw(rd, ZERO, value);
  } else {
    // Set 16 most significant bits of value. The "lui" instruction
    // also clears the 16 least significant bits to zero.
    a->Lui(rd, value >> 16);
    if (value & 0xFFFF) {
      // If the 16 least significant bits are non-zero, set them
      // here.
      a->Ori(rd, rd, value);
    }
  }
}

static inline int InstrCountForLoadReplicatedConst32(int64_t value) {
  int32_t x = Low32Bits(value);
  int32_t y = High32Bits(value);

  if (x == y) {
    return (IsUint<16>(x) || IsInt<16>(x) || ((x & 0xFFFF) == 0)) ? 2 : 3;
  }

  return INT_MAX;
}

template <typename Asm, typename Rtype, typename Vtype>
void TemplateLoadConst64(Asm* a, Rtype rd, Vtype value) {
  int bit31 = (value & UINT64_C(0x80000000)) != 0;
  int rep32_count = InstrCountForLoadReplicatedConst32(value);

  // Loads with 1 instruction.
  if (IsUint<16>(value)) {
    // 64-bit value can be loaded as an unsigned 16-bit number.
    a->RecordLoadConst64Path(kLoadConst64PathOri);
    a->Ori(rd, ZERO, value);
  } else if (IsInt<16>(value)) {
    // 64-bit value can be loaded as an signed 16-bit number.
    a->RecordLoadConst64Path(kLoadConst64PathDaddiu);
    a->Daddiu(rd, ZERO, value);
  } else if ((value & 0xFFFF) == 0 && IsInt<16>(value >> 16)) {
    // 64-bit value can be loaded as an signed 32-bit number which has all
    // of its 16 least significant bits set to zero.
    a->RecordLoadConst64Path(kLoadConst64PathLui);
    a->Lui(rd, value >> 16);
  } else if (IsInt<32>(value)) {
    // Loads with 2 instructions.
    // 64-bit value can be loaded as an signed 32-bit number which has some
    // or all of its 16 least significant bits set to one.
    a->RecordLoadConst64Path(kLoadConst64PathLuiOri);
    a->Lui(rd, value >> 16);
    a->Ori(rd, rd, value);
  } else if ((value & 0xFFFF0000) == 0 && IsInt<16>(value >> 32)) {
    // 64-bit value which consists of an unsigned 16-bit value in its
    // least significant 32-bits, and a signed 16-bit value in its
    // most significant 32-bits.
    a->RecordLoadConst64Path(kLoadConst64PathOriDahi);
    a->Ori(rd, ZERO, value);
    a->Dahi(rd, value >> 32);
  } else if ((value & UINT64_C(0xFFFFFFFF0000)) == 0) {
    // 64-bit value which consists of an unsigned 16-bit value in its
    // least significant 48-bits, and a signed 16-bit value in its
    // most significant 16-bits.
    a->RecordLoadConst64Path(kLoadConst64PathOriDati);
    a->Ori(rd, ZERO, value);
    a->Dati(rd, value >> 48);
  } else if ((value & 0xFFFF) == 0 &&
             (-32768 - bit31) <= (value >> 32) && (value >> 32) <= (32767 - bit31)) {
    // 16 LSBs (Least Significant Bits) all set to zero.
    // 48 MSBs (Most Significant Bits) hold a signed 32-bit value.
    a->RecordLoadConst64Path(kLoadConst64PathLuiDahi);
    a->Lui(rd, value >> 16);
    a->Dahi(rd, (value >> 32) + bit31);
  } else if ((value & 0xFFFF) == 0 && ((value >> 31) & 0x1FFFF) == ((0x20000 - bit31) & 0x1FFFF)) {
    // 16 LSBs all set to zero.
    // 48 MSBs hold a signed value which can't be represented by signed
    // 32-bit number, and the middle 16 bits are all zero, or all one.
    a->RecordLoadConst64Path(kLoadConst64PathLuiDati);
    a->Lui(rd, value >> 16);
    a->Dati(rd, (value >> 48) + bit31);
  } else if (IsInt<16>(static_cast<int32_t>(value)) &&
             (-32768 - bit31) <= (value >> 32) && (value >> 32) <= (32767 - bit31)) {
    // 32 LSBs contain an unsigned 16-bit number.
    // 32 MSBs contain a signed 16-bit number.
    a->RecordLoadConst64Path(kLoadConst64PathDaddiuDahi);
    a->Daddiu(rd, ZERO, value);
    a->Dahi(rd, (value >> 32) + bit31);
  } else if (IsInt<16>(static_cast<int32_t>(value)) &&
             ((value >> 31) & 0x1FFFF) == ((0x20000 - bit31) & 0x1FFFF)) {
    // 48 LSBs contain an unsigned 16-bit number.
    // 16 MSBs contain a signed 16-bit number.
    a->RecordLoadConst64Path(kLoadConst64PathDaddiuDati);
    a->Daddiu(rd, ZERO, value);
    a->Dati(rd, (value >> 48) + bit31);
  } else if (IsPowerOfTwo(value + UINT64_C(1))) {
    // 64-bit values which have their "n" MSBs set to one, and their
    // "64-n" LSBs set to zero. "n" must meet the restrictions 0 < n < 64.
    int shift_cnt = 64 - CTZ(value + UINT64_C(1));
    a->RecordLoadConst64Path(kLoadConst64PathDaddiuDsrlX);
    a->Daddiu(rd, ZERO, -1);
    if (shift_cnt < 32) {
      a->Dsrl(rd, rd, shift_cnt);
    } else {
      a->Dsrl32(rd, rd, shift_cnt & 31);
    }
  } else {
    int shift_cnt = CTZ(value);
    int64_t tmp = value >> shift_cnt;
    a->RecordLoadConst64Path(kLoadConst64PathOriDsllX);
    if (IsUint<16>(tmp)) {
      // Value can be computed by loading a 16-bit unsigned value, and
      // then shifting left.
      a->Ori(rd, ZERO, tmp);
      if (shift_cnt < 32) {
        a->Dsll(rd, rd, shift_cnt);
      } else {
        a->Dsll32(rd, rd, shift_cnt & 31);
      }
    } else if (IsInt<16>(tmp)) {
      // Value can be computed by loading a 16-bit signed value, and
      // then shifting left.
      a->RecordLoadConst64Path(kLoadConst64PathDaddiuDsllX);
      a->Daddiu(rd, ZERO, tmp);
      if (shift_cnt < 32) {
        a->Dsll(rd, rd, shift_cnt);
      } else {
        a->Dsll32(rd, rd, shift_cnt & 31);
      }
    } else if (rep32_count < 3) {
      // Value being loaded has 32 LSBs equal to the 32 MSBs, and the
      // value loaded into the 32 LSBs can be loaded with a single
      // MIPS instruction.
      a->LoadConst32(rd, value);
      a->Dinsu(rd, rd, 32, 32);
      a->RecordLoadConst64Path(kLoadConst64PathDinsu1);
    } else if (IsInt<32>(tmp)) {
      // Loads with 3 instructions.
      // Value can be computed by loading a 32-bit signed value, and
      // then shifting left.
      a->RecordLoadConst64Path(kLoadConst64PathLuiOriDsllX);
      a->Lui(rd, tmp >> 16);
      a->Ori(rd, rd, tmp);
      if (shift_cnt < 32) {
        a->Dsll(rd, rd, shift_cnt);
      } else {
        a->Dsll32(rd, rd, shift_cnt & 31);
      }
    } else {
      shift_cnt = 16 + CTZ(value >> 16);
      tmp = value >> shift_cnt;
      if (IsUint<16>(tmp)) {
        // Value can be computed by loading a 16-bit unsigned value,
        // shifting left, and "or"ing in another 16-bit unsigned value.
        a->RecordLoadConst64Path(kLoadConst64PathOriDsllXOri);
        a->Ori(rd, ZERO, tmp);
        if (shift_cnt < 32) {
          a->Dsll(rd, rd, shift_cnt);
        } else {
          a->Dsll32(rd, rd, shift_cnt & 31);
        }
        a->Ori(rd, rd, value);
      } else if (IsInt<16>(tmp)) {
        // Value can be computed by loading a 16-bit signed value,
        // shifting left, and "or"ing in a 16-bit unsigned value.
        a->RecordLoadConst64Path(kLoadConst64PathDaddiuDsllXOri);
        a->Daddiu(rd, ZERO, tmp);
        if (shift_cnt < 32) {
          a->Dsll(rd, rd, shift_cnt);
        } else {
          a->Dsll32(rd, rd, shift_cnt & 31);
        }
        a->Ori(rd, rd, value);
      } else if (rep32_count < 4) {
        // Value being loaded has 32 LSBs equal to the 32 MSBs, and the
        // value in the 32 LSBs requires 2 MIPS instructions to load.
        a->LoadConst32(rd, value);
        a->Dinsu(rd, rd, 32, 32);
        a->RecordLoadConst64Path(kLoadConst64PathDinsu2);
      } else {
        // Loads with 3-4 instructions.
        // Catch-all case to get any other 64-bit values which aren't
        // handled by special cases above.
        uint64_t tmp2 = value;
        a->RecordLoadConst64Path(kLoadConst64PathCatchAll);
        a->LoadConst32(rd, value);
        if (bit31) {
          tmp2 += UINT64_C(0x100000000);
        }
        if (((tmp2 >> 32) & 0xFFFF) != 0) {
          a->Dahi(rd, tmp2 >> 32);
        }
        if (tmp2 & UINT64_C(0x800000000000)) {
          tmp2 += UINT64_C(0x1000000000000);
        }
        if ((tmp2 >> 48) != 0) {
          a->Dati(rd, tmp2 >> 48);
        }
      }
    }
  }
}

static constexpr size_t kRiscv64HalfwordSize = 2;
static constexpr size_t kRiscv64WordSize = 4;
static constexpr size_t kRiscv64DoublewordSize = 8;

enum LoadOperandType {
  kLoadSignedByte,
  kLoadUnsignedByte,
  kLoadSignedHalfword,
  kLoadUnsignedHalfword,
  kLoadWord,
  kLoadUnsignedWord,
  kLoadDoubleword,
  kLoadQuadword
};

enum StoreOperandType {
  kStoreByte,
  kStoreHalfword,
  kStoreWord,
  kStoreDoubleword,
  kStoreQuadword
};

// Used to test the values returned by ClassS/ClassD.
enum FPClassMaskType {
  kSignalingNaN      = 0x001,
  kQuietNaN          = 0x002,
  kNegativeInfinity  = 0x004,
  kNegativeNormal    = 0x008,
  kNegativeSubnormal = 0x010,
  kNegativeZero      = 0x020,
  kPositiveInfinity  = 0x040,
  kPositiveNormal    = 0x080,
  kPositiveSubnormal = 0x100,
  kPositiveZero      = 0x200,
};

class Riscv64Label : public Label {
 public:
  Riscv64Label() : prev_branch_id_plus_one_(0) {}

  Riscv64Label(Riscv64Label&& src)
      : Label(std::move(src)), prev_branch_id_plus_one_(src.prev_branch_id_plus_one_) {}

 private:
  uint32_t prev_branch_id_plus_one_;  // To get distance from preceding branch, if any.

  friend class Riscv64Assembler;
  DISALLOW_COPY_AND_ASSIGN(Riscv64Label);
};

// Assembler literal is a value embedded in code, retrieved using a PC-relative load.
class Literal {
 public:
  static constexpr size_t kMaxSize = 8;

  Literal(uint32_t size, const uint8_t* data)
      : label_(), size_(size) {
    DCHECK_LE(size, Literal::kMaxSize);
    memcpy(data_, data, size);
  }

  template <typename T>
  T GetValue() const {
    DCHECK_EQ(size_, sizeof(T));
    T value;
    memcpy(&value, data_, sizeof(T));
    return value;
  }

  uint32_t GetSize() const {
    return size_;
  }

  const uint8_t* GetData() const {
    return data_;
  }

  Riscv64Label* GetLabel() {
    return &label_;
  }

  const Riscv64Label* GetLabel() const {
    return &label_;
  }

 private:
  Riscv64Label label_;
  const uint32_t size_;
  uint8_t data_[kMaxSize];

  DISALLOW_COPY_AND_ASSIGN(Literal);
};

// Jump table: table of labels emitted after the code and before the literals. Similar to literals.
class JumpTable {
 public:
  explicit JumpTable(std::vector<Riscv64Label*>&& labels)
      : label_(), labels_(std::move(labels)) {
  }

  size_t GetSize() const {
    return labels_.size() * sizeof(uint32_t);
  }

  const std::vector<Riscv64Label*>& GetData() const {
    return labels_;
  }

  Riscv64Label* GetLabel() {
    return &label_;
  }

  const Riscv64Label* GetLabel() const {
    return &label_;
  }

 private:
  Riscv64Label label_;
  std::vector<Riscv64Label*> labels_;

  DISALLOW_COPY_AND_ASSIGN(JumpTable);
};

// Slowpath entered when Thread::Current()->_exception is non-null.
class Riscv64ExceptionSlowPath {
 public:
  explicit Riscv64ExceptionSlowPath(Riscv64ManagedRegister scratch, size_t stack_adjust)
      : scratch_(scratch), stack_adjust_(stack_adjust) {}

  Riscv64ExceptionSlowPath(Riscv64ExceptionSlowPath&& src)
      : scratch_(src.scratch_),
        stack_adjust_(src.stack_adjust_),
        exception_entry_(std::move(src.exception_entry_)) {}

 private:
  Riscv64Label* Entry() { return &exception_entry_; }
  const Riscv64ManagedRegister scratch_;
  const size_t stack_adjust_;
  Riscv64Label exception_entry_;

  friend class Riscv64Assembler;
  DISALLOW_COPY_AND_ASSIGN(Riscv64ExceptionSlowPath);
};

class Riscv64Assembler final : public Assembler, public JNIMacroAssembler<PointerSize::k64> {
 public:
  using JNIBase = JNIMacroAssembler<PointerSize::k64>;

  explicit Riscv64Assembler(ArenaAllocator* allocator,
                           const Riscv64InstructionSetFeatures* instruction_set_features = nullptr)
      : Assembler(allocator),
        overwriting_(false),
        overwrite_location_(0),
        literals_(allocator->Adapter(kArenaAllocAssembler)),
        long_literals_(allocator->Adapter(kArenaAllocAssembler)),
        jump_tables_(allocator->Adapter(kArenaAllocAssembler)),
        last_position_adjustment_(0),
        last_old_position_(0),
        last_branch_id_(0),
        has_msa_(false) {
    (void) instruction_set_features;
    cfi().DelayEmittingAdvancePCs();
  }

  virtual ~Riscv64Assembler() {
    for (auto& branch : branches_) {
      CHECK(branch.IsResolved());
    }
  }

  size_t CodeSize() const override { return Assembler::CodeSize(); }
  DebugFrameOpCodeWriterForAssembler& cfi() override { return Assembler::cfi(); }

  // Emit Machine Instructions.
  [[noreturn]] void Addu(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void Addiu(GpuRegister rt, GpuRegister rs, int16_t imm16);
  [[noreturn]] void Daddu(GpuRegister rd, GpuRegister rs, GpuRegister rt);  // RISCV64
  [[noreturn]] void Daddiu(GpuRegister rt, GpuRegister rs, int16_t imm16);  // RISCV64
  [[noreturn]] void Subu(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void Dsubu(GpuRegister rd, GpuRegister rs, GpuRegister rt);  // RISCV64

  [[noreturn]] void MulR6(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void MuhR6(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void DivR6(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void ModR6(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void DivuR6(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void ModuR6(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void Dmul(GpuRegister rd, GpuRegister rs, GpuRegister rt);  // RISCV64
  [[noreturn]] void Dmuh(GpuRegister rd, GpuRegister rs, GpuRegister rt);  // RISCV64
  [[noreturn]] void Ddiv(GpuRegister rd, GpuRegister rs, GpuRegister rt);  // RISCV64
  [[noreturn]] void Dmod(GpuRegister rd, GpuRegister rs, GpuRegister rt);  // RISCV64
  [[noreturn]] void Ddivu(GpuRegister rd, GpuRegister rs, GpuRegister rt);  // RISCV64
  [[noreturn]] void Dmodu(GpuRegister rd, GpuRegister rs, GpuRegister rt);  // RISCV64

  [[noreturn]] void Bitswap(GpuRegister rd, GpuRegister rt);
  [[noreturn]] void Dbitswap(GpuRegister rd, GpuRegister rt);  // RISCV64
  [[noreturn]] void Seb(GpuRegister rd, GpuRegister rt);
  [[noreturn]] void Seh(GpuRegister rd, GpuRegister rt);
  [[noreturn]] void Dsbh(GpuRegister rd, GpuRegister rt);  // RISCV64
  [[noreturn]] void Dshd(GpuRegister rd, GpuRegister rt);  // RISCV64
  [[noreturn]] void Dext(GpuRegister rs, GpuRegister rt, int pos, int size);  // RISCV64
  [[noreturn]] void Ins(GpuRegister rt, GpuRegister rs, int pos, int size);
  [[noreturn]] void Dins(GpuRegister rt, GpuRegister rs, int pos, int size);  // RISCV64
  [[noreturn]] void Dinsm(GpuRegister rt, GpuRegister rs, int pos, int size);  // RISCV64
  [[noreturn]] void Dinsu(GpuRegister rt, GpuRegister rs, int pos, int size);  // RISCV64
  [[noreturn]] void DblIns(GpuRegister rt, GpuRegister rs, int pos, int size);  // RISCV64
  [[noreturn]] void Lsa(GpuRegister rd, GpuRegister rs, GpuRegister rt, int saPlusOne);
  [[noreturn]] void Dlsa(GpuRegister rd, GpuRegister rs, GpuRegister rt, int saPlusOne);  // RISCV64
  [[noreturn]] void Wsbh(GpuRegister rd, GpuRegister rt);
  [[noreturn]] void Sc(GpuRegister rt, GpuRegister base, int16_t imm9 = 0);
  [[noreturn]] void Scd(GpuRegister rt, GpuRegister base, int16_t imm9 = 0);  // RISCV64
  [[noreturn]] void Ll(GpuRegister rt, GpuRegister base, int16_t imm9 = 0);
  [[noreturn]] void Lld(GpuRegister rt, GpuRegister base, int16_t imm9 = 0);  // RISCV64

  [[noreturn]] void Sll(GpuRegister rd, GpuRegister rt, int shamt);
  [[noreturn]] void Srl(GpuRegister rd, GpuRegister rt, int shamt);
  [[noreturn]] void Rotr(GpuRegister rd, GpuRegister rt, int shamt);
  [[noreturn]] void Sra(GpuRegister rd, GpuRegister rt, int shamt);
  [[noreturn]] void Sllv(GpuRegister rd, GpuRegister rt, GpuRegister rs);
  [[noreturn]] void Srlv(GpuRegister rd, GpuRegister rt, GpuRegister rs);
  [[noreturn]] void Rotrv(GpuRegister rd, GpuRegister rt, GpuRegister rs);
  [[noreturn]] void Srav(GpuRegister rd, GpuRegister rt, GpuRegister rs);
  [[noreturn]] void Dsll(GpuRegister rd, GpuRegister rt, int shamt);  // RISCV64
  [[noreturn]] void Dsrl(GpuRegister rd, GpuRegister rt, int shamt);  // RISCV64
  [[noreturn]] void Drotr(GpuRegister rd, GpuRegister rt, int shamt);  // RISCV64
  [[noreturn]] void Dsra(GpuRegister rd, GpuRegister rt, int shamt);  // RISCV64
  [[noreturn]] void Dsll32(GpuRegister rd, GpuRegister rt, int shamt);  // RISCV64
  [[noreturn]] void Dsrl32(GpuRegister rd, GpuRegister rt, int shamt);  // RISCV64
  [[noreturn]] void Drotr32(GpuRegister rd, GpuRegister rt, int shamt);  // RISCV64
  [[noreturn]] void Dsra32(GpuRegister rd, GpuRegister rt, int shamt);  // RISCV64
  [[noreturn]] void Dsllv(GpuRegister rd, GpuRegister rt, GpuRegister rs);  // RISCV64
  [[noreturn]] void Dsrlv(GpuRegister rd, GpuRegister rt, GpuRegister rs);  // RISCV64
  [[noreturn]] void Drotrv(GpuRegister rd, GpuRegister rt, GpuRegister rs);  // RISCV64
  [[noreturn]] void Dsrav(GpuRegister rd, GpuRegister rt, GpuRegister rs);  // RISCV64

  [[noreturn]] void Lwpc(GpuRegister rs, uint32_t imm19);
  [[noreturn]] void Lwupc(GpuRegister rs, uint32_t imm19);  // RISCV64
  [[noreturn]] void Ldpc(GpuRegister rs, uint32_t imm18);  // RISCV64
  /*
  [[noreturn]] void Lui(GpuRegister rt, uint16_t imm16);
  */
  [[noreturn]] void Aui(GpuRegister rt, GpuRegister rs, uint16_t imm16);
  [[noreturn]] void Daui(GpuRegister rt, GpuRegister rs, uint16_t imm16);  // RISCV64
  [[noreturn]] void Dahi(GpuRegister rs, uint16_t imm16);  // RISCV64
  [[noreturn]] void Dati(GpuRegister rs, uint16_t imm16);  // RISCV64
  [[noreturn]] void Sync(uint32_t stype);

  [[noreturn]] void Seleqz(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void Selnez(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  [[noreturn]] void Clz(GpuRegister rd, GpuRegister rs);
  [[noreturn]] void Clo(GpuRegister rd, GpuRegister rs);
  [[noreturn]] void Dclz(GpuRegister rd, GpuRegister rs);  // RISCV64
  [[noreturn]] void Dclo(GpuRegister rd, GpuRegister rs);  // RISCV64

  [[noreturn]] void Jalr(GpuRegister rd, GpuRegister rs);
  [[noreturn]] void Jalr(GpuRegister rs);
  [[noreturn]] void Jr(GpuRegister rs);
  [[noreturn]] void Addiupc(GpuRegister rs, uint32_t imm19);
  [[noreturn]] void Bc(uint32_t imm20);
  [[noreturn]] void Balc(uint32_t imm20);
  [[noreturn]] void Jic(GpuRegister rt, uint16_t imm16);
  [[noreturn]] void Jialc(GpuRegister rt, uint16_t imm16);
  [[noreturn]] void Bltc(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Bltzc(GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Bgtzc(GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Bgec(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Bgezc(GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Blezc(GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Bltuc(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Bgeuc(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Beqc(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Bnec(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  [[noreturn]] void Beqzc(GpuRegister rs, uint32_t imm12);
  [[noreturn]] void Bnezc(GpuRegister rs, uint32_t imm12);

  [[noreturn]] void AddS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void SubS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void MulS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void DivS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void AddD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void SubD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void MulD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void DivD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void SqrtS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void SqrtD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void AbsS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void AbsD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void MovS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void MovD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void NegS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void NegD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void RoundLS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void RoundLD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void RoundWS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void RoundWD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void TruncLS(GpuRegister rd, FpuRegister fs);
  [[noreturn]] void TruncLD(GpuRegister rd, FpuRegister fs);
  [[noreturn]] void TruncWS(GpuRegister rd, FpuRegister fs);
  [[noreturn]] void TruncWD(GpuRegister rd, FpuRegister fs);
  [[noreturn]] void CeilLS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void CeilLD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void CeilWS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void CeilWD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void FloorLS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void FloorLD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void FloorWS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void FloorWD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void SelS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void SelD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void SeleqzS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void SeleqzD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void SelnezS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void SelnezD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void RintS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void RintD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void ClassS(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void ClassD(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void MinS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void MinD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void MaxS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void MaxD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUnS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpEqS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUeqS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpLtS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUltS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpLeS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUleS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpOrS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUneS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpNeS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUnD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpEqD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUeqD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpLtD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUltD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpLeD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUleD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpOrD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpUneD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  [[noreturn]] void CmpNeD(GpuRegister rd, FpuRegister fs, FpuRegister ft);

  [[noreturn]] void Cvtsw(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void Cvtdw(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void Cvtsd(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void Cvtds(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void Cvtsl(FpuRegister fd, FpuRegister fs);
  [[noreturn]] void Cvtdl(FpuRegister fd, FpuRegister fs);

  [[noreturn]] void Mfc1(GpuRegister rt, FpuRegister fs);
  [[noreturn]] void Mfhc1(GpuRegister rt, FpuRegister fs);
  [[noreturn]] void Mtc1(GpuRegister rt, FpuRegister fs);
  [[noreturn]] void Mthc1(GpuRegister rt, FpuRegister fs);
  [[noreturn]] void Dmfc1(GpuRegister rt, FpuRegister fs);  // RISCV64
  [[noreturn]] void Dmtc1(GpuRegister rt, FpuRegister fs);  // RISCV64
  [[noreturn]] void Lwc1(FpuRegister ft, GpuRegister rs, uint16_t imm12);
  [[noreturn]] void Ldc1(FpuRegister ft, GpuRegister rs, uint16_t imm12);
  [[noreturn]] void Swc1(FpuRegister ft, GpuRegister rs, uint16_t imm12);
  [[noreturn]] void Sdc1(FpuRegister ft, GpuRegister rs, uint16_t imm12);

  [[noreturn]] void Break();
  [[noreturn]] void Nop();
  [[noreturn]] void Move(GpuRegister rd, GpuRegister rs);
  [[noreturn]] void Clear(GpuRegister rd);
  [[noreturn]] void Not(GpuRegister rd, GpuRegister rs);

  // MSA instructions.
  [[noreturn]] void AndV(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void OrV(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void NorV(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void XorV(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  [[noreturn]] void AddvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void AddvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void AddvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void AddvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SubvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SubvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SubvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SubvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Asub_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Asub_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Asub_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Asub_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Asub_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Asub_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Asub_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Asub_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MulvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MulvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MulvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MulvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Div_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Div_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Div_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Div_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Div_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Div_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Div_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Div_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Mod_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Mod_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Mod_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Mod_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Mod_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Mod_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Mod_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Mod_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Add_aB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Add_aH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Add_aW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Add_aD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Ave_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Ave_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Ave_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Ave_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Ave_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Ave_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Ave_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Ave_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Aver_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Aver_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Aver_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Aver_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Aver_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Aver_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Aver_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Aver_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Max_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Max_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Max_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Max_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Max_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Max_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Max_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Max_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Min_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Min_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Min_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Min_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Min_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Min_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Min_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Min_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  [[noreturn]] void FaddW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FaddD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FsubW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FsubD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FmulW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FmulD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FdivW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FdivD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FmaxW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FmaxD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FminW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FminD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  [[noreturn]] void Ffint_sW(VectorRegister wd, VectorRegister ws);
  [[noreturn]] void Ffint_sD(VectorRegister wd, VectorRegister ws);
  [[noreturn]] void Ftint_sW(VectorRegister wd, VectorRegister ws);
  [[noreturn]] void Ftint_sD(VectorRegister wd, VectorRegister ws);

  [[noreturn]] void SllB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SllH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SllW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SllD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SraB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SraH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SraW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SraD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SrlB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SrlH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SrlW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void SrlD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  // Immediate shift instructions, where shamtN denotes shift amount (must be between 0 and 2^N-1).
  [[noreturn]] void SlliB(VectorRegister wd, VectorRegister ws, int shamt3);
  [[noreturn]] void SlliH(VectorRegister wd, VectorRegister ws, int shamt4);
  [[noreturn]] void SlliW(VectorRegister wd, VectorRegister ws, int shamt5);
  [[noreturn]] void SlliD(VectorRegister wd, VectorRegister ws, int shamt6);
  [[noreturn]] void SraiB(VectorRegister wd, VectorRegister ws, int shamt3);
  [[noreturn]] void SraiH(VectorRegister wd, VectorRegister ws, int shamt4);
  [[noreturn]] void SraiW(VectorRegister wd, VectorRegister ws, int shamt5);
  [[noreturn]] void SraiD(VectorRegister wd, VectorRegister ws, int shamt6);
  [[noreturn]] void SrliB(VectorRegister wd, VectorRegister ws, int shamt3);
  [[noreturn]] void SrliH(VectorRegister wd, VectorRegister ws, int shamt4);
  [[noreturn]] void SrliW(VectorRegister wd, VectorRegister ws, int shamt5);
  [[noreturn]] void SrliD(VectorRegister wd, VectorRegister ws, int shamt6);

  [[noreturn]] void MoveV(VectorRegister wd, VectorRegister ws);
  [[noreturn]] void SplatiB(VectorRegister wd, VectorRegister ws, int n4);
  [[noreturn]] void SplatiH(VectorRegister wd, VectorRegister ws, int n3);
  [[noreturn]] void SplatiW(VectorRegister wd, VectorRegister ws, int n2);
  [[noreturn]] void SplatiD(VectorRegister wd, VectorRegister ws, int n1);
  [[noreturn]] void Copy_sB(GpuRegister rd, VectorRegister ws, int n4);
  [[noreturn]] void Copy_sH(GpuRegister rd, VectorRegister ws, int n3);
  [[noreturn]] void Copy_sW(GpuRegister rd, VectorRegister ws, int n2);
  [[noreturn]] void Copy_sD(GpuRegister rd, VectorRegister ws, int n1);
  [[noreturn]] void Copy_uB(GpuRegister rd, VectorRegister ws, int n4);
  [[noreturn]] void Copy_uH(GpuRegister rd, VectorRegister ws, int n3);
  [[noreturn]] void Copy_uW(GpuRegister rd, VectorRegister ws, int n2);
  [[noreturn]] void InsertB(VectorRegister wd, GpuRegister rs, int n4);
  [[noreturn]] void InsertH(VectorRegister wd, GpuRegister rs, int n3);
  [[noreturn]] void InsertW(VectorRegister wd, GpuRegister rs, int n2);
  [[noreturn]] void InsertD(VectorRegister wd, GpuRegister rs, int n1);
  [[noreturn]] void FillB(VectorRegister wd, GpuRegister rs);
  [[noreturn]] void FillH(VectorRegister wd, GpuRegister rs);
  [[noreturn]] void FillW(VectorRegister wd, GpuRegister rs);
  [[noreturn]] void FillD(VectorRegister wd, GpuRegister rs);

  [[noreturn]] void LdiB(VectorRegister wd, int imm8);
  [[noreturn]] void LdiH(VectorRegister wd, int imm10);
  [[noreturn]] void LdiW(VectorRegister wd, int imm10);
  [[noreturn]] void LdiD(VectorRegister wd, int imm10);
  [[noreturn]] void LdB(VectorRegister wd, GpuRegister rs, int offset);
  [[noreturn]] void LdH(VectorRegister wd, GpuRegister rs, int offset);
  [[noreturn]] void LdW(VectorRegister wd, GpuRegister rs, int offset);
  [[noreturn]] void LdD(VectorRegister wd, GpuRegister rs, int offset);
  [[noreturn]] void StB(VectorRegister wd, GpuRegister rs, int offset);
  [[noreturn]] void StH(VectorRegister wd, GpuRegister rs, int offset);
  [[noreturn]] void StW(VectorRegister wd, GpuRegister rs, int offset);
  [[noreturn]] void StD(VectorRegister wd, GpuRegister rs, int offset);

  [[noreturn]] void IlvlB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvlH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvlW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvlD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvrB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvrH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvrW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvrD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvevB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvevH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvevW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvevD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvodB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvodH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvodW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void IlvodD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  [[noreturn]] void MaddvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MaddvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MaddvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MaddvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MsubvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MsubvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MsubvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void MsubvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FmaddW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FmaddD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FmsubW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void FmsubD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  [[noreturn]] void Hadd_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Hadd_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Hadd_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Hadd_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Hadd_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  [[noreturn]] void Hadd_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  [[noreturn]] void PcntB(VectorRegister wd, VectorRegister ws);
  [[noreturn]] void PcntH(VectorRegister wd, VectorRegister ws);
  [[noreturn]] void PcntW(VectorRegister wd, VectorRegister ws);
  [[noreturn]] void PcntD(VectorRegister wd, VectorRegister ws);


  // TODO dvt porting...
  /////////////////////////////// RV32I ///////////////////////////////
  // RV32I-R
  [[noreturn]] void Add(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Sub(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Sll(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Slt(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Sltu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Xor(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Srl(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Sra(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Or(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void And(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  // RV32I-I
  [[noreturn]] void Jalr(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Lb(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Lh(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Lw(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Lbu(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Lhu(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Addi(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Slti(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Sltiu(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Xori(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Ori(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Andi(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Slli(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Srli(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Srai(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Fence(uint8_t pred, uint8_t succ);
  [[noreturn]] void FenceI();
  [[noreturn]] void Ecall();
  [[noreturn]] void Ebreak();
  [[noreturn]] void Csrrw(GpuRegister rd, GpuRegister rs1, uint16_t csr);  // the order is not consitence with instruction
  [[noreturn]] void Csrrs(GpuRegister rd, GpuRegister rs1, uint16_t csr);  // the order is not consitence with instruction
  [[noreturn]] void Csrrc(GpuRegister rd, GpuRegister rs1, uint16_t csr);  // the order is not consitence with instruction
  [[noreturn]] void Csrrwi(GpuRegister rd, uint16_t csr, uint8_t zimm /*imm5*/);
  [[noreturn]] void Csrrsi(GpuRegister rd, uint16_t csr, uint8_t zimm /*imm5*/);
  [[noreturn]] void Csrrci(GpuRegister rd, uint16_t csr, uint8_t zimm /*imm5*/);

  // RV32I-S
  [[noreturn]] void Sb(GpuRegister rs2, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Sh(GpuRegister rs2, GpuRegister rs1, uint16_t offset);
  [[noreturn]] void Sw(GpuRegister rs2, GpuRegister rs1, uint16_t offset);
  // RV32I-B
  [[noreturn]] void Beq(GpuRegister rs1, GpuRegister rs2, uint16_t offset);
  [[noreturn]] void Bne(GpuRegister rs1, GpuRegister rs2, uint16_t offset);
  [[noreturn]] void Blt(GpuRegister rs1, GpuRegister rs2, uint16_t offset);
  [[noreturn]] void Bge(GpuRegister rs1, GpuRegister rs2, uint16_t offset);
  [[noreturn]] void Bltu(GpuRegister rs1, GpuRegister rs2, uint16_t offset);
  [[noreturn]] void Bgeu(GpuRegister rs1, GpuRegister rs2, uint16_t offset);
  // RV32I-U
  [[noreturn]] void Lui(GpuRegister rd, uint32_t imm20);
  [[noreturn]] void Auipc(GpuRegister rd, uint32_t imm20);
  // RV32I-J
  [[noreturn]] void Jal(GpuRegister rd, uint32_t imm20);
  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV64I ///////////////////////////////
  // RV64I-R
  [[noreturn]] void Addw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Subw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Sllw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Srlw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Sraw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  // RV64I-I
  [[noreturn]] void Lwu(GpuRegister rd, GpuRegister rs1, uint16_t imm12);
  [[noreturn]] void Ld(GpuRegister rd, GpuRegister rs1, uint16_t imm12);
  // void Slli(GpuRegister rd, GpuRegister rs1, uint16_t shamt); // Duplicated with RV32I, why?
  // void Srli(GpuRegister rd, GpuRegister rs1, uint16_t shamt); // Duplicated with RV32I, why?
  // void Srai(GpuRegister rd, GpuRegister rs1, uint16_t shamt); // Duplicated with RV32I, why?
  [[noreturn]] void Addiw(GpuRegister rd, GpuRegister rs1, int16_t imm12);
  [[noreturn]] void Slliw(GpuRegister rd, GpuRegister rs1, int16_t shamt);
  [[noreturn]] void Srliw(GpuRegister rd, GpuRegister rs1, int16_t shamt);
  [[noreturn]] void Sraiw(GpuRegister rd, GpuRegister rs1, int16_t shamt);
  // RV64I-S
  [[noreturn]] void Sd(GpuRegister rs2, GpuRegister rs1, uint16_t imm12);
  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV32M ///////////////////////////////
  // RV32M-R
  [[noreturn]] void Mul(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Mulh(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Mulhsu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Mulhu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Div(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Divu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Rem(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Remu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV32A ///////////////////////////////
  // TODO confirm aq=? rl=?
  [[noreturn]] void LrW(GpuRegister rd, GpuRegister rs1);
  [[noreturn]] void ScW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoSwapW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoAddW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoXorW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoAndW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoOrW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoMinW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoMaxW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoMinuW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoMaxuW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV64A ///////////////////////////////
  // TODO confirm aq=? rl=?
  [[noreturn]] void LrD(GpuRegister rd, GpuRegister rs1);
  [[noreturn]] void ScD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoSwapD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoAddD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoXorD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoAndD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoOrD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoMinD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoMaxD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoMinuD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  [[noreturn]] void AmoMaxuD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1);
  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV64M ///////////////////////////////
  // RV64M-R
  [[noreturn]] void Mulw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Divw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Divuw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Remw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  [[noreturn]] void Remuw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);

  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV32F ///////////////////////////////
  // RV32F-I
  [[noreturn]] void FLw(FpuRegister rd, GpuRegister rs1, uint16_t offset);
  // RV32F-S
  [[noreturn]] void FSw(FpuRegister rs2, GpuRegister rs1, uint16_t offset);
  // RV32F-R
  [[noreturn]] void FMAddS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  [[noreturn]] void FMSubS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  [[noreturn]] void FNMSubS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  [[noreturn]] void FNMAddS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  [[noreturn]] void FAddS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FSubS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FMulS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FDivS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FSqrtS(FpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FSgnjS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FSgnjnS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FSgnjxS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FMinS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FMaxS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FCvtWS(GpuRegister rd, FpuRegister rs1, FPRoundingMode frm = FRM);
  [[noreturn]] void FCvtWuS(GpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FMvXW(GpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FEqS(GpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FLtS(GpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FLeS(GpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FClassS(GpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FCvtSW(FpuRegister rd, GpuRegister rs1);
  [[noreturn]] void FCvtSWu(FpuRegister rd, GpuRegister rs1);
  [[noreturn]] void FMvWX(FpuRegister rd, GpuRegister rs1);
  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV64F ///////////////////////////////
  // RV64F-R
  [[noreturn]] void FCvtLS(GpuRegister rd, FpuRegister rs1, FPRoundingMode frm = FRM);
  [[noreturn]] void FCvtLuS(GpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FCvtSL(FpuRegister rd, GpuRegister rs1);
  [[noreturn]] void FCvtSLu(FpuRegister rd, GpuRegister rs1);
  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV32D ///////////////////////////////
  // RV32D-I
  [[noreturn]] void FLd(FpuRegister rd, GpuRegister rs1, uint16_t offset);
  // RV32D-S
  [[noreturn]] void FSd(FpuRegister rs2, GpuRegister rs1, uint16_t offset);
  // RV32D-R
  [[noreturn]] void FMAddD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  [[noreturn]] void FMSubD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  [[noreturn]] void FNMSubD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  [[noreturn]] void FNMAddD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  [[noreturn]] void FAddD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FSubD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FMulD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FDivD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FSqrtD(FpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FSgnjD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FSgnjnD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FSgnjxD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FMinD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FMaxD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FCvtSD(FpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FCvtDS(FpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FEqD(GpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FLtD(GpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FLeD(GpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  [[noreturn]] void FClassD(GpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FCvtWD(GpuRegister rd, FpuRegister rs1, FPRoundingMode frm = FRM);
  [[noreturn]] void FCvtWuD(GpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FCvtDW(FpuRegister rd, GpuRegister rs1);
  [[noreturn]] void FCvtDWu(FpuRegister rd, GpuRegister rs1);
  ///////////////////////////////////////////////////////////////////

  /////////////////////////////// RV64D ///////////////////////////////
  [[noreturn]] void FCvtLD(GpuRegister rd, FpuRegister rs1, FPRoundingMode frm = FRM);
  [[noreturn]] void FCvtLuD(GpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FMvXD(GpuRegister rd, FpuRegister rs1);
  [[noreturn]] void FCvtDL(FpuRegister rd, GpuRegister rs1);
  [[noreturn]] void FCvtDLu(FpuRegister rd, GpuRegister rs1);
  [[noreturn]] void FMvDX(FpuRegister rd, GpuRegister rs1);
  ///////////////////////////////////////////////////////////////////

  // Helper for replicating floating point value in all destination elements.
  [[noreturn]] void ReplicateFPToVectorRegister(VectorRegister dst, FpuRegister src, bool is_double);

  // Higher level composite instructions.
  int InstrCountForLoadReplicatedConst32(int64_t);
  [[noreturn]] void LoadConst32(GpuRegister rd, int32_t value);
  [[noreturn]] void LoadConst64(GpuRegister rd, int64_t value);  // RISCV64

  // This function is only used for testing purposes.
  [[noreturn]] void RecordLoadConst64Path(int value);

  [[noreturn]] void Addiu32(GpuRegister rt, GpuRegister rs, int32_t value);
  [[noreturn]] void Daddiu64(GpuRegister rt, GpuRegister rs, int64_t value, GpuRegister rtmp = AT);  // RISCV64

  //
  // Heap poisoning.
  //

  // Poison a heap reference contained in `src` and store it in `dst`.
  [[noreturn]] void PoisonHeapReference(GpuRegister dst, GpuRegister src) {
    // dst = -src.
    // Negate the 32-bit ref.
    Dsubu(dst, ZERO, src);
    // And constrain it to 32 bits (zero-extend into bits 32 through 63) as on Arm64 and x86/64.
    Dext(dst, dst, 0, 32);
  }
  // Poison a heap reference contained in `reg`.
  [[noreturn]] void PoisonHeapReference(GpuRegister reg) {
    // reg = -reg.
    PoisonHeapReference(reg, reg);
  }
  // Unpoison a heap reference contained in `reg`.
  [[noreturn]] void UnpoisonHeapReference(GpuRegister reg) {
    // reg = -reg.
    // Negate the 32-bit ref.
    Sub(reg, ZERO, reg);
    // And constrain it to 32 bits (zero-extend into bits 32 through 63) as on Arm64 and x86/64.
    Addiw(reg, reg, 0);
  }
  // Poison a heap reference contained in `reg` if heap poisoning is enabled.
  void MaybePoisonHeapReference(GpuRegister reg) {
    if (kPoisonHeapReferences) {
      PoisonHeapReference(reg);
    }
  }

  // Unpoison a heap reference contained in `reg` if heap poisoning is enabled.
  void MaybeUnpoisonHeapReference(GpuRegister reg) {
    if (kPoisonHeapReferences) {
      UnpoisonHeapReference(reg);
    }
  }

  [[noreturn]] void Bind(Label* label) override {
    Bind(down_cast<Riscv64Label*>(label));
  }
  
  void Jump(Label* label ATTRIBUTE_UNUSED) override {
    UNIMPLEMENTED(FATAL) << "Do not use Jump for RISCV64";
  }

  [[noreturn]] void Bind(Riscv64Label* label);

  // Don't warn about a different virtual Bind/Jump in the base class.
  using JNIBase::Bind;
  using JNIBase::Jump;

  // Create a new label that can be used with Jump/Bind calls.
  std::unique_ptr<JNIMacroLabel> CreateLabel() override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }

  virtual void TestGcMarking(JNIMacroLabel* label ATTRIBUTE_UNUSED,
                   JNIMacroUnaryCondition cond ATTRIBUTE_UNUSED) override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }

  // Emit a conditional jump to the label by applying a unary condition test to the register.
  [[noreturn]] void Jump(ManagedRegister base ATTRIBUTE_UNUSED, Offset offset ATTRIBUTE_UNUSED) override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }

  // Code at this offset will serve as the target for the Jump call.
  [[noreturn]] void Bind(JNIMacroLabel* label ATTRIBUTE_UNUSED) override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }

  [[noreturn]] void Jump(JNIMacroLabel* label ATTRIBUTE_UNUSED) override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }

  //void ExceptionPoll(size_t stack_adjust ATTRIBUTE_UNUSED) override {
  //  LOG(FATAL) << "Not implemented on RISCV64";
  //  UNREACHABLE();
  //}

  [[noreturn]] void CopyRef(FrameOffset dest ATTRIBUTE_UNUSED,
                       ManagedRegister base ATTRIBUTE_UNUSED,
                       MemberOffset offs ATTRIBUTE_UNUSED,
                       bool unpoison_reference ATTRIBUTE_UNUSED) override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }


  [[noreturn]] void CreateJObject(FrameOffset out_off ATTRIBUTE_UNUSED, FrameOffset spilled_reference_offset ATTRIBUTE_UNUSED,
                             bool null_allowed ATTRIBUTE_UNUSED) override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }

  [[noreturn]] void CreateJObject(ManagedRegister out_reg ATTRIBUTE_UNUSED,
                             FrameOffset spilled_reference_offset ATTRIBUTE_UNUSED,
                             ManagedRegister in_reg ATTRIBUTE_UNUSED,
                             bool null_allowed ATTRIBUTE_UNUSED) override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }

  [[noreturn]] void MoveArguments(ArrayRef<ArgumentLocation> dests ATTRIBUTE_UNUSED, ArrayRef<ArgumentLocation> srcs ATTRIBUTE_UNUSED) override {
    LOG(FATAL) << "Not implemented on RISCV64";
    UNREACHABLE();
  }

  // Create a new literal with a given value.
  // NOTE: Force the template parameter to be explicitly specified.
  template <typename T>
  Literal* NewLiteral(typename Identity<T>::type value) {
    static_assert(std::is_integral<T>::value, "T must be an integral type.");
    return NewLiteral(sizeof(value), reinterpret_cast<const uint8_t*>(&value));
  }

  // Load label address using PC-relative loads. To be used with data labels in the literal /
  // jump table area only and not with regular code labels.
  [[noreturn]] void LoadLabelAddress(GpuRegister dest_reg, Riscv64Label* label);

  // Create a new literal with the given data.
  Literal* NewLiteral(size_t size, const uint8_t* data);

  // Load literal using PC-relative loads.
  [[noreturn]] void LoadLiteral(GpuRegister dest_reg, LoadOperandType load_type, Literal* literal);

  // Create a jump table for the given labels that will be emitted when finalizing.
  // When the table is emitted, offsets will be relative to the location of the table.
  // The table location is determined by the location of its label (the label precedes
  // the table data) and should be loaded using LoadLabelAddress().
  JumpTable* CreateJumpTable(std::vector<Riscv64Label*>&& labels);

  // When `is_bare` is false, the branches will promote to long (if the range
  // of the individual branch instruction is insufficient) and the delay/
  // forbidden slots will be taken care of.
  // Use `is_bare = false` when the branch target may be out of reach of the
  // individual branch instruction. IOW, this is for general purpose use.
  //
  // When `is_bare` is true, just the branch instructions will be generated
  // leaving delay/forbidden slot filling up to the caller and the branches
  // won't promote to long if the range is insufficient (you'll get a
  // compilation error when the range is exceeded).
  // Use `is_bare = true` when the branch target is known to be within reach
  // of the individual branch instruction. This is intended for small local
  // optimizations around delay/forbidden slots.
  // Also prefer using `is_bare = true` if the code near the branch is to be
  // patched or analyzed at run time (e.g. introspection) to
  // - show the intent and
  // - fail during compilation rather than during patching/execution if the
  //   bare branch range is insufficent but the code size and layout are
  //   expected to remain unchanged
  //
  // R6 compact branches without delay/forbidden slots.
  [[noreturn]] void Bc(Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Balc(Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Jal(Riscv64Label* label, bool is_bare = false);
  // R6 compact branches with forbidden slots.
  [[noreturn]] void Bltc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Bltzc(GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Bgtzc(GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Bgec(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Bgezc(GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Blezc(GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Bltuc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Bgeuc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Beqc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Bnec(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Beqzc(GpuRegister rs, Riscv64Label* label, bool is_bare = false);
  [[noreturn]] void Bnezc(GpuRegister rs, Riscv64Label* label, bool is_bare = false);

  [[noreturn]] void Bltz(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Bgtz(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Bgez(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Blez(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Jal(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Beq(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Bne(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Blt(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Bge(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Bltu(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Bgeu(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Beqz(GpuRegister rs, Riscv64Label* label, bool is_bare = false);  // R2
  [[noreturn]] void Bnez(GpuRegister rs, Riscv64Label* label, bool is_bare = false);  // R2

  [[noreturn]] void EmitLoad(ManagedRegister m_dst, GpuRegister src_register, int32_t src_offset, size_t size);
  [[noreturn]] void AdjustBaseAndOffset(GpuRegister& base, int32_t& offset, bool is_doubleword);
  // If element_size_shift is negative at entry, its value will be calculated based on the offset.
  [[noreturn]] void AdjustBaseOffsetAndElementSizeShift(GpuRegister& base,
                                           int32_t& offset,
                                           int& element_size_shift);

 private:
  // This will be used as an argument for loads/stores
  // when there is no need for implicit null checks.
  struct NoImplicitNullChecker {
    void operator()() const {}
  };

 public:
  template <typename ImplicitNullChecker = NoImplicitNullChecker>
  [[noreturn]] void StoreConstToOffset(StoreOperandType type,
                          int64_t value,
                          GpuRegister base,
                          int32_t offset,
                          GpuRegister temp,
                          ImplicitNullChecker null_checker = NoImplicitNullChecker()) {
    // We permit `base` and `temp` to coincide (however, we check that neither is AT),
    // in which case the `base` register may be overwritten in the process.
    CHECK_NE(temp, AT);  // Must not use AT as temp, so as not to overwrite the adjusted base.
    AdjustBaseAndOffset(base, offset, /* is_doubleword= */ (type == kStoreDoubleword));
    GpuRegister reg;
    // If the adjustment left `base` unchanged and equal to `temp`, we can't use `temp`
    // to load and hold the value but we can use AT instead as AT hasn't been used yet.
    // Otherwise, `temp` can be used for the value. And if `temp` is the same as the
    // original `base` (that is, `base` prior to the adjustment), the original `base`
    // register will be overwritten.
    if (base == temp) {
      temp = AT;
    }

    if (type == kStoreDoubleword && IsAligned<kRiscv64DoublewordSize>(offset)) {
      if (value == 0) {
        reg = ZERO;
      } else {
        reg = temp;
        LoadConst64(reg, value);
      }
      Sd(reg, base, offset);
      null_checker();
    } else {
      uint32_t low = Low32Bits(value);
      uint32_t high = High32Bits(value);
      if (low == 0) {
        reg = ZERO;
      } else {
        reg = temp;
        LoadConst32(reg, low);
      }
      switch (type) {
        case kStoreByte:
          Sb(reg, base, offset);
          break;
        case kStoreHalfword:
          Sh(reg, base, offset);
          break;
        case kStoreWord:
          Sw(reg, base, offset);
          break;
        case kStoreDoubleword:
          // not aligned to kRiscv64DoublewordSize
          CHECK_ALIGNED(offset, kRiscv64WordSize);
          Sw(reg, base, offset);
          null_checker();
          if (high == 0) {
            reg = ZERO;
          } else {
            reg = temp;
            if (high != low) {
              LoadConst32(reg, high);
            }
          }
          Sw(reg, base, offset + kRiscv64WordSize);
          break;
        default:
          LOG(FATAL) << "UNREACHABLE";
      }
      if (type != kStoreDoubleword) {
        null_checker();
      }
    }
  }

  template <typename ImplicitNullChecker = NoImplicitNullChecker>
  [[noreturn]] void LoadFromOffset(LoadOperandType type,
                      GpuRegister reg,
                      GpuRegister base,
                      int32_t offset,
                      ImplicitNullChecker null_checker = NoImplicitNullChecker()) {
    AdjustBaseAndOffset(base, offset, /* is_doubleword= */ (type == kLoadDoubleword));

    switch (type) {
      case kLoadSignedByte:
        Lb(reg, base, offset);
        break;
      case kLoadUnsignedByte:
        Lbu(reg, base, offset);
        break;
      case kLoadSignedHalfword:
        Lh(reg, base, offset);
        break;
      case kLoadUnsignedHalfword:
        Lhu(reg, base, offset);
        break;
      case kLoadWord:
        CHECK_ALIGNED(offset, kRiscv64WordSize);
        Lw(reg, base, offset);
        break;
      case kLoadUnsignedWord:
        CHECK_ALIGNED(offset, kRiscv64WordSize);
        Lwu(reg, base, offset);
        break;
      case kLoadDoubleword:
        Ld(reg, base, offset);
        null_checker();
        break;
      default:
        LOG(FATAL) << "UNREACHABLE";
    }
    if (type != kLoadDoubleword) {
      null_checker();
    }
  }

  template <typename ImplicitNullChecker = NoImplicitNullChecker>
  [[noreturn]] void LoadFpuFromOffset(LoadOperandType type,
                         FpuRegister reg,
                         GpuRegister base,
                         int32_t offset,
                         ImplicitNullChecker null_checker = NoImplicitNullChecker()) {
    // int element_size_shift = -1;
    if (type != kLoadQuadword) {
      AdjustBaseAndOffset(base, offset, /* is_doubleword= */ (type == kLoadDoubleword));
    } else {
      // AdjustBaseOffsetAndElementSizeShift(base, offset, element_size_shift);
    }

    switch (type) {
      case kLoadWord:
        CHECK_ALIGNED(offset, kRiscv64WordSize);
        FLw(reg, base, offset);
        null_checker();
        break;
      case kLoadDoubleword:
        FLd(reg, base, offset);
        null_checker();
        break;
      case kLoadQuadword:
        UNIMPLEMENTED(FATAL) << "store kStoreQuadword not implemented";
        break;
      default:
        LOG(FATAL) << "UNREACHABLE";
    }
  }

  template <typename ImplicitNullChecker = NoImplicitNullChecker>
  [[noreturn]] void StoreToOffset(StoreOperandType type,
                     GpuRegister reg,
                     GpuRegister base,
                     int32_t offset,
                     ImplicitNullChecker null_checker = NoImplicitNullChecker()) {
    // Must not use AT as `reg`, so as not to overwrite the value being stored
    // with the adjusted `base`.
    CHECK_NE(reg, AT);
    AdjustBaseAndOffset(base, offset, /* is_doubleword= */ (type == kStoreDoubleword));

    switch (type) {
      case kStoreByte:
        Sb(reg, base, offset);
        break;
      case kStoreHalfword:
        Sh(reg, base, offset);
        break;
      case kStoreWord:
        CHECK_ALIGNED(offset, kRiscv64WordSize);
        Sw(reg, base, offset);
        break;
      case kStoreDoubleword:
        Sd(reg, base, offset);
        null_checker();
        break;
      default:
        LOG(FATAL) << "UNREACHABLE";
    }
    if (type != kStoreDoubleword) {
      null_checker();
    }
  }

  template <typename ImplicitNullChecker = NoImplicitNullChecker>
  [[noreturn]] void StoreFpuToOffset(StoreOperandType type,
                        FpuRegister reg,
                        GpuRegister base,
                        int32_t offset,
                        ImplicitNullChecker null_checker = NoImplicitNullChecker()) {
    // int element_size_shift = -1;
    if (type != kStoreQuadword) {
      AdjustBaseAndOffset(base, offset, /* is_doubleword= */ (type == kStoreDoubleword));
    } else {
      // AdjustBaseOffsetAndElementSizeShift(base, offset, element_size_shift);
    }

    switch (type) {
      case kStoreWord:
        CHECK_ALIGNED(offset, kRiscv64WordSize);
        FSw(reg, base, offset);
        null_checker();
        break;
      case kStoreDoubleword:
        FSd(reg, base, offset);
        null_checker();
        break;
      case kStoreQuadword:
        UNIMPLEMENTED(FATAL) << "store kStoreQuadword not implemented";
        null_checker();
        break;
      default:
        LOG(FATAL) << "UNREACHABLE";
    }
  }

  [[noreturn]] void LoadFromOffset(LoadOperandType type, GpuRegister reg, GpuRegister base, int32_t offset);
  [[noreturn]] void LoadFpuFromOffset(LoadOperandType type, FpuRegister reg, GpuRegister base, int32_t offset);
  [[noreturn]] void StoreToOffset(StoreOperandType type, GpuRegister reg, GpuRegister base, int32_t offset);
  [[noreturn]] void StoreFpuToOffset(StoreOperandType type, FpuRegister reg, GpuRegister base, int32_t offset);

  // Emit data (e.g. encoded instruction or immediate) to the instruction stream.
  [[noreturn]] void Emit(uint32_t value);

  //
  // Overridden common assembler high-level functionality.
  //

  // Emit code that will create an activation on the stack.
  [[noreturn]] void BuildFrame(size_t frame_size,
                  ManagedRegister method_reg,
                  ArrayRef<const ManagedRegister> callee_save_regs) override;

  // Emit code that will remove an activation from the stack.
  [[noreturn]] void RemoveFrame(size_t frame_size,
                   ArrayRef<const ManagedRegister> callee_save_regs,
                   bool may_suspend) override;

  [[noreturn]] void IncreaseFrameSize(size_t adjust) override;
  [[noreturn]] void DecreaseFrameSize(size_t adjust) override;

  // Store routines.
  [[noreturn]] void Store(FrameOffset offs, ManagedRegister msrc, size_t size) override;
  [[noreturn]] void StoreRef(FrameOffset dest, ManagedRegister msrc) override;
  [[noreturn]] void StoreRawPtr(FrameOffset dest, ManagedRegister msrc) override;

  [[noreturn]] void StoreImmediateToFrame(FrameOffset dest, uint32_t imm) override;

  [[noreturn]] void StoreStackOffsetToThread(ThreadOffset64 thr_offs,
                                        FrameOffset fr_offs) override;

  [[noreturn]] void StoreStackPointerToThread(ThreadOffset64 thr_offs) override;

  [[noreturn]] void StoreSpanning(FrameOffset dest, ManagedRegister msrc, FrameOffset in_off) override;

  // Load routines.
  [[noreturn]] void Load(ManagedRegister mdest, FrameOffset src, size_t size) override;

  [[noreturn]] void LoadFromThread(ManagedRegister mdest, ThreadOffset64 src, size_t size) override;

  [[noreturn]] void LoadRef(ManagedRegister dest, FrameOffset src) override;

  [[noreturn]] void LoadRef(ManagedRegister mdest, ManagedRegister base, MemberOffset offs,
               bool unpoison_reference) override;

  [[noreturn]] void LoadRawPtr(ManagedRegister mdest, ManagedRegister base, Offset offs) override;

  [[noreturn]] void LoadRawPtrFromThread(ManagedRegister mdest, ThreadOffset64 offs) override;

  // Copying routines.
  [[noreturn]] void Move(ManagedRegister mdest, ManagedRegister msrc, size_t size) override;

  [[noreturn]] void CopyRawPtrFromThread(FrameOffset fr_offs,
                            ThreadOffset64 THREAD_FLAGS_OFFSET) override;

  [[noreturn]] void CopyRawPtrToThread(ThreadOffset64 thr_offs,
                          FrameOffset fr_offs,
                          ManagedRegister mscratch) override;

  [[noreturn]] void CopyRef(FrameOffset dest, FrameOffset src) override;

  [[noreturn]] void Copy(FrameOffset dest, FrameOffset src, size_t size) override;

  [[noreturn]] void Copy(FrameOffset dest, ManagedRegister src_base, Offset src_offset,
            ManagedRegister scratch, size_t size) override;

  [[noreturn]] void Copy(ManagedRegister dest_base, Offset dest_offset, FrameOffset src,
            ManagedRegister scratch, size_t size) override;

  [[noreturn]] void Copy(FrameOffset dest, FrameOffset src_base, Offset src_offset,
            ManagedRegister scratch, size_t size) override;

  [[noreturn]] void Copy(ManagedRegister dest, Offset dest_offset, ManagedRegister src, Offset src_offset,
            ManagedRegister scratch, size_t size) override;

  [[noreturn]] void Copy(FrameOffset dest, Offset dest_offset, FrameOffset src, Offset src_offset,
            ManagedRegister scratch, size_t size) override;

  [[noreturn]] void MemoryBarrier(ManagedRegister) override;

  // Sign extension.
  [[noreturn]] void SignExtend(ManagedRegister mreg, size_t size) override;

  // Zero extension.
  [[noreturn]] void ZeroExtend(ManagedRegister mreg, size_t size) override;

  // Exploit fast access in managed code to Thread::Current().
  [[noreturn]] void GetCurrentThread(ManagedRegister tr) override;
  [[noreturn]] void GetCurrentThread(FrameOffset dest_offset) override;

  // Set up out_reg to hold a Object** into the handle scope, or to be null if the
  // value is null and null_allowed. in_reg holds a possibly stale reference
  // that can be used to avoid loading the handle scope entry to see if the value is
  // null.
  [[noreturn]] void CreateHandleScopeEntry(ManagedRegister out_reg, FrameOffset handlescope_offset,
                              ManagedRegister in_reg, bool null_allowed);

  // Set up out_off to hold a Object** into the handle scope, or to be null if the
  // value is null and null_allowed.
  [[noreturn]] void CreateHandleScopeEntry(FrameOffset out_off, FrameOffset handlescope_offset, ManagedRegister
                              mscratch, bool null_allowed);

  // src holds a handle scope entry (Object**) load this into dst.
  [[noreturn]] void LoadReferenceFromHandleScope(ManagedRegister dst, ManagedRegister src);

  // Heap::VerifyObject on src. In some cases (such as a reference to this) we
  // know that src may not be null.
  [[noreturn]] void VerifyObject(ManagedRegister src, bool could_be_null) override;
  [[noreturn]] void VerifyObject(FrameOffset src, bool could_be_null) override;

  // Call to address held at [base+offset].
  [[noreturn]] void Call(ManagedRegister base, Offset offset) override;
  [[noreturn]] void Call(FrameOffset base, Offset offset) override;
  [[noreturn]] void CallFromThread(ThreadOffset64 offset) override;

  // Generate code to check if Thread::Current()->exception_ is non-null
  // and branch to a ExceptionSlowPath if it is.
  [[noreturn]] void ExceptionPoll(size_t stack_adjust) override;

  // Emit slow paths queued during assembly and promote short branches to long if needed.
  [[noreturn]] void FinalizeCode() override;

  // Emit branches and finalize all instructions.
  [[noreturn]] void FinalizeInstructions(const MemoryRegion& region) override;

  // Returns the (always-)current location of a label (can be used in class CodeGeneratorRISCV64,
  // must be used instead of Riscv64Label::GetPosition()).
  uint32_t GetLabelLocation(const Riscv64Label* label) const;

  // Get the final position of a label after local fixup based on the old position
  // recorded before FinalizeCode().
  uint32_t GetAdjustedPosition(uint32_t old_position);

  // Note that PC-relative literal loads are handled as pseudo branches because they need very
  // similar relocation and may similarly expand in size to accomodate for larger offsets relative
  // to PC.
  enum BranchCondition {
    kCondLT,
    kCondGE,
    kCondLE,
    kCondGT,
    kCondLTZ,
    kCondGEZ,
    kCondLEZ,
    kCondGTZ,
    kCondEQ,
    kCondNE,
    kCondEQZ,
    kCondNEZ,
    kCondLTU,
    kCondGEU,
    kUncond,
  };
  friend std::ostream& operator<<(std::ostream& os, const BranchCondition& rhs);

 private:
  class Branch {
   public:
    enum Type {
      // R6 short branches (can be promoted to long).
      kUncondBranch,
      kCondBranch,
      kCall,
      // R6 short branches (can't be promoted to long), forbidden/delay slots filled manually.
      kBareUncondBranch,
      kBareCondBranch,
      kBareCall,
      // label.
      kLabel,
      // literals.
      kLiteral,
      kLiteralUnsigned,
      kLiteralLong,
      // Long branches.
      kLongUncondBranch,
      kLongCondBranch,
      kLongCall,
    };

    // Bit sizes of offsets defined as enums to minimize chance of typos.
    enum OffsetBits {
      kOffset12 = 12,  // reserved for jalr
      kOffset13 = 13,
      kOffset21 = 21,
      kOffset32 = 32,
    };

    static constexpr uint32_t kUnresolved = 0xffffffff;  // Unresolved target_
    static constexpr int32_t kMaxBranchLength = 32;
    static constexpr int32_t kMaxBranchSize = kMaxBranchLength * sizeof(uint32_t);

    struct BranchInfo {
      // Branch length as a number of 4-byte-long instructions.
      uint32_t length;
      // Ordinal number (0-based) of the first (or the only) instruction that contains the branch's
      // PC-relative offset (or its most significant 16-bit half, which goes first).
      uint32_t instr_offset;
      // Different MIPS instructions with PC-relative offsets apply said offsets to slightly
      // different origins, e.g. to PC or PC+4. Encode the origin distance (as a number of 4-byte
      // instructions) from the instruction containing the offset.
      uint32_t pc_org;
      // How large (in bits) a PC-relative offset can be for a given type of branch (kCondBranch
      // and kBareCondBranch are an exception: use kOffset23 for beqzc/bnezc).
      OffsetBits offset_size;
      // Some MIPS instructions with PC-relative offsets shift the offset by 2. Encode the shift
      // count.
      int offset_shift;
    };
    static const BranchInfo branch_info_[/* Type */];

    // Unconditional branch or call.
    Branch(uint32_t location, uint32_t target, bool is_call, bool is_bare);
    // Conditional branch.
    Branch(uint32_t location,
           uint32_t target,
           BranchCondition condition,
           GpuRegister lhs_reg,
           GpuRegister rhs_reg,
           bool is_bare);
    // Label address (in literal area) or literal.
    Branch(uint32_t location, GpuRegister dest_reg, Type label_or_literal_type);

    // Some conditional branches with lhs = rhs are effectively NOPs, while some
    // others are effectively unconditional. MIPSR6 conditional branches require lhs != rhs.
    // So, we need a way to identify such branches in order to emit no instructions for them
    // or change them to unconditional.
    static bool IsNop(BranchCondition condition, GpuRegister lhs, GpuRegister rhs);
    static bool IsUncond(BranchCondition condition, GpuRegister lhs, GpuRegister rhs);

    static BranchCondition OppositeCondition(BranchCondition cond);

    Type GetType() const;
    BranchCondition GetCondition() const;
    GpuRegister GetLeftRegister() const;
    GpuRegister GetRightRegister() const;
    uint32_t GetTarget() const;
    uint32_t GetLocation() const;
    uint32_t GetOldLocation() const;
    uint32_t GetLength() const;
    uint32_t GetOldLength() const;
    uint32_t GetSize() const;
    uint32_t GetOldSize() const;
    uint32_t GetEndLocation() const;
    uint32_t GetOldEndLocation() const;
    bool IsBare() const;
    bool IsLong() const;
    bool IsResolved() const;

    // Returns the bit size of the signed offset that the branch instruction can handle.
    OffsetBits GetOffsetSize() const;

    // Calculates the distance between two byte locations in the assembler buffer and
    // returns the number of bits needed to represent the distance as a signed integer.
    //
    // Branch instructions have signed offsets of 16, 19 (addiupc), 21 (beqzc/bnezc),
    // and 26 (bc) bits, which are additionally shifted left 2 positions at run time.
    //
    // Composite branches (made of several instructions) with longer reach have 32-bit
    // offsets encoded as 2 16-bit "halves" in two instructions (high half goes first).
    // The composite branches cover the range of PC + ~+/-2GB. The range is not end-to-end,
    // however. Consider the following implementation of a long unconditional branch, for
    // example:
    //
    //   auipc at, offset_31_16  // at = pc + sign_extend(offset_31_16) << 16
    //   jic   at, offset_15_0   // pc = at + sign_extend(offset_15_0)
    //
    // Both of the above instructions take 16-bit signed offsets as immediate operands.
    // When bit 15 of offset_15_0 is 1, it effectively causes subtraction of 0x10000
    // due to sign extension. This must be compensated for by incrementing offset_31_16
    // by 1. offset_31_16 can only be incremented by 1 if it's not 0x7FFF. If it is
    // 0x7FFF, adding 1 will overflow the positive offset into the negative range.
    // Therefore, the long branch range is something like from PC - 0x80000000 to
    // PC + 0x7FFF7FFF, IOW, shorter by 32KB on one side.
    //
    // The returned values are therefore: 18, 21, 23, 28 and 32. There's also a special
    // case with the addiu instruction and a 16 bit offset.
    static OffsetBits GetOffsetSizeNeeded(uint32_t location, uint32_t target);

    // Resolve a branch when the target is known.
    [[noreturn]] void Resolve(uint32_t target);

    // Relocate a branch by a given delta if needed due to expansion of this or another
    // branch at a given location by this delta (just changes location_ and target_).
    [[noreturn]] void Relocate(uint32_t expand_location, uint32_t delta);

    // If the branch is short, changes its type to long.
    [[noreturn]] void PromoteToLong();

    // If necessary, updates the type by promoting a short branch to a long branch
    // based on the branch location and target. Returns the amount (in bytes) by
    // which the branch size has increased.
    // max_short_distance caps the maximum distance between location_ and target_
    // that is allowed for short branches. This is for debugging/testing purposes.
    // max_short_distance = 0 forces all short branches to become long.
    // Use the implicit default argument when not debugging/testing.
    uint32_t PromoteIfNeeded(uint32_t max_short_distance = std::numeric_limits<uint32_t>::max());

    // Returns the location of the instruction(s) containing the offset.
    uint32_t GetOffsetLocation() const;

    // Calculates and returns the offset ready for encoding in the branch instruction(s).
    uint32_t GetOffset() const;

   private:
    // Completes branch construction by determining and recording its type.
    [[noreturn]] void InitializeType(Type initial_type);
    // Helper for the above.
    [[noreturn]] void InitShortOrLong(OffsetBits ofs_size, Type short_type, Type long_type);

    uint32_t old_location_;      // Offset into assembler buffer in bytes.
    uint32_t location_;          // Offset into assembler buffer in bytes.
    uint32_t target_;            // Offset into assembler buffer in bytes.

    GpuRegister lhs_reg_;        // Left-hand side register in conditional branches or
                                 // destination register in literals.
    GpuRegister rhs_reg_;        // Right-hand side register in conditional branches.
    BranchCondition condition_;  // Condition for conditional branches.

    Type type_;                  // Current type of the branch.
    Type old_type_;              // Initial type of the branch.
  };
  friend std::ostream& operator<<(std::ostream& os, const Branch::Type& rhs);
  friend std::ostream& operator<<(std::ostream& os, const Branch::OffsetBits& rhs);

  [[noreturn]] void EmitRsd(int opcode, GpuRegister rs, GpuRegister rd, int shamt, int funct);
  [[noreturn]] void EmitRtd(int opcode, GpuRegister rt, GpuRegister rd, int shamt, int funct);
  [[noreturn]] void EmitI(int opcode, GpuRegister rs, GpuRegister rt, uint16_t imm);
  [[noreturn]] void EmitI21(int opcode, GpuRegister rs, uint32_t imm21);
  [[noreturn]] void EmitI26(int opcode, uint32_t imm26);
  [[noreturn]] void EmitFR(int opcode, int fmt, FpuRegister ft, FpuRegister fs, FpuRegister fd, int funct);
  [[noreturn]] void EmitFI(int opcode, int fmt, FpuRegister rt, uint16_t imm);
  [[noreturn]] void EmitBcond(BranchCondition cond, GpuRegister rs, GpuRegister rt, uint32_t imm16_21);
  [[noreturn]] void EmitMsa3R(int operation,
                 int df,
                 VectorRegister wt,
                 VectorRegister ws,
                 VectorRegister wd,
                 int minor_opcode);
  [[noreturn]] void EmitMsaBIT(int operation, int df_m, VectorRegister ws, VectorRegister wd, int minor_opcode);
  [[noreturn]] void EmitMsaELM(int operation, int df_n, VectorRegister ws, VectorRegister wd, int minor_opcode);
  [[noreturn]] void EmitMsaMI10(int s10, GpuRegister rs, VectorRegister wd, int minor_opcode, int df);
  [[noreturn]] void EmitMsaI10(int operation, int df, int i10, VectorRegister wd, int minor_opcode);
  [[noreturn]] void EmitMsa2R(int operation, int df, VectorRegister ws, VectorRegister wd, int minor_opcode);
  [[noreturn]] void EmitMsa2RF(int operation, int df, VectorRegister ws, VectorRegister wd, int minor_opcode);

  // TODO dvt porting...
  template<typename Reg1, typename Reg2, typename Reg3>
  [[noreturn]] void EmitR(int funct7, Reg1 rs2, Reg2 rs1, int funct3, Reg3 rd, int opcode) {
    // TODO validate params
    uint32_t encoding = static_cast<uint32_t>(funct7) << 25 |
                        static_cast<uint32_t>(rs2) << 20 |
                        static_cast<uint32_t>(rs1) << 15 |
                        static_cast<uint32_t>(funct3) << 12 |
                        static_cast<uint32_t>(rd) << 7 |
                        opcode;
    Emit(encoding);
  }

  template<typename Reg1, typename Reg2, typename Reg3, typename Reg4>
  [[noreturn]] void EmitR4(Reg1 rs3, int funct2, Reg2 rs2, Reg3 rs1, int funct3, Reg4 rd, int opcode) {
    // TODO validate params
    uint32_t encoding = static_cast<uint32_t>(rs3) << 27 |
                        static_cast<uint32_t>(funct2) << 25 |
                        static_cast<uint32_t>(rs2) << 20 |
                        static_cast<uint32_t>(rs1) << 15 |
                        static_cast<uint32_t>(funct3) << 12 |
                        static_cast<uint32_t>(rd) << 7 |
                        opcode;
    Emit(encoding);
  }

  template<typename Reg1, typename Reg2>
  [[noreturn]] void EmitI(uint16_t imm, Reg1 rs1, int funct3, Reg2 rd, int opcode) {
    uint32_t encoding = static_cast<uint32_t>(imm) << 20 |
                        static_cast<uint32_t>(rs1) << 15 |
                        static_cast<uint32_t>(funct3) << 12 |
                        static_cast<uint32_t>(rd) << 7 |
                        opcode;
    Emit(encoding);
  }

  [[noreturn]] void EmitI5(uint16_t funct7, uint16_t imm5, GpuRegister rs1, int funct3, GpuRegister rd, int opcode);
  [[noreturn]] void EmitI6(uint16_t funct6, uint16_t imm6, GpuRegister rs1, int funct3, GpuRegister rd, int opcode);

  template<typename Reg1, typename Reg2>
  [[noreturn]] void EmitS(uint16_t imm, Reg1 rs2, Reg2 rs1, int funct3, int opcode) {
    // TODO validate params
    uint32_t encoding = (static_cast<uint32_t>(imm)&0xFE0) << 20 |
                        static_cast<uint32_t>(rs2) << 20 |
                        static_cast<uint32_t>(rs1) << 15 |
                        static_cast<uint32_t>(funct3) << 12 |
                        (static_cast<uint32_t>(imm)&0x1F) << 7 |
                      opcode;
    Emit(encoding);
  }

  [[noreturn]] void EmitB(uint16_t imm, GpuRegister rs2, GpuRegister rs1, int funct3, int opcode);
  [[noreturn]] void EmitU(uint32_t imm, GpuRegister rd, int opcode);
  [[noreturn]] void EmitJ(uint32_t imm, GpuRegister rd, int opcode);

  [[noreturn]] void Buncond(Riscv64Label* label, bool is_bare);
  [[noreturn]] void Bcond(Riscv64Label* label,
             bool is_bare,
             BranchCondition condition,
             GpuRegister lhs,
             GpuRegister rhs = ZERO);
  [[noreturn]] void Call(Riscv64Label* label, bool is_bare);
  [[noreturn]] void FinalizeLabeledBranch(Riscv64Label* label);

  Branch* GetBranch(uint32_t branch_id);
  const Branch* GetBranch(uint32_t branch_id) const;

  [[noreturn]] void EmitLiterals();
  [[noreturn]] void ReserveJumpTableSpace();
  [[noreturn]] void EmitJumpTables();
  [[noreturn]] void PromoteBranches();
  [[noreturn]] void EmitBranch(Branch* branch);
  [[noreturn]] void EmitBranches();
  [[noreturn]] void PatchCFI();

  // Emits exception block.
  [[noreturn]] void EmitExceptionPoll(Riscv64ExceptionSlowPath* exception);

  bool HasMsa() const {
    return has_msa_;
  }

  // List of exception blocks to generate at the end of the code cache.
  std::vector<Riscv64ExceptionSlowPath> exception_blocks_;

  std::vector<Branch> branches_;

  // Whether appending instructions at the end of the buffer or overwriting the existing ones.
  bool overwriting_;
  // The current overwrite location.
  uint32_t overwrite_location_;

  // Use std::deque<> for literal labels to allow insertions at the end
  // without invalidating pointers and references to existing elements.
  ArenaDeque<Literal> literals_;
  ArenaDeque<Literal> long_literals_;  // 64-bit literals separated for alignment reasons.

  // Jump table list.
  ArenaDeque<JumpTable> jump_tables_;

  // Data for AdjustedPosition(), see the description there.
  uint32_t last_position_adjustment_;
  uint32_t last_old_position_;
  uint32_t last_branch_id_;

  const bool has_msa_;

  DISALLOW_COPY_AND_ASSIGN(Riscv64Assembler);
};

}  // namespace riscv64
}  // namespace art

#endif  // ART_COMPILER_UTILS_RISCV64_ASSEMBLER_RISCV64_H_
