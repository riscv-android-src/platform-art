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

#define RISCV64_VARIANTS_HAS_VECTOR

inline uint16_t Low12Bits(uint32_t value) {
  return static_cast<uint16_t>(value & 0xFFF);
}

inline uint32_t High20Bits(uint32_t value) {
  return static_cast<uint32_t>(value >> 12);
}


static constexpr size_t kRiscv64HalfwordSize   = 2;
static constexpr size_t kRiscv64WordSize       = 4;
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
  kNegativeInfinity  = 0x001,
  kNegativeNormal    = 0x002,
  kNegativeSubnormal = 0x004,
  kNegativeZero      = 0x008,
  kPositiveZero      = 0x010,
  kPositiveSubnormal = 0x020,
  kPositiveNormal    = 0x040,
  kPositiveInfinity  = 0x080,
  kSignalingNaN      = 0x100,
  kQuietNaN          = 0x200,
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

class Riscv64Assembler final : public Assembler {
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
        last_branch_id_(0) {
    (void) instruction_set_features;  // XC-TODO, set features, for example CRC32, Vector etc
    cfi().DelayEmittingAdvancePCs();
  }

  virtual ~Riscv64Assembler() {
    for (auto& branch : branches_) {
      CHECK(branch.IsResolved());
    }
  }

  size_t CodeSize() const override { return Assembler::CodeSize(); }
  DebugFrameOpCodeWriterForAssembler& cfi() { return Assembler::cfi(); }

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

  /////////////////////////////// RV64 "IM" Instructions ///////////////////////////////
  // Load instructions: opcode = 0x03, subfunc(func3) from 0x0 ~ 0x6
  void Lb(GpuRegister   rd, GpuRegister rs1, uint16_t offset);
  void Lh(GpuRegister   rd, GpuRegister rs1, uint16_t offset);
  void Lw(GpuRegister   rd, GpuRegister rs1, uint16_t offset);
  void Ld(GpuRegister   rd, GpuRegister rs1, uint16_t offset);
  void Lbu(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
  void Lhu(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
  void Lwu(GpuRegister  rd, GpuRegister rs1, uint16_t offset);

  // Store instructions: opcode = 0x23, subfunc(func3) from 0x0 ~ 0x3
  void Sb(GpuRegister rs2, GpuRegister rs1, uint16_t offset);
  void Sh(GpuRegister rs2, GpuRegister rs1, uint16_t offset);
  void Sw(GpuRegister rs2, GpuRegister rs1, uint16_t offset);
  void Sd(GpuRegister rs2, GpuRegister rs1, uint16_t offset);

  // IMM ALU instructions: opcode = 0x13, subfunc(func3) from 0x0 ~ 0x7
  void Addi(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
  void Slli(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
  void Slti(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
  void Sltiu(GpuRegister rd, GpuRegister rs1, uint16_t offset);
  void Xori(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
  void Srli(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
  void Srai(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
  void Ori(GpuRegister   rd, GpuRegister rs1, uint16_t offset);
  void Andi(GpuRegister  rd, GpuRegister rs1, uint16_t offset);
 
  // ALU instructions: opcode = 0x33, subfunc(func3) from 0x0 ~ 0x7; func7 also changed
  void Add(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Sll(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Slt(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Sltu(GpuRegister  rd, GpuRegister rs1, GpuRegister rs2);
  void Xor(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Srl(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Or(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2);
  void And(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);

  // RV64-M
  void Mul(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2);
  void Mulh(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Mulhsu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  void Mulhu(GpuRegister  rd, GpuRegister rs1, GpuRegister rs2);
  void Div(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2);
  void Divu(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Rem(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2);
  void Remu(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Sub(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2);
  void Sra(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2);

  // 32bit Imm ALU instructions: opcode = 0x1b, subfunc(func3) - 0x0, 0x1, 0x5
  void Addiw(GpuRegister  rd, GpuRegister rs1, int16_t imm12);
  void Slliw(GpuRegister  rd, GpuRegister rs1, int16_t shamt);
  void Srliw(GpuRegister  rd, GpuRegister rs1, int16_t shamt);
  void Sraiw(GpuRegister  rd, GpuRegister rs1, int16_t shamt);

  // 32bit ALU instructions: opcode = 0x3b, subfunc(func3) from 0x0 ~ 0x7
  void Addw(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Mulw(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Subw(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Sllw(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Divw(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Srlw(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Divuw(GpuRegister  rd, GpuRegister rs1, GpuRegister rs2);
  void Sraw(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Remw(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2);
  void Remuw(GpuRegister  rd, GpuRegister rs1, GpuRegister rs2);

  // opcode = 0x17 & 0x37
  void Auipc(GpuRegister  rd, uint32_t imm20);
  void Lui(GpuRegister    rd, uint32_t imm20);

  // Branch and Jump instructions, opcode = 0x63 (subfunc from 0x0 ~ 0x7), 0x67, 0x6f
  void Beq(GpuRegister    rs1, GpuRegister rs2, uint16_t offset);
  void Bne(GpuRegister    rs1, GpuRegister rs2, uint16_t offset);
  void Blt(GpuRegister    rs1, GpuRegister rs2, uint16_t offset);
  void Bge(GpuRegister    rs1, GpuRegister rs2, uint16_t offset);
  void Bltu(GpuRegister   rs1, GpuRegister rs2, uint16_t offset);
  void Bgeu(GpuRegister   rs1, GpuRegister rs2, uint16_t offset);

  void Jalr(GpuRegister   rd, GpuRegister rs1, uint16_t offset);
  void Jal(GpuRegister    rd, uint32_t imm20);

  // opcode - 0xf 0xf and 0x73
  void Fence(uint8_t pred, uint8_t succ);
  void FenceI();
  void Ecall();
  void Ebreak();

  // Control register instructions
  void Csrrw(GpuRegister  rd, GpuRegister rs1, uint16_t csr);
  void Csrrs(GpuRegister  rd, GpuRegister rs1, uint16_t csr);
  void Csrrc(GpuRegister  rd, GpuRegister rs1, uint16_t csr);
  void Csrrwi(GpuRegister rd, uint16_t csr, uint8_t zimm /*imm5*/);
  void Csrrsi(GpuRegister rd, uint16_t csr, uint8_t zimm /*imm5*/);
  void Csrrci(GpuRegister rd, uint16_t csr, uint8_t zimm /*imm5*/);
  /////////////////////////////// RV64 "IM" Instructions  END ///////////////////////////////


  /////////////////////////////// RV64 "A" Instructions  START ///////////////////////////////
  void LrW(GpuRegister rd, GpuRegister rs1, uint8_t aqrl);
  void ScW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoSwapW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoAddW(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoXorW(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoAndW(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoOrW(GpuRegister   rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoMinW(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoMaxW(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoMinuW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoMaxuW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);

  void LrD(GpuRegister rd, GpuRegister rs1, uint8_t aqrl);
  void ScD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoSwapD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoAddD(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoXorD(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoAndD(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoOrD(GpuRegister   rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoMinD(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoMaxD(GpuRegister  rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoMinuD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  void AmoMaxuD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1, uint8_t aqrl);
  /////////////////////////////// RV64 "A" Instructions  END ///////////////////////////////


  /////////////////////////////// RV64 "FD" Instructions  START ///////////////////////////////
  // opcode = 0x07 and 0x27
  void FLw(FpuRegister  rd,  GpuRegister rs1, uint16_t offset);
  void FLd(FpuRegister  rd,  GpuRegister rs1, uint16_t offset);
  void FSw(FpuRegister  rs2, GpuRegister rs1, uint16_t offset);
  void FSd(FpuRegister  rs2, GpuRegister rs1, uint16_t offset);

  // opcode = 0x43, 0x47, 0x4b, 0x4f
  void FMAddS(FpuRegister  rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  void FMAddD(FpuRegister  rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  void FMSubS(FpuRegister  rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  void FMSubD(FpuRegister  rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  void FNMSubS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  void FNMSubD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  void FNMAddS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);
  void FNMAddD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3);

  // opcode = 0x53, funct7 is even for float ops
  void FAddS(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FSubS(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FMulS(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FDivS(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FSgnjS(FpuRegister  rd, FpuRegister rs1, FpuRegister rs2);
  void FSgnjnS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  void FSgnjxS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  void FMinS(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FMaxS(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FCvtSD(FpuRegister  rd, FpuRegister rs1);
  void FSqrtS(FpuRegister  rd, FpuRegister rs1);
  void FEqS(GpuRegister    rd, FpuRegister rs1, FpuRegister rs2);
  void FLtS(GpuRegister    rd, FpuRegister rs1, FpuRegister rs2);
  void FLeS(GpuRegister    rd, FpuRegister rs1, FpuRegister rs2);

  void FCvtWS(GpuRegister  rd, FpuRegister rs1, FPRoundingMode frm = FRM);
  void FCvtWuS(GpuRegister rd, FpuRegister rs1);
  void FCvtLS(GpuRegister  rd, FpuRegister rs1, FPRoundingMode frm = FRM);
  void FCvtLuS(GpuRegister rd, FpuRegister rs1);
  void FCvtSW(FpuRegister  rd, GpuRegister rs1);
  void FCvtSWu(FpuRegister rd, GpuRegister rs1);
  void FCvtSL(FpuRegister  rd, GpuRegister rs1);
  void FCvtSLu(FpuRegister rd, GpuRegister rs1);

  void FMvXW(GpuRegister   rd, FpuRegister rs1);
  void FClassS(GpuRegister rd, FpuRegister rs1);
  void FMvWX(FpuRegister   rd, GpuRegister rs1);

  // opcode = 0x53, funct7 is odd for float ops
  void FAddD(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FSubD(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FMulD(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FDivD(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FSgnjD(FpuRegister  rd, FpuRegister rs1, FpuRegister rs2);
  void FSgnjnD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  void FSgnjxD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2);
  void FMinD(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FMaxD(FpuRegister   rd, FpuRegister rs1, FpuRegister rs2);
  void FCvtDS(FpuRegister  rd, FpuRegister rs1);
  void FSqrtD(FpuRegister  rd, FpuRegister rs1);

  void FLeD(GpuRegister    rd, FpuRegister rs1, FpuRegister rs2);
  void FLtD(GpuRegister    rd, FpuRegister rs1, FpuRegister rs2);
  void FEqD(GpuRegister    rd, FpuRegister rs1, FpuRegister rs2);
  void FCvtWD(GpuRegister  rd, FpuRegister rs1, FPRoundingMode frm = FRM);
  void FCvtWuD(GpuRegister rd, FpuRegister rs1);
  void FCvtLD(GpuRegister  rd, FpuRegister rs1, FPRoundingMode frm = FRM);
  void FCvtLuD(GpuRegister rd, FpuRegister rs1);
  void FCvtDW(FpuRegister  rd, GpuRegister rs1);
  void FCvtDWu(FpuRegister rd, GpuRegister rs1);
  void FCvtDL(FpuRegister  rd, GpuRegister rs1);
  void FCvtDLu(FpuRegister rd, GpuRegister rs1);

  void FMvXD(GpuRegister   rd, FpuRegister rs1);
  void FClassD(GpuRegister rd, FpuRegister rs1);
  void FMvDX(FpuRegister   rd, GpuRegister rs1);

  void MinS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  void MinD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  void MaxS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  void MaxD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  /////////////////////////////// RV64 "FD" Instructions  END ///////////////////////////////


  ////////////////////////////// RV64 MACRO Instructions  START ///////////////////////////////
  void Nop();
  void Move(GpuRegister rd, GpuRegister rs);
  void Clear(GpuRegister rd);
  void Not(GpuRegister rd, GpuRegister rs);
  void Break();
  void Sync(uint32_t stype);

  // ALU
  void Addiuw(GpuRegister  rt, GpuRegister rs, int16_t imm16);
  void Addiu(GpuRegister rt, GpuRegister rs, int16_t imm16);
  void Addiuw32(GpuRegister rt, GpuRegister rs, int32_t value);
  void Addiu64(GpuRegister rt, GpuRegister rs, int64_t value, GpuRegister rtmp = AT);
  
  void Srriw(GpuRegister rd, GpuRegister rs1, int imm5);
  void Srri(GpuRegister rd, GpuRegister rs1, int imm6);
  void Srrw(GpuRegister rd, GpuRegister rt, GpuRegister rs);
  void Srr(GpuRegister rd, GpuRegister rt, GpuRegister rs);

  void Muhh(GpuRegister rd, GpuRegister rs, GpuRegister rt);

  // Large const load
  void Aui(GpuRegister rt, GpuRegister rs, uint16_t imm16);
  void Ahi(GpuRegister rs, uint16_t imm16);
  void Ati(GpuRegister rs, uint16_t imm16);
  void LoadConst32(GpuRegister rd, int32_t value);
  void LoadConst64(GpuRegister rd, int64_t value);

  // shift and add
  void Addsl(GpuRegister rd, GpuRegister rs, GpuRegister rt, int saPlusOne);
  void Extb(GpuRegister rs,  GpuRegister rt, int pos, int size);
  void Extub(GpuRegister rs, GpuRegister rt, int pos, int size);

  // Branches
  void Seleqz(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  void Selnez(GpuRegister rd, GpuRegister rs, GpuRegister rt);
  void Bltc(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  void Bltzc(GpuRegister rt, uint16_t imm12);
  void Bgtzc(GpuRegister rt, uint16_t imm12);
  void Bgec(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  void Bgezc(GpuRegister rt, uint16_t imm12);
  void Blezc(GpuRegister rt, uint16_t imm12);
  void Bltuc(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  void Bgeuc(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  void Beqc(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  void Bnec(GpuRegister rs, GpuRegister rt, uint16_t imm12);
  void Beqzc(GpuRegister rs, uint32_t imm12);
  void Bnezc(GpuRegister rs, uint32_t imm12);
  void Bc(uint32_t imm20);
  void Balc(uint32_t imm20);
  void EmitBcond(BranchCondition cond, GpuRegister rs, GpuRegister rt, uint32_t imm16_21);

  // Jump
  void Jalr(GpuRegister  rd, GpuRegister rs);
  void Jic(GpuRegister   rt, uint16_t imm16);
  void Jalr(GpuRegister  rs);
  void Jialc(GpuRegister rt, uint16_t imm16);
  void Jr(GpuRegister    rs);

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
  void Bc(Riscv64Label* label, bool is_bare = false);
  void Balc(Riscv64Label* label, bool is_bare = false);
  void Jal(Riscv64Label* label, bool is_bare = false);
  // R6 compact branches with forbidden slots.
  void Bltc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Bltzc(GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Bgtzc(GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Bgec(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Bgezc(GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Blezc(GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Bltuc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Bgeuc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Beqc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Bnec(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);
  void Beqzc(GpuRegister rs, Riscv64Label* label, bool is_bare = false);
  void Bnezc(GpuRegister rs, Riscv64Label* label, bool is_bare = false);

  void Bltz(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Bgtz(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Bgez(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Blez(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Jal(GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Beq(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Bne(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Blt(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Bge(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Bltu(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Bgeu(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare = false);  // R2
  void Beqz(GpuRegister rs, Riscv64Label* label, bool is_bare = false);  // R2
  void Bnez(GpuRegister rs, Riscv64Label* label, bool is_bare = false);  // R2

  // atomic
  void Sc(GpuRegister    rt, GpuRegister base);
  void Scd(GpuRegister   rt, GpuRegister base);
  void Ll(GpuRegister    rt, GpuRegister base);
  void Lld(GpuRegister   rt, GpuRegister base);

  // Float
  void AddS(FpuRegister  fd, FpuRegister fs, FpuRegister ft);
  void SubS(FpuRegister  fd, FpuRegister fs, FpuRegister ft);
  void MulS(FpuRegister  fd, FpuRegister fs, FpuRegister ft);
  void DivS(FpuRegister  fd, FpuRegister fs, FpuRegister ft);
  void AbsS(FpuRegister  fd, FpuRegister fs);
  void MovS(FpuRegister  fd, FpuRegister fs);
  void NegS(FpuRegister  fd, FpuRegister fs);
  void SqrtS(FpuRegister fd, FpuRegister fs);

  // Double
  void AddD(FpuRegister   fd, FpuRegister fs, FpuRegister ft);
  void SubD(FpuRegister   fd, FpuRegister fs, FpuRegister ft);
  void MulD(FpuRegister   fd, FpuRegister fs, FpuRegister ft);
  void DivD(FpuRegister   fd, FpuRegister fs, FpuRegister ft);
  void AbsD(FpuRegister   fd, FpuRegister fs);
  void MovD(FpuRegister   fd, FpuRegister fs);
  void NegD(FpuRegister   fd, FpuRegister fs);
  void SqrtD(FpuRegister  fd, FpuRegister fs);

  void Cvtsd(FpuRegister  fd, FpuRegister fs);
  void Cvtds(FpuRegister  fd, FpuRegister fs);

  void TruncLS(GpuRegister rd, FpuRegister fs);
  void TruncLD(GpuRegister rd, FpuRegister fs);
  void TruncWS(GpuRegister rd, FpuRegister fs);
  void TruncWD(GpuRegister rd, FpuRegister fs);


  // float/double compare and branch
  void FJMaxMinS(FpuRegister fd, FpuRegister fs, FpuRegister ft, bool is_min);
  void FJMaxMinD(FpuRegister fd, FpuRegister fs, FpuRegister ft, bool is_min);
  void SelS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  void SelD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  void SeleqzS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  void SeleqzD(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  void SelnezS(FpuRegister fd, FpuRegister fs, FpuRegister ft);
  void SelnezD(FpuRegister fd, FpuRegister fs, FpuRegister ft);

  void CmpUltS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpLeS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpUleS(GpuRegister rd, FpuRegister fs, FpuRegister ft);

  void CmpUneS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpNeS(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpUnD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpEqD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpUeqD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpLtD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpUltD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpLeD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpUleD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpOrD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpUneD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  void CmpNeD(GpuRegister rd, FpuRegister fs, FpuRegister ft);
  /////////////////////////////// RV64 MACRO Instructions END ///////////////////////////////


  /////////////////////////////// RV64 "V" Instructions  START ///////////////////////////////
  #ifdef RISCV64_VARIANTS_HAS_VECTOR
  // Helper for replicating floating point value in all destination elements.
  void ReplicateFPToVectorRegister(VectorRegister dst, FpuRegister src, bool is_double);

  void AndV(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void OrV(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void NorV(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void XorV(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  void AddvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void AddvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void AddvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void AddvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SubvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SubvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SubvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SubvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Asub_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Asub_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Asub_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Asub_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Asub_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Asub_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Asub_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Asub_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MulvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MulvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MulvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MulvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Div_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Div_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Div_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Div_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Div_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Div_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Div_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Div_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Mod_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Mod_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Mod_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Mod_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Mod_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Mod_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Mod_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Mod_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Add_aB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Add_aH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Add_aW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Add_aD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Ave_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Ave_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Ave_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Ave_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Ave_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Ave_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Ave_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Ave_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Aver_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Aver_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Aver_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Aver_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Aver_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Aver_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Aver_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Aver_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Max_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Max_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Max_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Max_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Max_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Max_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Max_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Max_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Min_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Min_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Min_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Min_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Min_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Min_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Min_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Min_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  void FaddW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FaddD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FsubW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FsubD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FmulW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FmulD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FdivW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FdivD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FmaxW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FmaxD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FminW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FminD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  void Ffint_sW(VectorRegister wd, VectorRegister ws);
  void Ffint_sD(VectorRegister wd, VectorRegister ws);
  void Ftint_sW(VectorRegister wd, VectorRegister ws);
  void Ftint_sD(VectorRegister wd, VectorRegister ws);

  void SllB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SllH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SllW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SllD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SraB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SraH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SraW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SraD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SrlB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SrlH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SrlW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void SrlD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  // Immediate shift instructions, where shamtN denotes shift amount (must be between 0 and 2^N-1).
  void SlliB(VectorRegister wd, VectorRegister ws, int shamt3);
  void SlliH(VectorRegister wd, VectorRegister ws, int shamt4);
  void SlliW(VectorRegister wd, VectorRegister ws, int shamt5);
  void SlliD(VectorRegister wd, VectorRegister ws, int shamt6);
  void SraiB(VectorRegister wd, VectorRegister ws, int shamt3);
  void SraiH(VectorRegister wd, VectorRegister ws, int shamt4);
  void SraiW(VectorRegister wd, VectorRegister ws, int shamt5);
  void SraiD(VectorRegister wd, VectorRegister ws, int shamt6);
  void SrliB(VectorRegister wd, VectorRegister ws, int shamt3);
  void SrliH(VectorRegister wd, VectorRegister ws, int shamt4);
  void SrliW(VectorRegister wd, VectorRegister ws, int shamt5);
  void SrliD(VectorRegister wd, VectorRegister ws, int shamt6);

  void MoveV(VectorRegister wd, VectorRegister ws);
  void SplatiB(VectorRegister wd, VectorRegister ws, int n4);
  void SplatiH(VectorRegister wd, VectorRegister ws, int n3);
  void SplatiW(VectorRegister wd, VectorRegister ws, int n2);
  void SplatiD(VectorRegister wd, VectorRegister ws, int n1);
  void Copy_sB(GpuRegister rd, VectorRegister ws, int n4);
  void Copy_sH(GpuRegister rd, VectorRegister ws, int n3);
  void Copy_sW(GpuRegister rd, VectorRegister ws, int n2);
  void Copy_sD(GpuRegister rd, VectorRegister ws, int n1);
  void Copy_uB(GpuRegister rd, VectorRegister ws, int n4);
  void Copy_uH(GpuRegister rd, VectorRegister ws, int n3);
  void Copy_uW(GpuRegister rd, VectorRegister ws, int n2);
  void InsertB(VectorRegister wd, GpuRegister rs, int n4);
  void InsertH(VectorRegister wd, GpuRegister rs, int n3);
  void InsertW(VectorRegister wd, GpuRegister rs, int n2);
  void InsertD(VectorRegister wd, GpuRegister rs, int n1);
  void FillB(VectorRegister wd, GpuRegister rs);
  void FillH(VectorRegister wd, GpuRegister rs);
  void FillW(VectorRegister wd, GpuRegister rs);
  void FillD(VectorRegister wd, GpuRegister rs);

  void LdiB(VectorRegister wd, int imm8);
  void LdiH(VectorRegister wd, int imm10);
  void LdiW(VectorRegister wd, int imm10);
  void LdiD(VectorRegister wd, int imm10);
  void LdB(VectorRegister wd, GpuRegister rs, int offset);
  void LdH(VectorRegister wd, GpuRegister rs, int offset);
  void LdW(VectorRegister wd, GpuRegister rs, int offset);
  void LdD(VectorRegister wd, GpuRegister rs, int offset);
  void StB(VectorRegister wd, GpuRegister rs, int offset);
  void StH(VectorRegister wd, GpuRegister rs, int offset);
  void StW(VectorRegister wd, GpuRegister rs, int offset);
  void StD(VectorRegister wd, GpuRegister rs, int offset);

  void IlvlB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvlH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvlW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvlD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvrB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvrH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvrW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvrD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvevB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvevH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvevW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvevD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvodB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvodH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvodW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void IlvodD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  void MaddvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MaddvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MaddvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MaddvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MsubvB(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MsubvH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MsubvW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void MsubvD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FmaddW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FmaddD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FmsubW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void FmsubD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  void Hadd_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Hadd_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Hadd_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Hadd_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Hadd_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt);
  void Hadd_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt);

  void PcntB(VectorRegister wd, VectorRegister ws);
  void PcntH(VectorRegister wd, VectorRegister ws);
  void PcntW(VectorRegister wd, VectorRegister ws);
  void PcntD(VectorRegister wd, VectorRegister ws);
  #endif


  /////////////////////////////// RV64 VARIANTS extension ////////////////
  #ifdef RISCV64_VARIANTS_THEAD
  // alu, in spec 16.3.
  void addsl(GpuRegister rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);
  void mula(GpuRegister  rd, GpuRegister rs1, GpuRegister rs2);
  void muls(GpuRegister  rd, GpuRegister rs1, GpuRegister rs2);
  void mveqz(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  void mvnez(GpuRegister rd, GpuRegister rs1, GpuRegister rs2);
  void srri(GpuRegister  rd, GpuRegister rs1, uint8_t uimm6);
  void srriw(GpuRegister  rd, GpuRegister rs1, uint8_t uimm5);

  // bit ops, in spec 16.4.
  void ext(GpuRegister   rd, GpuRegister rs1, uint8_t uimm6_1, uint8_t uimm6_2);
  void extu(GpuRegister  rd, GpuRegister rs1, uint8_t uimm6_1, uint8_t uimm6_2);
  void ff0(GpuRegister   rd, GpuRegister rs1);
  void ff1(GpuRegister   rd, GpuRegister rs1);
  void rev(GpuRegister   rd, GpuRegister rs1);
  void revw(GpuRegister  rd, GpuRegister rs1);
  void tst(GpuRegister   rd, GpuRegister rs1, uint8_t uimm6);
  void tstnbz(GpuRegister rd, GpuRegister rs1);

  // load & store, in spec 16.5.
  void lbia(GpuRegister   rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void lbib(GpuRegister   rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void lbuia(GpuRegister  rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void lbuib(GpuRegister  rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);

  void lwia(GpuRegister   rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void lwib(GpuRegister   rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void lwuia(GpuRegister  rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void lwuib(GpuRegister  rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);

  void sbia(GpuRegister  rs2, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void sbib(GpuRegister  rs2, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void swia(GpuRegister  rs2, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void swib(GpuRegister  rs2, GpuRegister rs1, int8_t imm5, uint8_t uimm2);

  void ldia(GpuRegister   rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void ldib(GpuRegister   rd, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void sdia(GpuRegister  rs2, GpuRegister rs1, int8_t imm5, uint8_t uimm2);
  void sdib(GpuRegister  rs2, GpuRegister rs1, int8_t imm5, uint8_t uimm2);

  void lrb(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);
  void lrbu(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);
  void lrw(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);
  void lrwu(GpuRegister   rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);
  void lrd(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);
  void srb(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);
  void srw(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);
  void srd(GpuRegister    rd, GpuRegister rs1, GpuRegister rs2, uint8_t uimm2);

  void ldd(GpuRegister   rd1, GpuRegister rd2, GpuRegister rs1, uint8_t uimm2);
  void sdd(GpuRegister   rd1, GpuRegister rd2, GpuRegister rs1, uint8_t uimm2);
  #endif
  ///////////////////////////////////////////////////////////////////


  //
  // Heap poisoning.
  //

  // Poison a heap reference contained in `src` and store it in `dst`.
  void PoisonHeapReference(GpuRegister dst, GpuRegister src) {
    // dst = -src.
    // Negate the 32-bit ref.
    Sub(dst, ZERO, src);
    // And constrain it to 32 bits (zero-extend into bits 32 through 63) as on Arm64 and x86/64.
    Extub(dst, dst, 0, 32);
  }
  // Poison a heap reference contained in `reg`.
  void PoisonHeapReference(GpuRegister reg) {
    // reg = -reg.
    PoisonHeapReference(reg, reg);
  }
  // Unpoison a heap reference contained in `reg`.
  void UnpoisonHeapReference(GpuRegister reg) {
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

  void Bind(Label* label) override {
    Bind(down_cast<Riscv64Label*>(label));
  }
  void Jump(Label* label ATTRIBUTE_UNUSED) override {
    UNIMPLEMENTED(FATAL) << "Do not use Jump for RISCV64";
  }

  void Bind(Riscv64Label* label);

  // Create a new literal with a given value.
  // NOTE: Force the template parameter to be explicitly specified.
  template <typename T>
  Literal* NewLiteral(typename Identity<T>::type value) {
    static_assert(std::is_integral<T>::value, "T must be an integral type.");
    return NewLiteral(sizeof(value), reinterpret_cast<const uint8_t*>(&value));
  }

  // Load label address using PC-relative loads. To be used with data labels in the literal /
  // jump table area only and not with regular code labels.
  void LoadLabelAddress(GpuRegister dest_reg, Riscv64Label* label);

  // Create a new literal with the given data.
  Literal* NewLiteral(size_t size, const uint8_t* data);

  // Load literal using PC-relative loads.
  void LoadLiteral(GpuRegister dest_reg, LoadOperandType load_type, Literal* literal);

  // Create a jump table for the given labels that will be emitted when finalizing.
  // When the table is emitted, offsets will be relative to the location of the table.
  // The table location is determined by the location of its label (the label precedes
  // the table data) and should be loaded using LoadLabelAddress().
  JumpTable* CreateJumpTable(std::vector<Riscv64Label*>&& labels);

  void EmitLoad(ManagedRegister m_dst, GpuRegister src_register, int32_t src_offset, size_t size);
  void AdjustBaseAndOffset(GpuRegister& base, int32_t& offset, bool is_doubleword);
  // If element_size_shift is negative at entry, its value will be calculated based on the offset.
  void AdjustBaseOffsetAndElementSizeShift(GpuRegister& base,
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
  void StoreConstToOffset(StoreOperandType type,
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
  void LoadFromOffset(LoadOperandType type,
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
  void LoadFpuFromOffset(LoadOperandType type,
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
  void StoreToOffset(StoreOperandType type,
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
  void StoreFpuToOffset(StoreOperandType type,
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

  void LoadFromOffset(LoadOperandType type, GpuRegister reg, GpuRegister base, int32_t offset);
  void LoadFpuFromOffset(LoadOperandType type, FpuRegister reg, GpuRegister base, int32_t offset);
  void StoreToOffset(StoreOperandType type, GpuRegister reg, GpuRegister base, int32_t offset);
  void StoreFpuToOffset(StoreOperandType type, FpuRegister reg, GpuRegister base, int32_t offset);

  // Emit data (e.g. encoded instruction or immediate) to the instruction stream.
  void Emit(uint32_t value);

  // Emit code that will create an activation on the stack.
  void BuildFrame(size_t frame_size,
                  ManagedRegister method_reg,
                  ArrayRef<const ManagedRegister> callee_save_regs);

  // Emit code that will remove an activation from the stack.
  void RemoveFrame(size_t frame_size,
                   ArrayRef<const ManagedRegister> callee_save_regs,
                   bool may_suspend);

  void IncreaseFrameSize(size_t adjust);
  void DecreaseFrameSize(size_t adjust);

  // Store routines.
  void Store(FrameOffset offs, ManagedRegister msrc, size_t size);
  void StoreRef(FrameOffset dest, ManagedRegister msrc);
  void StoreRawPtr(FrameOffset dest, ManagedRegister msrc);

  void StoreImmediateToFrame(FrameOffset dest, uint32_t imm, ManagedRegister mscratch);

  void StoreStackOffsetToThread(ThreadOffset64 thr_offs,
                                FrameOffset fr_offs,
                                ManagedRegister mscratch);

  void StoreStackPointerToThread(ThreadOffset64 thr_offs);

  void StoreSpanning(FrameOffset dest, ManagedRegister msrc, FrameOffset in_off,
                     ManagedRegister mscratch);

  // Load routines.
  void Load(ManagedRegister mdest, FrameOffset src, size_t size);

  void LoadFromThread(ManagedRegister mdest, ThreadOffset64 src, size_t size);

  void LoadRef(ManagedRegister dest, FrameOffset src);

  void LoadRef(ManagedRegister mdest, ManagedRegister base, MemberOffset offs,
               bool unpoison_reference);

  void LoadRawPtr(ManagedRegister mdest, ManagedRegister base, Offset offs);

  void LoadRawPtrFromThread(ManagedRegister mdest, ThreadOffset64 offs);

  // Copying routines.
  void Move(ManagedRegister mdest, ManagedRegister msrc, size_t size);

  void CopyRawPtrFromThread(FrameOffset fr_offs,
                            ThreadOffset64 thr_offs,
                            ManagedRegister mscratch);

  void CopyRawPtrToThread(ThreadOffset64 thr_offs,
                          FrameOffset fr_offs,
                          ManagedRegister mscratch);

  void CopyRef(FrameOffset dest, FrameOffset src, ManagedRegister mscratch);

  void Copy(FrameOffset dest, FrameOffset src, ManagedRegister mscratch, size_t size);

  void Copy(FrameOffset dest, ManagedRegister src_base, Offset src_offset, ManagedRegister mscratch,
            size_t size);

  void Copy(ManagedRegister dest_base, Offset dest_offset, FrameOffset src,
            ManagedRegister mscratch, size_t size);

  void Copy(FrameOffset dest, FrameOffset src_base, Offset src_offset, ManagedRegister mscratch,
            size_t size);

  void Copy(ManagedRegister dest, Offset dest_offset, ManagedRegister src, Offset src_offset,
            ManagedRegister mscratch, size_t size);

  void Copy(FrameOffset dest, Offset dest_offset, FrameOffset src, Offset src_offset,
            ManagedRegister mscratch, size_t size);

  void MemoryBarrier(ManagedRegister);

  // Sign extension.
  void SignExtend(ManagedRegister mreg, size_t size);

  // Zero extension.
  void ZeroExtend(ManagedRegister mreg, size_t size);

  // Exploit fast access in managed code to Thread::Current().
  void GetCurrentThread(ManagedRegister tr);
  void GetCurrentThread(FrameOffset dest_offset, ManagedRegister mscratch);

  // Set up out_reg to hold a Object** into the handle scope, or to be null if the
  // value is null and null_allowed. in_reg holds a possibly stale reference
  // that can be used to avoid loading the handle scope entry to see if the value is
  // null.
  void CreateHandleScopeEntry(ManagedRegister out_reg, FrameOffset handlescope_offset,
                              ManagedRegister in_reg, bool null_allowed);

  // Set up out_off to hold a Object** into the handle scope, or to be null if the
  // value is null and null_allowed.
  void CreateHandleScopeEntry(FrameOffset out_off, FrameOffset handlescope_offset, ManagedRegister
                              mscratch, bool null_allowed);

  // src holds a handle scope entry (Object**) load this into dst.
  void LoadReferenceFromHandleScope(ManagedRegister dst, ManagedRegister src);

  // Heap::VerifyObject on src. In some cases (such as a reference to this) we
  // know that src may not be null.
  void VerifyObject(ManagedRegister src, bool could_be_null);
  void VerifyObject(FrameOffset src, bool could_be_null);

  // Call to address held at [base+offset].
  void Call(ManagedRegister base, Offset offset, ManagedRegister mscratch);
  void Call(FrameOffset base, Offset offset, ManagedRegister mscratch);
  void CallFromThread(ThreadOffset64 offset, ManagedRegister mscratch);

  // Generate code to check if Thread::Current()->exception_ is non-null
  // and branch to a ExceptionSlowPath if it is.
  void ExceptionPoll(ManagedRegister mscratch, size_t stack_adjust);

  // Emit slow paths queued during assembly and promote short branches to long if needed.
  void FinalizeCode() override;

  // Emit branches and finalize all instructions.
  void FinalizeInstructions(const MemoryRegion& region) override;

  // Returns the (always-)current location of a label (can be used in class CodeGeneratorRISCV64,
  // must be used instead of Riscv64Label::GetPosition()).
  uint32_t GetLabelLocation(const Riscv64Label* label) const;

  // Get the final position of a label after local fixup based on the old position
  // recorded before FinalizeCode().
  uint32_t GetAdjustedPosition(uint32_t old_position);

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
    void Resolve(uint32_t target);

    // Relocate a branch by a given delta if needed due to expansion of this or another
    // branch at a given location by this delta (just changes location_ and target_).
    void Relocate(uint32_t expand_location, uint32_t delta);

    // If the branch is short, changes its type to long.
    void PromoteToLong();

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
    void InitializeType(Type initial_type);
    // Helper for the above.
    void InitShortOrLong(OffsetBits ofs_size, Type short_type, Type long_type);

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

  template<typename Reg1, typename Reg2>
  void EmitI(uint16_t imm, Reg1 rs1, int funct3, Reg2 rd, int opcode) {
    uint32_t encoding = static_cast<uint32_t>(imm) << 20 |
                        static_cast<uint32_t>(rs1) << 15 |
                        static_cast<uint32_t>(funct3) << 12 |
                        static_cast<uint32_t>(rd) << 7 |
                        opcode;
    Emit(encoding);
  }

  template<typename Reg1, typename Reg2, typename Reg3>
  void EmitR(int funct7, Reg1 rs2, Reg2 rs1, int funct3, Reg3 rd, int opcode) {
    uint32_t encoding = static_cast<uint32_t>(funct7) << 25 |
                        static_cast<uint32_t>(rs2) << 20 |
                        static_cast<uint32_t>(rs1) << 15 |
                        static_cast<uint32_t>(funct3) << 12 |
                        static_cast<uint32_t>(rd) << 7 |
                        opcode;
    Emit(encoding);
  }

  template<typename Reg1, typename Reg2, typename Reg3, typename Reg4>
  void EmitR4(Reg1 rs3, int funct2, Reg2 rs2, Reg3 rs1, int funct3, Reg4 rd, int opcode) {
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
  void EmitS(uint16_t imm, Reg1 rs2, Reg2 rs1, int funct3, int opcode) {
    uint32_t encoding = (static_cast<uint32_t>(imm)&0xFE0) << 20 |
                        static_cast<uint32_t>(rs2) << 20 |
                        static_cast<uint32_t>(rs1) << 15 |
                        static_cast<uint32_t>(funct3) << 12 |
                        (static_cast<uint32_t>(imm)&0x1F) << 7 |
                      opcode;
    Emit(encoding);
  }

  void EmitI6(uint16_t funct6, uint16_t imm6, GpuRegister rs1, int funct3, GpuRegister rd, int opcode);
  void EmitB(uint16_t imm, GpuRegister rs2, GpuRegister rs1, int funct3, int opcode);
  void EmitU(uint32_t imm, GpuRegister rd, int opcode);
  void EmitJ(uint32_t imm, GpuRegister rd, int opcode);

  /////////////////////////////// RV64 VARIANTS extension ////////////////
  #ifdef RISCV64_VARIANTS_THEAD
  void EmitRsd(int funct5, int funct2, int funct_rs, GpuRegister rs1, int funct3, GpuRegister rd, int opcode);
  void EmitRsd(int funct5, int funct2, GpuRegister funct_rs, GpuRegister rs1, int funct3, GpuRegister rd, int opcode);
  #endif
  ///////////////////////////////////////////////////////////////////


  void EmitLiterals();
  void EmitBranch(Branch* branch);
  void EmitBranches();
  void EmitJumpTables();
  // Emits exception block.
  void EmitExceptionPoll(Riscv64ExceptionSlowPath* exception);

  void Buncond(Riscv64Label* label, bool is_bare);
  void Bcond(Riscv64Label* label,
             bool is_bare,
             BranchCondition condition,
             GpuRegister lhs,
             GpuRegister rhs = ZERO);
 

  Branch* GetBranch(uint32_t branch_id);
  const Branch* GetBranch(uint32_t branch_id) const;
  void ReserveJumpTableSpace();
  void PromoteBranches();
  void PatchCFI();

  void Call(Riscv64Label* label, bool is_bare);
  void FinalizeLabeledBranch(Riscv64Label* label);


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

  DISALLOW_COPY_AND_ASSIGN(Riscv64Assembler);
};

}  // namespace riscv64
}  // namespace art

#endif  // ART_COMPILER_UTILS_RISCV64_ASSEMBLER_RISCV64_H_
