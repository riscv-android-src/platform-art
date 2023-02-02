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

#ifndef ART_RUNTIME_ARCH_RISCV64_ASM_SUPPORT_RISCV64_S_
#define ART_RUNTIME_ARCH_RISCV64_ASM_SUPPORT_RISCV64_S_

#include "asm_support_riscv64.h"
#include "interpreter/cfi_asm_support.h"

// Define special registers (s0, s1, s11, ra).
// Frame Pointer
#define rFP   s0
// Register holding Thread::Current().
#define rSELF s1
// Register holding suspend check count down.
#define rSUSPEND s11
// Link Register
#define rLR   ra

.macro CFI_EXPRESSION_BREG n, b, offset
    .if (-0x40 <= (\offset)) && ((\offset) < 0x40)
        CFI_EXPRESSION_BREG_1(\n, \b, \offset)
    .elseif (-0x2000 <= (\offset)) && ((\offset) < 0x2000)
        CFI_EXPRESSION_BREG_2(\n, \b, \offset)
    .else
        .error "Unsupported offset"
    .endif
.endm

// Declare a function called name, doesn't set up $gp.
.macro ENTRY_NO_GP_CFA name, cfa_offset
    .type   \name, %function
    .global \name
    .hidden \name
    // ART-compiled functions have OatQuickMethodHeader but assembly funtions do not.
    // Prefix the assembly code with 0xFFs, which means there is no method header.
    .byte 0xFF, 0xFF, 0xFF, 0xFF
    // Cache alignment for function entry.
    // NB: 0xFF because there is a bug in balign where 0x00 creates nop instructions.
    .balign 16, 0xFF
\name:
    .cfi_startproc
     // Should def new CFA?
    .cfi_def_cfa sp, \cfa_offset
.endm


// Declare a function called name, doesn't set up $gp.
.macro ENTRY_NO_GP name
    ENTRY_NO_GP_CFA \name, 0
.endm

// Declare a function called name, sets up $gp.
// This macro modifies t8.
.macro ENTRY name
    ENTRY_NO_GP \name
    // Set up $gp and store the previous $gp value to $t8. It will be pushed to the
    // stack after the frame has been constructed.
    // .cpsetup $t9, $t8, \name
    // Declare a local convenience label to be branched to when $gp is already set up.
.L\name\()_gp_set:
.endm

.macro END name
    .cfi_endproc
    .size \name, .-\name
.endm

.macro UNIMPLEMENTED name
    ENTRY \name
    break
    break
    END \name
.endm

// Macros to poison (negate) the reference for heap poisoning.
.macro POISON_HEAP_REF rRef
#ifdef USE_HEAP_POISONING
    neg \rRef, \rRef  // pseudo instr
#endif  // USE_HEAP_POISONING
.endm

// Macros to unpoison (negate) the reference for heap poisoning.
.macro UNPOISON_HEAP_REF rRef
#ifdef USE_HEAP_POISONING
    neg \rRef, \rRef
#endif  // USE_HEAP_POISONING
.endm

// Byte size of the instructions (un)poisoning heap references.
#ifdef USE_HEAP_POISONING
#define HEAP_POISON_INSTR_SIZE 8
#else
#define HEAP_POISON_INSTR_SIZE 0
#endif  // USE_HEAP_POISONING

.macro INCREASE_FRAME frame_adjustment
    addi sp, sp, -\frame_adjustment
    .cfi_adjust_cfa_offset (\frame_adjustment)
.endm

.macro DECREASE_FRAME frame_adjustment
    addi sp, sp, \frame_adjustment
    .cfi_adjust_cfa_offset -(\frame_adjustment)
.endm

.macro SAVE_REG_BASE base, reg, offset
    sd \reg, (\offset)(\base)
    .cfi_rel_offset \reg, (\offset)
.endm

.macro SAVE_REG reg, offset
    SAVE_REG_BASE sp, \reg, \offset
.endm

.macro RESTORE_REG_BASE base, reg, offset
    ld \reg, (\offset)(\base)
    .cfi_restore \reg
.endm

.macro RESTORE_REG reg, offset
    RESTORE_REG_BASE sp, \reg, \offset
.endm

//***************  T-Head extension start   ***********************
.macro SAVE_TWO_REGS_BASE base, reg1, reg2, offset
    // stp \reg1, \reg2, [\base, #(\offset)]
    .cfi_rel_offset \reg1, (\offset)
    .cfi_rel_offset \reg2, (\offset) + 8
.endm

.macro SAVE_TWO_REGS reg1, reg2, offset
    SAVE_TWO_REGS_BASE sp, \reg1, \reg2, \offset
.endm

.macro RESTORE_TWO_REGS_BASE base, reg1, reg2, offset
    // ldp \reg1, \reg2, [\base, #(\offset)]
    .cfi_restore \reg1
    .cfi_restore \reg2
.endm

.macro RESTORE_TWO_REGS reg1, reg2, offset
    RESTORE_TWO_REGS_BASE sp, \reg1, \reg2, \offset
.endm
//***************  T-Head extension end     ***********************

// s0(rFP), s1(rSELF), s11(rSUSPEND). so,= only s2~s10
// please note order, lower number prefer low addr
.macro SAVE_REG_CALLEE_S2_S10 offset
    SAVE_REG s10,  (\offset +  64)
    SAVE_REG s9,   (\offset +  56)
    SAVE_REG s8,   (\offset +  48)
    SAVE_REG s7,   (\offset +  40)
    SAVE_REG s6,   (\offset +  32)
    SAVE_REG s5,   (\offset +  24)
    SAVE_REG s4,   (\offset +  16)
    SAVE_REG s3,   (\offset +   8)
    SAVE_REG s2,   (\offset +   0)
.endm

.macro RESTORE_REG_CALLEE_S2_S10 offset
    RESTORE_REG  s2,  (\offset +    0)
    RESTORE_REG  s3,  (\offset +    8)
    RESTORE_REG  s4,  (\offset +   16)
    RESTORE_REG  s5,  (\offset +   24)
    RESTORE_REG  s6,  (\offset +   32)
    RESTORE_REG  s7,  (\offset +   40)
    RESTORE_REG  s8,  (\offset +   48)
    RESTORE_REG  s9,  (\offset +   56)
    RESTORE_REG  s10, (\offset +   64)
.endm

// FA registers: f10 ~ f17
.macro SAVE_FREG_ARGS offset
    fsd     f10,     (\offset +  0)(sp)
    fsd     f11,     (\offset +  8)(sp)
    fsd     f12,     (\offset + 16)(sp)
    fsd     f13,     (\offset + 24)(sp)
    fsd     f14,     (\offset + 32)(sp)
    fsd     f15,     (\offset + 40)(sp)
    fsd     f16,     (\offset + 48)(sp)
    fsd     f17,     (\offset + 56)(sp)
.endm

// FT registers: f0 ~ f7, f28 ~ f31
.macro SAVE_FREG_TEMP_T0_T7 offset
    fsd     f0,      (\offset +  0)(sp)
    fsd     f1,      (\offset +  8)(sp)
    fsd     f2,      (\offset + 16)(sp)
    fsd     f3,      (\offset + 24)(sp)
    fsd     f4,      (\offset + 32)(sp)
    fsd     f5,      (\offset + 40)(sp)
    fsd     f6,      (\offset + 48)(sp)
    fsd     f7,      (\offset + 56)(sp)
.endm

.macro SAVE_FREG_TEMP_T8_T11 offset
    fsd     f28,     (\offset +  0)(sp)
    fsd     f29,     (\offset +  8)(sp)
    fsd     f30,     (\offset + 16)(sp)
    fsd     f31,     (\offset + 24)(sp)
.endm

// FS registers: f8-f9 (fs0 - fs1), f18 ~ f27 (fs2 - fs11)
.macro SAVE_FREG_CALLEE_S0_S1 offset
    fsd     f8,      (\offset +  0)(sp)
    fsd     f9,      (\offset +  8)(sp)
.endm

.macro SAVE_FREG_CALLEE_S2_S11 offset
    fsd     f18,     (\offset +  0)(sp)
    fsd     f19,     (\offset +  8)(sp)
    fsd     f20,     (\offset + 16)(sp)
    fsd     f21,     (\offset + 24)(sp)
    fsd     f22,     (\offset + 32)(sp)
    fsd     f23,     (\offset + 40)(sp)
    fsd     f24,     (\offset + 48)(sp)
    fsd     f25,     (\offset + 56)(sp)
    fsd     f26,     (\offset + 64)(sp)
    fsd     f27,     (\offset + 72)(sp)
.endm

.macro SAVE_FREG_ALL offset
    SAVE_FREG_TEMP_T0_T7    (\offset  +   0)   # 0~7
    SAVE_FREG_CALLEE_S0_S1  (\offset  +  64)   # 8~9
    SAVE_FREG_ARGS          (\offset  +  80)   # 10 ~ 17
    SAVE_FREG_CALLEE_S2_S11 (\offset  + 144)   # 18 ~ 27
    SAVE_FREG_TEMP_T8_T11   (\offset  + 224)   # 28 ~ 31
.endm

.macro RESTORE_FREG_ARGS offset
    fld     f10,     (\offset +  0)(sp)
    fld     f11,     (\offset +  8)(sp)
    fld     f12,     (\offset + 16)(sp)
    fld     f13,     (\offset + 24)(sp)
    fld     f14,     (\offset + 32)(sp)
    fld     f15,     (\offset + 40)(sp)
    fld     f16,     (\offset + 48)(sp)
    fld     f17,     (\offset + 56)(sp)
.endm

.macro RESTORE_FREG_TEMPT0_T7 offset
    fld     f0,      (\offset +  0)(sp)
    fld     f1,      (\offset +  8)(sp)
    fld     f2,      (\offset + 16)(sp)
    fld     f3,      (\offset + 24)(sp)
    fld     f4,      (\offset + 32)(sp)
    fld     f5,      (\offset + 40)(sp)
    fld     f6,      (\offset + 48)(sp)
    fld     f7,      (\offset + 56)(sp)
.endm

.macro RESTORE_FREG_TEMP_T8_T11 offset
    fld     f28,     (\offset +  0)(sp)
    fld     f29,     (\offset +  8)(sp)
    fld     f30,     (\offset + 16)(sp)
    fld     f31,     (\offset + 24)(sp)
.endm

.macro RESTORE_FREG_CALLEE_S0_S1 offset
    fld     f8,      (\offset +  0)(sp)
    fld     f9,      (\offset +  8)(sp)
.endm

.macro RESTORE_FREG_CALLEE_S2_S11 offset
    fld     f18,     (\offset +  0)(sp)
    fld     f19,     (\offset +  8)(sp)
    fld     f20,     (\offset + 16)(sp)
    fld     f21,     (\offset + 24)(sp)
    fld     f22,     (\offset + 32)(sp)
    fld     f23,     (\offset + 40)(sp)
    fld     f24,     (\offset + 48)(sp)
    fld     f25,     (\offset + 56)(sp)
    fld     f26,     (\offset + 64)(sp)
    fld     f27,     (\offset + 72)(sp)
.endm

.macro RESTORE_FREG_ALL offset
    RESTORE_FREG_TEMPT0_T7      (\offset  +   0)   # 0~7
    RESTORE_FREG_CALLEE_S0_S1   (\offset  +  64)   # 8~9
    RESTORE_FREG_ARGS           (\offset  +  80)   # 10 ~ 17
    RESTORE_FREG_CALLEE_S2_S11  (\offset  + 144)   # 18 ~ 27
    RESTORE_FREG_TEMP_T8_T11    (\offset  + 224)   # 28 ~ 31
.endm

.macro LOAD_RUNTIME_INSTANCE reg
    la \reg, _ZN3art7Runtime9instance_E
    ld \reg, 0(\reg)
.endm

.macro SAVE_RUNTIME_INSTANCE_METHOD reg, offset
    LOAD_RUNTIME_INSTANCE  \reg
    ld      \reg, RUNTIME_SAVE_ALL_CALLEE_SAVES_METHOD_OFFSET(\reg)
    sd      \reg, (\offset)(sp)         # Place ArtMethod* at bottom of stack.
.endm

    /*
     * Macro that sets up the callee save frame to conform with
     * Runtime::CreateCalleeSaveMethod(kSaveRefsOnly). Restoration assumes
     * non-moving GC.
     * ** Does not include rSUSPEND or rSELF **
     * callee-save: padding + $s2-$s10 + $ra + $s0 = 11 total + 1x8 bytes padding
     */
.macro SETUP_SAVE_REFS_ONLY_FRAME
    INCREASE_FRAME  96

    // Ugly compile-time check, but we only have the preprocessor.
#if (FRAME_SIZE_SAVE_REFS_ONLY != 96)
#error "FRAME_SIZE_SAVE_REFS_ONLY(RISCV64) size not as expected."
#endif

    SAVE_REG ra,   88
    SAVE_REG s0,   80
    SAVE_REG_CALLEE_S2_S10  8

    # load appropriate callee-save-method
    LOAD_RUNTIME_INSTANCE  t1
    ld      t1, RUNTIME_SAVE_REFS_ONLY_METHOD_OFFSET(t1)
    sd      t1, 0(sp)                                   # Place Method* at bottom of stack.

    sd      sp, THREAD_TOP_QUICK_FRAME_OFFSET(rSELF)    # Place sp in Thread::Current()->top_quick_frame.
.endm

.macro RESTORE_SAVE_REFS_ONLY_FRAME
    RESTORE_REG_CALLEE_S2_S10   8
    RESTORE_REG  s0,  80
    RESTORE_REG  ra,  88

    DECREASE_FRAME  96
.endm

.macro SETUP_SAVE_REFS_AND_ARGS_FRAME_INTERNAL
    INCREASE_FRAME  224

    // Ugly compile-time check, but we only have the preprocessor.
#if (FRAME_SIZE_SAVE_REFS_AND_ARGS != 224)
#error "FRAME_SIZE_SAVE_REFS_AND_ARGS(RISCV64) size not as expected."
#endif
    SAVE_REG  ra,  216            # = kQuickCalleeSaveFrame_RefAndArgs_LrOffset
                             
    SAVE_REG  s0,  208          # s0(Riscv64) == s8(mips64) == rFP
    SAVE_REG  s10, 200
    SAVE_REG  s9,  192
    SAVE_REG  s8,  184
    SAVE_REG  s7,  176
    SAVE_REG  s6,  168
    SAVE_REG  s5,  160
    SAVE_REG  s4,  152

    SAVE_REG  s3,  144
    SAVE_REG  s2,  136
    SAVE_REG  a7,  128
    SAVE_REG  a6,  120
    SAVE_REG  a5,  112
    SAVE_REG  a4,  104
    SAVE_REG  a3,  96
    SAVE_REG  a2,  88
    SAVE_REG  a1,  80          # = kQuickCalleeSaveFrame_RefAndArgs_Gpr1Offset

    SAVE_FREG_ARGS  16
    # 2x8 bytes padding + Method*
.endm

    /*
     * Macro that sets up the callee save frame to conform with
     * Runtime::CreateCalleeSaveMethod(kSaveRefsAndArgs). Restoration assumes
     * non-moving GC.
     * callee-save: padding + $f10-$f17 + $a1-$a7 + $s2-$s10 + $ra + $s0 = 26 total + 1 words padding + Method*
     */
.macro SETUP_SAVE_REFS_AND_ARGS_FRAME
    SETUP_SAVE_REFS_AND_ARGS_FRAME_INTERNAL

    # load appropriate callee-save-method
    LOAD_RUNTIME_INSTANCE  t1
    ld      t1, RUNTIME_SAVE_REFS_AND_ARGS_METHOD_OFFSET(t1)
    sd      t1, 0(sp)                                 # Place Method* at bottom of stack.
    sd      sp, THREAD_TOP_QUICK_FRAME_OFFSET(rSELF)  # Place sp in Thread::Current()->top_quick_frame.
.endm

.macro SETUP_SAVE_REFS_AND_ARGS_FRAME_WITH_METHOD_IN_A0
    SETUP_SAVE_REFS_AND_ARGS_FRAME_INTERNAL

    sd      a0, 0(sp)                                 # Place Method* at bottom of stack.
    sd      sp, THREAD_TOP_QUICK_FRAME_OFFSET(rSELF)  # Place sp in Thread::Current()->top_quick_frame.
.endm

.macro RESTORE_SAVE_REFS_AND_ARGS_FRAME
    RESTORE_REG  ra,   216

    RESTORE_REG  s0,   208      # s0(Riscv64) == s8(mips64) == fp
    RESTORE_REG  s10,  200
    RESTORE_REG  s9,   192
    RESTORE_REG  s8,   184
    RESTORE_REG  s7,   176
    RESTORE_REG  s6,   168
    RESTORE_REG  s5,   160
    RESTORE_REG  s4,   152

    RESTORE_REG  s3,   144
    RESTORE_REG  s2,   136
    RESTORE_REG  a7,   128
    RESTORE_REG  a6,   120
    RESTORE_REG  a5,   112
    RESTORE_REG  a4,   104
    RESTORE_REG  a3,   96
    RESTORE_REG  a2,   88
    RESTORE_REG  a1,   80

    RESTORE_FREG_ARGS   16

    DECREASE_FRAME  224
.endm


    /*
     * Macro that sets up the callee save frame to conform with
     * Runtime::CreateCalleeSaveMethod(kSaveAllCalleeSaves)
     * callee-save: padding + $f8-$f9 + $f18-$f27 + $s1-$s11 + $ra + $s0 = 25 total + 1x8 bytes padding
     */
.macro SETUP_SAVE_ALL_CALLEE_SAVES_FRAME
    INCREASE_FRAME 208

     // Ugly compile-time check, but we only have the preprocessor.
#if (FRAME_SIZE_SAVE_ALL_CALLEE_SAVES != 208)
#error "FRAME_SIZE_SAVE_ALL_CALLEE_SAVES(RISCV64) size not as expected."
#endif

    SAVE_REG ra,   200
    SAVE_REG s0,   192
    SAVE_REG s11,  184
    SAVE_REG_CALLEE_S2_S10   112
    SAVE_REG s1,   104

    // FP callee-saves:  f8, f9; f18 ~ f27
    SAVE_FREG_CALLEE_S0_S1  8
    SAVE_FREG_CALLEE_S2_S11  24

    # load appropriate callee-save-method
    SAVE_RUNTIME_INSTANCE_METHOD  t1, 0

    sd sp, THREAD_TOP_QUICK_FRAME_OFFSET(rSELF)  # Place sp in Thread::Current()->top_quick_frame.
.endm


    /*
     * Macro that sets up the callee save frame to conform with
     * Runtime::CreateCalleeSaveMethod(kSaveEverything).
     * when the $sp has already been decremented by FRAME_SIZE_SAVE_EVERYTHING.
     * callee-save: $a0-$a7 + $t0-$t6 + $s1-$s11 + $fp(s0) + $ra,
     *              $f0-$f31; 28(GPR)+ 32(FPR) + 1x8 bytes padding + method*
     * This macro sets up $gp; entrypoints using it should start with ENTRY_NO_GP.
     */
.macro SETUP_SAVE_EVERYTHING_FRAME_DECREMENTED_SP runtime_method_offset = RUNTIME_SAVE_EVERYTHING_METHOD_OFFSET
     // Ugly compile-time check, but we only have the preprocessor.
#if (FRAME_SIZE_SAVE_EVERYTHING != 496)
#error "FRAME_SIZE_SAVE_EVERYTHING(RISCV64) size not as expected."
#endif

    // Save core registers.
    # save ra, fp
    SAVE_REG ra,   488
    SAVE_REG s0,   480

    # save t3 - t6
    SAVE_REG t6,   472
    SAVE_REG t5,   464
    SAVE_REG t4,   456
    SAVE_REG t3,   448

    # save s2 - s11
    SAVE_REG s11,  440
    SAVE_REG_CALLEE_S2_S10   368

    # save a0 - a7
    SAVE_REG a7,   360
    SAVE_REG a6,   352
    SAVE_REG a5,   344
    SAVE_REG a4,   336
    SAVE_REG a3,   328
    SAVE_REG a2,   320
    SAVE_REG a1,   312
    SAVE_REG a0,   304

    # save s1
    SAVE_REG s1,   296

    # save t0 - t2
    SAVE_REG t2,   288
    SAVE_REG t1,   280
    SAVE_REG t0,   272

    // Save FP registers.
    SAVE_FREG_ALL   16

    # load appropriate callee-save-method
    LOAD_RUNTIME_INSTANCE  t1
    ld      t1, \runtime_method_offset(t1)
    sd      t1, 0(sp)                                # Place ArtMethod* at bottom of stack.

    sd      sp, THREAD_TOP_QUICK_FRAME_OFFSET(rSELF)  # Place sp in Thread::Current()->top_quick_frame.
.endm

    /*
     * Macro that sets up the callee save frame to conform with
     * Runtime::CreateCalleeSaveMethod(kSaveEverything).
     * callee-save: $at + $v0-$v1 + $a0-$a7 + $t0-$t3 + $s0-$s7 + $t5-$t6 + $gp + $s8 + $ra + $s8,
     *              $f0-$f31; 28(GPR)+ 32(FPR) + 1x8 bytes padding + method*
     * This macro sets up $gp; entrypoints using it should start with ENTRY_NO_GP.
     */
.macro SETUP_SAVE_EVERYTHING_FRAME runtime_method_offset = RUNTIME_SAVE_EVERYTHING_METHOD_OFFSET
    addi sp, sp, -(FRAME_SIZE_SAVE_EVERYTHING)
    .cfi_adjust_cfa_offset (FRAME_SIZE_SAVE_EVERYTHING)
    SETUP_SAVE_EVERYTHING_FRAME_DECREMENTED_SP \runtime_method_offset
.endm

.macro RESTORE_SAVE_EVERYTHING_FRAME restore_a0=1
    // Restore FP registers.
    RESTORE_FREG_ALL  16

    // Restore core registers.
    # restore ra, fp
    RESTORE_REG  ra,  488
    RESTORE_REG  s0,  480

    # restore t0 - t6
    RESTORE_REG  t6,  472
    RESTORE_REG  t5,  464
    RESTORE_REG  t4,  456
    RESTORE_REG  t3,  448

    // restore s1 - s11
    RESTORE_REG  s11, 440
    RESTORE_REG_CALLEE_S2_S10   368

    // restore a0 - a7
    RESTORE_REG  a7,  360
    RESTORE_REG  a6,  352
    RESTORE_REG  a5,  344
    RESTORE_REG  a4,  336
    RESTORE_REG  a3,  328
    RESTORE_REG  a2,  320
    RESTORE_REG  a1,  312
    .if \restore_a0
    RESTORE_REG  a0,  304
    .endif

    RESTORE_REG  s1,  296
    RESTORE_REG  t2,  288
    RESTORE_REG  t1,  280
    RESTORE_REG  t0,  272

    DECREASE_FRAME  496
.endm

// This assumes the top part of these stack frame types are identical.
#define REFS_AND_ARGS_MINUS_REFS_SIZE (FRAME_SIZE_SAVE_REFS_AND_ARGS - FRAME_SIZE_SAVE_REFS_ONLY)

    /*
     * Macro that calls through to artDeliverPendingExceptionFromCode, where the pending
     * exception is Thread::Current()->exception_ when the runtime method frame is ready.
     * Requires $gp properly set up.
     */
.macro DELIVER_PENDING_EXCEPTION_FRAME_READY
    la      t6, artDeliverPendingExceptionFromCode
    mv      a0, rSELF                   # pass Thread::Current
    jalr    zero, 0(t6)                 # artDeliverPendingExceptionFromCode(Thread*)
    ebreak                              # Unreached
.endm

    /*
     * Macro that calls through to artDeliverPendingExceptionFromCode, where the pending
     * exception is Thread::Current()->exception_.
     */
.macro DELIVER_PENDING_EXCEPTION
    SETUP_SAVE_ALL_CALLEE_SAVES_FRAME
    DELIVER_PENDING_EXCEPTION_FRAME_READY
.endm


.macro RETURN_OR_DELIVER_PENDING_EXCEPTION_REG reg
    ld   \reg, THREAD_EXCEPTION_OFFSET(rSELF) # load Thread::Current()->exception_
    bne  \reg, zero, 1f
    jalr    zero, ra
1:
    DELIVER_PENDING_EXCEPTION
.endm


.macro RETURN_OR_DELIVER_PENDING_EXCEPTION
    RETURN_OR_DELIVER_PENDING_EXCEPTION_REG t0
.endm


#endif  // ART_RUNTIME_ARCH_RISCV64_ASM_SUPPORT_RISCV64_S_
