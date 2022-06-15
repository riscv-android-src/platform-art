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

#include "asm_support_riscv64.S"
#include "interpreter/cfi_asm_support.h"

#include "arch/quick_alloc_entrypoints.S"

    //.set noreorder
    .balign 16

    /* Deliver the given exception */
    .extern artDeliverExceptionFromCode
    /* Deliver an exception pending on a thread */
    .extern artDeliverPendingExceptionFromCode

    /*
     * Macro that calls through to artDeliverPendingExceptionFromCode, where the pending
     * exception is Thread::Current()->exception_.
     */
.macro DELIVER_PENDING_EXCEPTION
    SETUP_GP
    SETUP_SAVE_ALL_CALLEE_SAVES_FRAME    # save callee saves for throw
    DELIVER_PENDING_EXCEPTION_FRAME_READY
.endm

.macro RETURN_IF_NO_EXCEPTION
    ld     t0, THREAD_EXCEPTION_OFFSET(rSELF) # load Thread::Current()->exception_
    RESTORE_SAVE_REFS_ONLY_FRAME
    bne    t0, zero, 1f                      # success if no exception is pending
    jalr   zero, 0(ra)
1:
    DELIVER_PENDING_EXCEPTION
.endm

.macro RETURN_IF_ZERO
    RESTORE_SAVE_REFS_ONLY_FRAME
    bne    a0, zero, 1f                   # success?
    jalr   zero, 0(ra)                    # return on success
1:
    DELIVER_PENDING_EXCEPTION
.endm

.macro RETURN_IF_RESULT_IS_NON_ZERO_OR_DELIVER
    RESTORE_SAVE_REFS_ONLY_FRAME
    beq    a0, zero, 1f                   # success?
    jalr   zero, 0(ra)                    # return on success
1:
    DELIVER_PENDING_EXCEPTION
.endm

    /*
     * On stack replacement stub.
     * On entry:
     *   a0 = stack to copy
     *   a1 = size of stack
     *   a2 = pc to call
     *   a3 = JValue* result
     *   a4 = shorty
     *   a5 = thread
     */
ENTRY art_quick_osr_stub
    mv     t0, sp               # save stack pointer; pseudo instr

    // Wendong: TBD, remove this, do NOT need this
    addi   t1, sp, -224         # reserve stack space
    srli   t1, t1, 4            # enforce 16 byte stack alignment
    slli   sp, t1, 4            # update stack pointer
    .cfi_adjust_cfa_offset 224

    // Save callee floating point registers. fs0 -- fs11
    SAVE_FREG_CALLEE2   144
    SAVE_FREG_CALLEE1   128

    // Save callee general purpose registers, SP, RA, A3, and A4 (8x14 bytes)
    sd     ra, 120(sp)
    .cfi_rel_offset ra, 120
    sd     s0, 112(sp)
    .cfi_rel_offset s0, 112
    sd     t0, 104(sp)           # save original stack pointer stored in t0
    .cfi_rel_offset t0, 104
    sd     s11, 96(sp)
    .cfi_rel_offset s11, 96

    SAVE_REG_CALLEE_S2_S10   24

    sd     s1, 16(sp)
    .cfi_rel_offset s1, 16

    sd     a4, 8(sp)
    .cfi_rel_offset a4, 8
    sd     a3, 0(sp)
    .cfi_rel_offset a3, 0
    mv     rSELF, a5                      # Save managed thread pointer into rSELF

    addi sp, sp, -16
    .cfi_adjust_cfa_offset 16
    sd     zero, 0(sp)                   # Store null for ArtMethod* at bottom of frame
    jal    .Losr_entry
    .cfi_remember_state

    addi sp, sp, 16
    .cfi_adjust_cfa_offset -16

    // Restore callee floating point registers. fs0 -- fs11
    RESTORE_FREG_CALLEE2  144
    RESTORE_FREG_CALLEE1  128

    // Restore callee registers
    ld     ra, 120(sp)
    .cfi_restore ra
    ld     s0, 112(sp)
    .cfi_restore s0
    ld     t0, 104(sp)           # save original stack pointer stored in t0
    .cfi_restore t0
    ld     s11, 96(sp)
    .cfi_restore s11

    RESTORE_REG_CALLEE_S2_S10   24

    ld     s1, 16(sp)
    .cfi_restore s1
    // Restore return value address and shorty address
    ld     a4, 8(sp)                     # shorty address
    .cfi_restore a4
    ld     a3, 0(sp)                     # result value address
    .cfi_restore a3

    mv     sp, t0                       # restore original SP
    .cfi_adjust_cfa_offset -224

    lbu    t1, 0(a4)                     # load return type
    li     t2, 'D'                        # put char 'D' into t2
    beq    t1, t2, .Losr_d_result       # branch if result type char == 'D'
    li     t2, 'F'                        # put char 'F' into t2
    beq    t1, t2, .Losr_f_result       # branch if result type char == 'F'
    sd     a0, 0(a3)                    #  Non-FP result
    jalr   zero, 0(ra)
.Losr_f_result:
    fmv.x.w  a0, f10                    # put Float result in a0
    sw     a0, 0(a3)
    li     a0, -1                       # NaN-Boxing Float result
    sw     a0, 4(a3)
    jalr   zero, 0(ra)
.Losr_d_result:
    fmv.x.d  a0, f10                    # put Double result in a0
    sd     a0, 0(a3)                    #  Non-FP result
    jalr   zero, 0(ra)

.Losr_entry:
    .cfi_restore_state                  // Reset unwind info so following code unwinds.
    mv     t3, sp
    .cfi_def_cfa_register t3

    sub    sp, sp, a1                   # Reserve space for callee stack
    addi   a1, a1, -8
    add    t0, a1, sp
    sd     ra, 0(t0)                     # Store RA

    // Copy arguments into callee stack
    // Use simple copy routine for now.
    // 4 bytes per slot.
    // a0 = source address
    // a1 = args length in bytes (does not include 8 bytes for RA)
    // sp = destination address
    beqz   a1, .Losr_loop_exit
    addi   a1, a1, -4
    add    t1, a0, a1
    add    t2, sp, a1
.Losr_loop_entry:
    lw     t0, 0(t1)
    addi   t1, t1, -4
    sw     t0, 0(t2)
    addi   t2, t2, -4
    bge    t2, sp, .Losr_loop_entry

    // Pass rSELF as well
    sd     rSELF, -104(t3)

.Losr_loop_exit:
    mv     t6, a2
    jalr   zero, 0(t6)                        # Jump to the OSR entry point.
END art_quick_osr_stub

    /*
     * On entry $a0 is uint32_t* gprs_ and $a1 is uint32_t* fprs_
     * FIXME: just guessing about the shape of the jmpbuf.  Where will pc be?
     */
ENTRY_NO_GP art_quick_do_long_jump
    mv      sp,   a1       # fprs
    RESTORE_FREG_ALL   0

    # skip zero
    ld      ra, 8(a0)
    ld      sp, 16(a0)
    # skip gp and tp
    
    # load t0 - t2
    ld      t0, 40(a0)
    ld      t1, 48(a0)
    ld      t2, 56(a0)

    # load s0/s1
    ld      s0, 64(a0)
    ld      s1, 72(a0)

    # a0 has to be loaded last
    # load a1 - a7
    ld      a1, 88(a0)
    ld      a2, 96(a0)
    ld      a3, 104(a0)
    ld      a4, 112(a0)
    ld      a5, 120(a0)
    ld      a6, 128(a0)
    ld      a7, 136(a0)

    # load s2 - s11
    ld      s2, 144(a0)
    ld      s3, 152(a0)
    ld      s4, 160(a0)
    ld      s5, 168(a0)
    ld      s6, 176(a0)
    ld      s7, 184(a0)
    ld      s8, 192(a0)
    ld      s9, 200(a0)
    ld      s10, 208(a0)
    
    ld      s11, 216(a0)

    # load t3 - t6
    ld      t3, 224(a0)
    ld      t4, 232(a0)
    ld      t5, 240(a0)
    ld      t6, 248(a0)

    # load a0 now.
    ld      a0, 80(a0)

    jalr    zero, 0(t6)       # do long jump (do not use ra, it must not be clobbered)
END art_quick_do_long_jump

    /*
     * Called by managed code, saves most registers (forms basis of long jump
     * context) and passes the bottom of the stack.
     * artDeliverExceptionFromCode will place the callee save Method* at
     * the bottom of the thread. On entry a0 holds Throwable*
     */
ENTRY art_quick_deliver_exception
    SETUP_SAVE_ALL_CALLEE_SAVES_FRAME
    la   t6, artDeliverExceptionFromCode  #@dla  $t6, artDeliverExceptionFromCode
    mv   a1, rSELF                 # pass Thread::Current
    jalr ra, 0(t6)                 # artDeliverExceptionFromCode(Throwable*, Thread*)
    ebreak;
END art_quick_deliver_exception

    .hidden art_quick_throw_null_pointer_exception
    .extern artThrowNullPointerExceptionFromCode
    /*
     * Called by managed code to create and deliver a NullPointerException
     */
ENTRY_NO_GP art_quick_throw_null_pointer_exception
    // Note that setting up $gp does not rely on $t6 here, so branching here directly is OK,
    // even after clobbering any registers we don't need to preserve, such as $gp or $t0.
    SETUP_SAVE_EVERYTHING_FRAME
    la   t6, artThrowNullPointerExceptionFromCode
    mv   a0, rSELF                   # pass Thread::Current
    jalr zero, 0(t6)                 # artThrowNullPointerExceptionFromCode(Thread*)
END art_quick_throw_null_pointer_exception

    /*
     * Call installed by a signal handler to create and deliver a NullPointerException
     */
    .extern artThrowNullPointerExceptionFromSignal
ENTRY_NO_GP_CFA art_quick_throw_null_pointer_exception_from_signal, FRAME_SIZE_SAVE_EVERYTHING
    SETUP_SAVE_EVERYTHING_FRAME_DECREMENTED_SP
    # Retrieve the fault address from the padding where the signal handler stores it.
    ld   a0, (__SIZEOF_POINTER__)(sp)
    la   t6, artThrowNullPointerExceptionFromSignal
    mv   a1, rSELF                   # pass Thread::Current
    jalr zero, 0(t6)                 # artThrowNullPointerExceptionFromSignal(uinptr_t, Thread*)
END art_quick_throw_null_pointer_exception_from_signal

    /*
     * Called by managed code to create and deliver an ArithmeticException
     */
    .extern artThrowDivZeroFromCode
ENTRY_NO_GP art_quick_throw_div_zero
    SETUP_SAVE_EVERYTHING_FRAME
    la   t6, artThrowDivZeroFromCode
    mv   a0, rSELF                   # pass Thread::Current
    jalr zero, 0(t6)                 # artThrowDivZeroFromCode(Thread*)
END art_quick_throw_div_zero

    /*
     * Called by managed code to create and deliver an
     * ArrayIndexOutOfBoundsException
     */
    .extern artThrowArrayBoundsFromCode
ENTRY_NO_GP art_quick_throw_array_bounds
    // Note that setting up $gp does not rely on $t6 here, so branching here directly is OK,
    // even after clobbering any registers we don't need to preserve, such as $gp or $t0.
    SETUP_SAVE_EVERYTHING_FRAME
    la   t6, artThrowArrayBoundsFromCode
    mv   a2, rSELF                   # pass Thread::Current
    jalr zero, 0(t6)                 # artThrowArrayBoundsFromCode(index, limit, Thread*)
END art_quick_throw_array_bounds

    /*
     * Called by managed code to create and deliver a StringIndexOutOfBoundsException
     * as if thrown from a call to String.charAt().
     */
    .extern artThrowStringBoundsFromCode
ENTRY_NO_GP art_quick_throw_string_bounds
    SETUP_SAVE_EVERYTHING_FRAME
    la   t6, artThrowStringBoundsFromCode
    mv   a2, rSELF                   # pass Thread::Current
    jalr zero, 0(t6)                 # artThrowStringBoundsFromCode(index, limit, Thread*)
END art_quick_throw_string_bounds

    /*
     * Called by managed code to create and deliver a StackOverflowError.
     */
    .extern artThrowStackOverflowFromCode
ENTRY art_quick_throw_stack_overflow
    SETUP_SAVE_ALL_CALLEE_SAVES_FRAME
    la   t6, artThrowStackOverflowFromCode
    mv   a0, rSELF                 # pass Thread::Current
    jalr zero, 0(t6)                 # artThrowStackOverflowFromCode(Thread*)
END art_quick_throw_stack_overflow

    /*
     * All generated callsites for interface invokes and invocation slow paths will load arguments
     * as usual - except instead of loading arg0/$a0 with the target Method*, arg0/$a0 will contain
     * the method_idx.  This wrapper will save arg1-arg3, load the caller's Method*, align the
     * stack and call the appropriate C helper.
     * NOTE: "this" is first visable argument of the target, and so can be found in arg1/$a1.
     *
     * The helper will attempt to locate the target and return a 128-bit result in $v0/$v1 consisting
     * of the target Method* in $v0 and method->code_ in $v1.
     *
     * If unsuccessful, the helper will return null/null. There will be a pending exception in the
     * thread and we branch to another stub to deliver it.
     *
     * On success this wrapper will restore arguments and *jump* to the target, leaving the ra
     * pointing back to the original caller.
     */
.macro INVOKE_TRAMPOLINE_BODY cxx_name, save_s4_thru_s8_only=0
    .extern \cxx_name
    SETUP_SAVE_REFS_AND_ARGS_FRAME \save_s4_thru_s8_only  # save callee saves in case
                                                          # allocation triggers GC
    mv    a2, rSELF                        # pass Thread::Current
    mv    a3, sp                           # pass $sp
    jal   \cxx_name                        # (method_idx, this, Thread*, $sp)
    mv    a0, a0                           # save target Method*
    mv    t6, a1                           # save $v0->code_
    RESTORE_SAVE_REFS_AND_ARGS_FRAME
    beq   a0, zero, 1f
    jalr  zero, 0(t6)
1:
    DELIVER_PENDING_EXCEPTION
.endm
.macro INVOKE_TRAMPOLINE c_name, cxx_name
ENTRY \c_name
    INVOKE_TRAMPOLINE_BODY \cxx_name
END \c_name
.endm

INVOKE_TRAMPOLINE art_quick_invoke_interface_trampoline_with_access_check, artInvokeInterfaceTrampolineWithAccessCheck

INVOKE_TRAMPOLINE art_quick_invoke_static_trampoline_with_access_check, artInvokeStaticTrampolineWithAccessCheck
INVOKE_TRAMPOLINE art_quick_invoke_direct_trampoline_with_access_check, artInvokeDirectTrampolineWithAccessCheck
INVOKE_TRAMPOLINE art_quick_invoke_super_trampoline_with_access_check, artInvokeSuperTrampolineWithAccessCheck
INVOKE_TRAMPOLINE art_quick_invoke_virtual_trampoline_with_access_check, artInvokeVirtualTrampolineWithAccessCheck

    # On entry:
    #   t0 = shorty[1] (skip 1 for return type)
    #   t1 = ptr to arg_array
    #   t5 = float/double arg count
    # This macro modifies t3, t6, t5 and v0
.macro LOOP_OVER_SHORTY_LOADING_INTEGER_REG gpu label_in lable_out
\label_in:
    lbu    t3, 0(t0)           # get argument type from shorty
    c.addi   t0, 1
    beqz   t3, \lable_out          # jump out

    li     t6, 68               # put char 'D' into t6
    beq    t6, t3, 1f          # branch if result type char == 'D'
    li     t6, 70               # put char 'F' into t6
    beq    t6, t3, 2f          # branch if result type char == 'F'
    li     t6, 74               # put char 'J' into t6
    beq    t6, t3, 3f          # branch if result type char == 'J'
    # found int (4 bytes)
    lw     \gpu, 0(t1)
    c.addi   t1, 4
    c.j      4f

1:  # found double and skip it if the count < 8
    c.addi t5, 1
    li t6, 8
    ble t5, t6, 11f
    ld \gpu, 0(t1)
    c.addi   t1, 8
    c.j      4f  
11:
    c.addi t1, 8
    c.j      \label_in

2:  # found float and and skip it if the count < 8
    c.addi t5, 1
    li t6, 8
    ble t5, t6, 22f
    lw \gpu, 0(t1)
    c.addi   t1, 4
    c.j      4f    
22:
    c.addi   t1, 4
    c.j      \label_in

3:  # found long (8 bytes)
    lwu    t3, 0(t1)
    lwu    t6, 4(t1)
    slli   t6, t6, 32
    or     \gpu, t6, t3
    c.addi t1, 8
4:
.endm

    # On entry:
    #   t0 = shorty[1] (skip 1 for return type)
    #   t1 = ptr to arg_array
    # This macro modifies t3, t6 and v0
.macro LOOP_OVER_SHORTY_LOADING_FLOAT_REG fpu label_in lable_out
\label_in:
    lbu    t3, 0(t0)           # get argument type from shorty
    c.addi   t0, 1
    beqz   t3, \lable_out      # jump out

    li     t6, 68               # put char 'D' into t6
    beq    t6, t3, 1f          # branch if result type char == 'D'
    li     t6, 70               # put char 'F' into t6
    beq    t6, t3, 2f          # branch if result type char == 'F'
    li     t6, 74               # put char 'J' into t6
    beq    t6, t3, 3f          # branch if result type char == 'J'
    # found int (4 bytes) skip it
    c.addi   t1, 4
    c.j      \label_in

1:  # found double (8 bytes)
    fld    \fpu, 0(t1)              # load double arg into \fpu
    c.addi t1, 8
    c.j      4f

2:  # found float (4 bytes)
    flw    \fpu, 0(t1)              # load float arg into \fpu
    c.addi   t1, 4
    c.j      4f

3:  # found long (8 bytes) skip it
    c.addi t1, 8
    c.j      \label_in
4:
.endm

    /*
     * Invocation stub for quick code.
     * On entry:
     *   a0 = method pointer
     *   a1 = argument array that must at least contain the this ptr.
     *   a2 = size of argument array in bytes
     *   a3 = (managed) thread pointer
     *   a4 = JValue* result
     *   a5 = shorty
     */
ENTRY_NO_GP art_quick_invoke_stub
    # push a4, a5, s11(rSUSPEND), s1(rSELF), s0(fp), ra onto the stack
    addi sp, sp, -48
    .cfi_adjust_cfa_offset 48
    sd     ra, 40(sp)
    .cfi_rel_offset ra, 40
    sd     s0, 32(sp)
    .cfi_rel_offset s0, 32
    sd     rSELF, 24(sp)
    .cfi_rel_offset rSELF, 24
    sd     rSUSPEND, 16(sp)
    .cfi_rel_offset rSUSPEND, 16
    sd     a5, 8(sp)
    .cfi_rel_offset a5, 8
    sd     a4, 0(sp)
    .cfi_rel_offset a4, 0

    mv     rSELF, a3           # move managed thread pointer into s1 (rSELF)
    mv     s0, sp              # save sp in s0 (fp)
    .cfi_def_cfa_register s0

    addi   t3, a2, 24          # add 8 for ArtMethod* and 16 for stack alignment
    srli   t3, t3, 4           # shift the frame size right 4
    slli   t3, t3, 4           # shift the frame size left 4 to align to 16 bytes
    sub    sp, sp, t3          # reserve stack space for argument array

    addi   t0, a5, 1           # t0 = shorty[1] (skip 1 for return type)
    addi   t1, a1, 4           # t1 = ptr to arg_array[4] (skip this ptr)
    addi   t2, a2, -4          # t2 = number of argument bytes remain (skip this ptr)
    addi   t4, sp, 12          # v0 (t4 in riscv64) points to where to copy arg_array

    # Copy all args into stack
1:
    beqz   t2, 2f
    addi   t2, t2, -4
    lw     t3, 0(t1)           # load from argument array
    addi   t1, t1, 4
    sw     t3, 0(t4)           # save to stack
    addi   t4, t4, 4
    j 1b
2:
    sd     zero, 0(sp)

    # Load float/double args into Floating Argument Registers
    addi   t0, a5, 1           # t0 = shorty[1] (skip 1 for return type)
    addi   t1, a1, 4           # t1 = ptr to arg_array[4] (skip this ptr)

    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f10 f_a0 load_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f11 f_a1 load_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f12 f_a2 load_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f13 f_a3 load_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f14 f_a4 load_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f15 f_a5 load_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f16 f_a6 load_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f17 f_a7 load_integer

    # Load int/long args into Integer Argument Registers
load_integer:
    addi   t0, a5, 1           # t0 = shorty[1] (skip 1 for return type)
    addi   t1, a1, 4           # t1 = ptr to arg_array[4] (skip this ptr)
    li   t5, 0               # t5 = float/double arg count
    
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a2 i_a2 call_fn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a3 i_a3 call_fn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a4 i_a4 call_fn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a5 i_a5 call_fn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a6 i_a6 call_fn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a7 i_a7 call_fn

call_fn:
    # call method (a0 and a1 have been untouched)
    lwu    a1, 0(a1)           # make a1 = this ptr
    sw     a1, 8(sp)           # copy this ptr (skip 8 bytes for ArtMethod*)
    sd     zero, 0(sp)         # store null for ArtMethod* at bottom of frame
    ld     t6, ART_METHOD_QUICK_CODE_OFFSET_64(a0)  # get pointer to the code
    jalr   t6                  # call the method
    mv     sp, s0              # restore sp from fp(s0)
    .cfi_def_cfa_register sp

    # pop a4, a5, s9(rSUSPEND), s1(rSELF), s0(fp), ra off of the stack
    ld     a4, 0(sp)
    .cfi_restore a4
    ld     a5, 8(sp)
    .cfi_restore a5
    ld     rSUSPEND, 16(sp)
    .cfi_restore rSUSPEND
    ld     rSELF, 24(sp)
    .cfi_restore rSELF
    ld     s0, 32(sp)
    .cfi_restore s0
    ld     ra, 40(sp)
    .cfi_restore ra
    addi   sp, sp, 48
    .cfi_adjust_cfa_offset -48

    # a4 = JValue* result
    # a5 = shorty string

    lbu   t1, 0(a5)
    li    t2, 'V'
    beq   t1, t2, 3f
    li    t2, 'D'
    beq   t1, t2, 1f
    li    t2, 'F'
    beq   t1, t2, 2f
    sd    a0, 0(a4)
    ret
1:
    fsd   f10, 0(a4)    # Double result
    ret
2:
    fsw   f10, 0(a4)    # Float result
    li    a0,  -1       # NaN-Boxing float result
    sw    a0,  4(a4)
    ret
3:
    ret
END art_quick_invoke_stub

    /*
     * Invocation static stub for quick code.
     * On entry:
     *   a0 = method pointer
     *   a1 = argument array that must at least contain the this ptr.
     *   a2 = size of argument array in bytes
     *   a3 = (managed) thread pointer
     *   a4 = JValue* result
     *   a5 = shorty
     */
ENTRY_NO_GP art_quick_invoke_static_stub
    # push a4, a5, s11(rSUSPEND), s1(rSELF), s0(fp), ra, onto the stack
    addi sp, sp, -48
    .cfi_adjust_cfa_offset 48
    sd     ra, 40(sp)
    .cfi_rel_offset ra, 40
    sd     s0, 32(sp)
    .cfi_rel_offset s0, 32
    sd     rSELF, 24(sp)
    .cfi_rel_offset rSELF, 24
    sd     rSUSPEND, 16(sp)
    .cfi_rel_offset rSUSPEND, 16
    sd     a5, 8(sp)
    .cfi_rel_offset a5, 8
    sd     a4, 0(sp)
    .cfi_rel_offset a4, 0

    mv     rSELF, a3           # move managed thread pointer into s1 (rSELF)
    mv     s0, sp              # save sp in s0 (fp)
    .cfi_def_cfa_register s0

    addi   t3, a2, 24          # add 8 for ArtMethod* and 16 for stack alignment
    srli   t3, t3, 4           # shift the frame size right 4
    slli   t3, t3, 4           # shift the frame size left 4 to align to 16 bytes
    sub    sp, sp, t3         # reserve stack space for argument array

    addi   t0, a5, 1           # t0 = shorty[1] (skip 1 for return type)
    mv     t1, a1              # t1 = arg_array
    mv     t2, a2              # t2 = number of argument bytes remain
    addi   t4, sp, 8           # v0 (t4 in riscv64) points to where to copy arg_array

    # Copy all args into stack
1:
    beqz   t2, 2f
    addi   t2, t2, -4
    lw     t3, 0(t1)           # load from argument array
    addi   t1, t1, 4
    sw     t3, 0(t4)           # save to stack
    addi   t4, t4, 4
    j 1b
2:
    sd     zero, 0(sp)

    # Load float/double args into Floating Argument Registers
    addi   t0, a5, 1           # t0 = shorty[1] (skip 1 for return type)
    mv     t1, a1              # t1 = arg_array

    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f10 sf_a0 sload_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f11 sf_a1 sload_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f12 sf_a2 sload_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f13 sf_a3 sload_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f14 sf_a4 sload_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f15 sf_a5 sload_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f16 sf_a6 sload_integer
    LOOP_OVER_SHORTY_LOADING_FLOAT_REG f17 sf_a7 sload_integer

    # Load int/long args into Integer Argument Registers
sload_integer:
    addi   t0, a5, 1           # t0 = shorty[1] (skip 1 for return type)
    mv     t1, a1              # t1 = arg_array
    li     t5, 0               # t5 = float/double arg count

    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a1 si_a1 call_sfn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a2 si_a2 call_sfn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a3 si_a3 call_sfn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a4 si_a4 call_sfn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a5 si_a5 call_sfn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a6 si_a6 call_sfn
    LOOP_OVER_SHORTY_LOADING_INTEGER_REG a7 si_a7 call_sfn

call_sfn:
    # call method (a0 has been untouched)
    sd     zero, 0(sp)         # store null for ArtMethod* at bottom of frame
    ld     t6, ART_METHOD_QUICK_CODE_OFFSET_64(a0)  # get pointer to the code
    jalr   t6                   # call the method
    mv     sp, s0              # restore sp
    .cfi_def_cfa_register sp

    # pop a4, a5, s11(rSUSPEND), s1(rSELF), s0(fp), ra off of the stack
    ld     a4, 0(sp)
    .cfi_restore a4
    ld     a5, 8(sp)
    .cfi_restore a5
    ld     rSUSPEND, 16(sp)
    .cfi_restore rSUSPEND
    ld     rSELF, 24(sp)
    .cfi_restore rSELF
    ld     s0, 32(sp)
    .cfi_restore s0
    ld     ra, 40(sp)
    .cfi_restore ra
    addi   sp, sp, 48
    .cfi_adjust_cfa_offset -48

    # a4 = JValue* result
    # a5 = shorty string
    lbu   t1, 0(a5)
    li    t2, 'V'
    beq   t1, t2, 3f
    li    t2, 'D'
    beq   t1, t2, 1f
    li    t2, 'F'
    beq   t1, t2, 2f
    sd    a0, 0(a4)
    ret
1:
    fsd   f10, 0(a4)    # Double result
    ret
2:
    fsw   f10, 0(a4)    # Float result
    li    a0,  -1       # NaN-Boxing float result
    sw    a0,  4(a4)
    ret
3:
    ret
END art_quick_invoke_static_stub

    /*
     * Entry from managed code that calls artHandleFillArrayDataFromCode and
     * delivers exception on failure.
     */
    .extern artHandleFillArrayDataFromCode
ENTRY art_quick_handle_fill_data
    SETUP_SAVE_REFS_ONLY_FRAME         # save callee saves in case exception allocation triggers GC
    ld      a2, FRAME_SIZE_SAVE_REFS_ONLY(sp)           # pass referrer's Method*
    mv      a3, rSELF                                   # pass Thread::Current
    jal     artHandleFillArrayDataFromCode              # (payload offset, Array*, method, Thread*)
    RETURN_IF_ZERO
END art_quick_handle_fill_data

    /*
     * Entry from managed code that calls artLockObjectFromCode, may block for GC.
     */
    .extern artLockObjectFromCode
ENTRY art_quick_lock_object
#if 1
    bnez    a0, 1f
    # Jmp to art_quick_throw_null_pointer_exception if a0 == zero
    la      t6, art_quick_throw_null_pointer_exception
    jr      t6
1:
    li      t6, LOCK_WORD_THIN_LOCK_COUNT_ONE
    li      t3, LOCK_WORD_GC_STATE_MASK_SHIFTED_TOGGLED
.Lretry_lock:
    lw      t0, THREAD_ID_OFFSET(rSELF)    # TODO: Can the thread ID really change during the loop?
    addi    t5, a0, MIRROR_OBJECT_LOCK_WORD_OFFSET
    lr.w    t1, (t5)
    and     t2, t1, t3                    # zero the gc bits   
    bnez    t2, .Lnot_unlocked            # already thin locked
    # Unlocked case - $t1: original lock word that's zero except for the read barrier bits.
    or      t2, t1, t0                 # $t2 holds thread id with count of 0 with preserved read barrier bits
    addi    t5, a0, MIRROR_OBJECT_LOCK_WORD_OFFSET
    sc.w    t2, t2, (t5)             # 
    bnez    t2, .Lretry_lock               # store failed, retry
    fence                                  # full (LoadLoad|LoadStore) memory barrier
    #.long   0x01b0000b                     # sync.is
    jr  ra
.Lnot_unlocked:
    # $t1: original lock word, $t0: thread_id with count of 0 and zero read barrier bits
    srl     t2, t1, LOCK_WORD_STATE_SHIFT         # t2 = t1>>LOCK_WORD_STATE_SHIFT
    bnez    t2, .Lslow_lock              # if either of the top two bits are set, go slow path
    xor     t2, t1, t0                  # lock_word.ThreadId() ^ self->ThreadId()
    li      t4, 0xFFFF
    and     t2, t2, t4                   # zero top 16 bits
    bnez    t2, .Lslow_lock              # lock word and self thread id's match -> recursive lock
                                          # otherwise contention, go to slow path
    and     t2, t1, t3                 # zero the gc bits
    add     t2, t2, t6                 # increment count in lock word
    srl     t2, t2, LOCK_WORD_STATE_SHIFT  # if the first gc state bit is set, we overflowed.
    bnez    t2, .Lslow_lock              # if we overflow the count go slow path
    add     t2, t1, t6                   # increment count for real
    addi    t5, a0, MIRROR_OBJECT_LOCK_WORD_OFFSET
    sc.w    t2, t2, (t5)
    bnez    t2, .Lretry_lock             # store failed, retry
    jr  ra
.Lslow_lock:
    # .cpsetup $t9, $t8, art_quick_lock_object
    SETUP_SAVE_REFS_ONLY_FRAME            # save callee saves in case we block
    move    a1, rSELF                    # pass Thread::Current
    jal     artLockObjectFromCode         # (Object* obj, Thread*)
    RETURN_IF_ZERO 
#else
    lw     a1, THREAD_ID_OFFSET(rSELF)
    beqz   a0, art_quick_lock_object_no_inline
                                      // Exclusive load/store has no immediate anymore.
    addi   a4, a0, MIRROR_OBJECT_LOCK_WORD_OFFSET
.Lretry_lock:
    lr.w   a2, (a4)                   // Acquire needed only in most common case.
    xor    a3, a2, a1                 // Prepare the value to store if unlocked
                                      //   (thread id, count of 0 and preserved read barrier bits),
                                      // or prepare to compare thread id for recursive lock check
                                      //   (lock_word.ThreadId() ^ self->ThreadId()).
    li     t1, LOCK_WORD_GC_STATE_MASK_SHIFTED_TOGGLED
    and   t0, a2, t1 // Test the non-gc bits.
    bnez   t0, .Lnot_unlocked             // Check if unlocked.
    // unlocked case - store w3: original lock word plus thread id, preserved read barrier bits.
    sc.w   a2, a3, (a4)
    bnez   a2, .Lretry_lock           // If the store failed, retry.
    ret
.Lnot_unlocked:  // w2: original lock word, w1: thread id, w3: w2 ^ w1
                                      // Check lock word state and thread id together,
    li     t1, LOCK_WORD_STATE_MASK_SHIFTED | LOCK_WORD_THIN_LOCK_OWNER_MASK_SHIFTED
    and    t0, a3, t1 
    bnez   t0, art_quick_lock_object_no_inline
    li     t1, LOCK_WORD_THIN_LOCK_COUNT_ONE
    add    a3, a2, t1  // Increment the recursive lock count.
    li     t1, LOCK_WORD_THIN_LOCK_COUNT_MASK_SHIFTED
    and    t0, a3, t1  // Test the new thin lock count.
    beqz   t0, art_quick_lock_object_no_inline  // Zero as the new count indicates overflow, go slow path.
    sc.w   a2, a3, (a4)
    bnez   a2, .Lretry_lock           // If the store failed, retry.
    ret
#endif
END art_quick_lock_object


    .hidden art_quick_lock_object_no_inline
ENTRY_NO_GP art_quick_lock_object_no_inline
    #beq     a0, zero, art_quick_throw_null_pointer_exception
    bnez    a0, 1f
    # Jmp to art_quick_throw_null_pointer_exception if a0 == zero
    la      t6, art_quick_throw_null_pointer_exception
    jr      t6
1:
    SETUP_SAVE_REFS_ONLY_FRAME            # save callee saves in case we block
    mv      a1, rSELF                    # pass Thread::Current
    jal     artLockObjectFromCode         # (Object* obj, Thread*)
    RETURN_IF_ZERO
END art_quick_lock_object_no_inline

    /*
     * Entry from managed code that calls artUnlockObjectFromCode and delivers exception on failure.
     */
      .extern artUnlockObjectFromCode
ENTRY_NO_GP art_quick_unlock_object
#if 1
    bnez    a0, 1f
    # Jmp to art_quick_throw_null_pointer_exception if a0 == zero
    la      t6, art_quick_throw_null_pointer_exception
    jr      t6
1:
    li      t6, LOCK_WORD_THIN_LOCK_COUNT_ONE
    li      t3, LOCK_WORD_GC_STATE_MASK_SHIFTED_TOGGLED
.Lretry_unlock:
#ifndef USE_READ_BARRIER
    lw      t1, MIRROR_OBJECT_LOCK_WORD_OFFSET(a0)
#else   
    addi    t5, a0, MIRROR_OBJECT_LOCK_WORD_OFFSET    # Need to use atomic read-modify-write for read barrier
    lr.w    t1, (t5)
#endif
    srlw    t2, t1, LOCK_WORD_STATE_SHIFT
    bnez    t2, .Lslow_unlock         # if either of the top two bits are set, go slow path
    lw      t0, THREAD_ID_OFFSET(rSELF)
    and     t2, t1, t3              # zero the gc bits
    xor     t2, t2, t0              # lock_word.ThreadId() ^ self->ThreadId()
    li      t4, 0xFFFF
    and     t2, t2, t4                   # zero top 16 bits
    bnez    t2, .Lslow_unlock         # do lock word and self thread id's match?
    and     t2, t1, t3              # zero the gc bits
    bgeu    t2, t6, .Lrecursive_thin_unlock
    # transition to unlocked
    or      t2, zero, t3            # t2 = LOCK_WORD_GC_STATE_MASK_SHIFTED
    not     t2, t2
    and     t2, t1, t2              # t2: zero except for the preserved gc bits
    fence                           # full (LoadStore|StoreStore) memory barrier
#ifndef USE_READ_BARRIER
    sw      t2, MIRROR_OBJECT_LOCK_WORD_OFFSET(a0)
#else
    addi    t5, a0, MIRROR_OBJECT_LOCK_WORD_OFFSET  # @todo sc      t2, MIRROR_OBJECT_LOCK_WORD_OFFSET(a0)
    sc.w    t2, t2, (t5)
    bnez    t2, .Lretry_unlock        # store failed, retry
#endif
    jr      ra    # @todo jic     ra, 0
.Lrecursive_thin_unlock:
    # t1: original lock word
    sub     t2, t1, t6              # decrement count
#ifndef USE_READ_BARRIER
    sw      t2, MIRROR_OBJECT_LOCK_WORD_OFFSET(a0)
#else
    addi    t5, a0, MIRROR_OBJECT_LOCK_WORD_OFFSET
    sc.w    t2, t2, (t5)
    bnez    t2, .Lretry_unlock        # store failed, retry
#endif
    jr      ra
.Lslow_unlock:
    SETUP_SAVE_REFS_ONLY_FRAME         # save callee saves in case exception allocation triggers GC
    mv      a1, rSELF                 # pass Thread::Current
    jal     artUnlockObjectFromCode    # (Object* obj, Thread*)
    RETURN_IF_ZERO
#else
    lw     a1, THREAD_ID_OFFSET(rSELF)
    beqz   a0, art_quick_unlock_object_no_inline
                                      // Exclusive load/store has no immediate anymore.
    li     t1, MIRROR_OBJECT_LOCK_WORD_OFFSET
    add    a4, a0, t1 
.Lretry_unlock:
#ifndef USE_READ_BARRIER
    lw     a2, (a4)
#else
    lr.w   a2, (a4)                   // Need to use atomic instructions for read barrier.
#endif
    xor    a3, a2, a1                 // Prepare the value to store if simply locked
                                      //   (mostly 0s, and preserved read barrier bits),
                                      // or prepare to compare thread id for recursive lock check
                                      //   (lock_word.ThreadId() ^ self->ThreadId()).
    li     t1, LOCK_WORD_GC_STATE_MASK_SHIFTED_TOGGLED
    and    t0, a3, t1  // Test the non-gc bits.
    bnez   t0, .Lnot_simply_locked        // Locked recursively or by other thread?
    // Transition to unlocked.
#ifndef USE_READ_BARRIER
    sw     a3, (a4)
#else
    sc.w   a2, a3, (a4)               // Need to use atomic instructions for read barrier.
    bnez   a2, .Lretry_unlock         // If the store failed, retry.
#endif
    ret
.Lnot_simply_locked:
                                      // Check lock word state and thread id together,
    li     t1, LOCK_WORD_STATE_MASK_SHIFTED | LOCK_WORD_THIN_LOCK_OWNER_MASK_SHIFTED
    and    t0, a3, t1
    bnez   t0, art_quick_unlock_object_no_inline
    li     t1, LOCK_WORD_THIN_LOCK_COUNT_ONE
    sub    a3, a2, t1  // decrement count
#ifndef USE_READ_BARRIER
    sw     a3, (a4)
#else
    sc.w   a2, a3, (a4)               // Need to use atomic instructions for read barrier.
    bnez   a2, .Lretry_unlock         // If the store failed, retry.
#endif
    ret
#endif
END art_quick_unlock_object

    .hidden art_quick_unlock_object_no_inline
ENTRY_NO_GP art_quick_unlock_object_no_inline
    # beq     a0, zero, art_quick_throw_null_pointer_exception
    bnez    a0, 1f
    # Jmp to art_quick_throw_null_pointer_exception if a0 == zero
    la      t6, art_quick_throw_null_pointer_exception
    jr      t6
1:
    SETUP_SAVE_REFS_ONLY_FRAME         # save callee saves in case exception allocation triggers GC
    mv      a1, rSELF                 # pass Thread::Current
    jal     artUnlockObjectFromCode    # (Object* obj, Thread*)
    RETURN_IF_ZERO
END art_quick_unlock_object_no_inline

    /*
     * Entry from managed code that calls artInstanceOfFromCode and delivers exception on failure.
     */
    .extern artInstanceOfFromCode
    .extern artThrowClassCastExceptionForObject
ENTRY art_quick_check_instance_of
    // Type check using the bit string passes null as the target class. In that case just throw.
    beqz   a1, .Lthrow_class_cast_exception_for_bitstring_check

    addi   sp, sp, -32
    .cfi_adjust_cfa_offset 32
    sd     ra, 24(sp)
    .cfi_rel_offset ra, 24
    sd     t6, 16(sp)
    sd     a1, 8(sp)
    sd     a0, 0(sp)
    jal    artInstanceOfFromCode

    ld     ra, 24(sp)
    beq    a0, zero, .Lthrow_class_cast_exception
    
    addi   sp, sp, 32
    .cfi_adjust_cfa_offset -32
    jalr   zero, ra

.Lthrow_class_cast_exception:
    ld     t6, 16(sp)
    ld     a1, 8(sp)
    ld     a0, 0(sp)
    addi sp, sp, 32
    .cfi_adjust_cfa_offset -32

.Lthrow_class_cast_exception_for_bitstring_check:
    SETUP_SAVE_ALL_CALLEE_SAVES_FRAME
    la    t6, artThrowClassCastExceptionForObject
    mv     a2, rSELF                 # pass Thread::Current
    jalr   zero, t6                 # artThrowClassCastException (Object*, Class*, Thread*)   
END art_quick_check_instance_of


    /*
     * Restore rReg's value from offset($sp) if rReg is not the same as rExclude.
     * nReg is the register number for rReg.
     */
.macro POP_REG_NE rReg, nReg, offset, rExclude
    .ifnc \rReg, \rExclude
        ld \rReg, \offset(sp)      # restore rReg
        .cfi_restore \nReg
    .endif
.endm

    /*
     * Macro to insert read barrier, only used in art_quick_aput_obj.
     * rObj and rDest are registers, offset is a defined literal such as MIRROR_OBJECT_CLASS_OFFSET.
     * TODO: When read barrier has a fast path, add heap unpoisoning support for the fast path.
     */
.macro READ_BARRIER rDest, rObj, offset
#ifdef USE_READ_BARRIER
    # saved registers used in art_quick_aput_obj: a0-a2, t0-t1, t6, ra. 16B-aligned.
    addi   sp, sp, -64
    .cfi_adjust_cfa_offset 64
    sd     ra, 56(sp)
    .cfi_rel_offset ra, 56
    sd     t6, 48(sp)
    .cfi_rel_offset t6, 48
    sd     t1, 40(sp)
    .cfi_rel_offset t1, 40
    sd     t0, 32(sp)
    .cfi_rel_offset t0, 32
    sd     a2, 16(sp)             # padding slot at offset 24 (padding can be any slot in the 64B)
    .cfi_rel_offset a2, 16
    sd     a1, 8(sp)
    .cfi_rel_offset a1, 8
    sd     a0, 0(sp)
    .cfi_rel_offset a0, 0

    # move a0, \rRef               # pass ref in a0 (no-op for now since parameter ref is unused)
    .ifnc \rObj, a1
        mv   a1, \rObj             # pass rObj
    .endif
    addi   a2, zero, \offset      # pass offset
    jal artReadBarrierSlow          # artReadBarrierSlow(ref, rObj, offset)

    # No need to unpoison return value in v0, artReadBarrierSlow() would do the unpoisoning.
    mv    \rDest, a0 # @todo mv    \rDest, v0                # save return value in rDest
                                    # (rDest cannot be v0 in art_quick_aput_obj)

    ld     a0, 0(sp)              # restore registers except rDest
                                    # (rDest can only be t0 or t1 in art_quick_aput_obj)
    .cfi_restore a0
    ld     a1, 8(sp)
    .cfi_restore a1
    ld     a2, 16(sp)
    .cfi_restore a2
    POP_REG_NE t0, 12, 32, \rDest
    POP_REG_NE t1, 13, 40, \rDest
    ld     t6, 48(sp)
    .cfi_restore t6
    ld     ra, 56(sp)             # restore ra
    .cfi_restore ra
    addi   sp, sp, 64
    .cfi_adjust_cfa_offset -64
    SETUP_GP                        # set up gp because we are not returning
#else
    lwu     \rDest, \offset(\rObj)
    UNPOISON_HEAP_REF \rDest
#endif  // USE_READ_BARRIER
.endm

ENTRY art_quick_aput_obj
    beq  a2, zero, .Ldo_aput_null
    READ_BARRIER t0, a0, MIRROR_OBJECT_CLASS_OFFSET
    READ_BARRIER t1, a2, MIRROR_OBJECT_CLASS_OFFSET
    READ_BARRIER t0, t0, MIRROR_CLASS_COMPONENT_TYPE_OFFSET
    bne t1, t0, .Lcheck_assignability  # value's type == array's component type - trivial assignability
.Ldo_aput:
    slli  a1, a1, 2
    add   t0, a0, a1
    POISON_HEAP_REF a2
    sw   a2, MIRROR_OBJECT_ARRAY_DATA_OFFSET(t0)
    ld   t0, THREAD_CARD_TABLE_OFFSET(rSELF)
    srli t1, a0, CARD_TABLE_CARD_SHIFT
    add  t1, t1, t0
    sb   t0, (t1)
    jalr zero, ra
.Ldo_aput_null:
    slli  a1, a1, 2
    add   t0, a0, a1
    sw    a2, MIRROR_OBJECT_ARRAY_DATA_OFFSET(t0)
    jalr  zero, ra
.Lcheck_assignability:
    addi  sp, sp, -64
    .cfi_adjust_cfa_offset 64
    sd     ra, 56(sp)
    .cfi_rel_offset ra, 56
    sd     t6, 24(sp)
    sd     a2, 16(sp)
    sd     a1, 8(sp)
    sd     a0, 0(sp)
    mv     a1, t1
    mv     a0, t0
    jal    artIsAssignableFromCode  # (Class*, Class*)

    // Check for exception
    beqz   a0, .Lthrow_array_store_exception

    ld     ra, 56(sp)
    ld     t6, 24(sp)
    ld     a2, 16(sp)
    ld     a1, 8(sp)
    ld     a0, 0(sp)
    addi   sp, sp, 64
    .cfi_adjust_cfa_offset -64

    j .Ldo_aput
.Lthrow_array_store_exception:
    ld     ra, 56(sp)
    ld     t6, 24(sp)
    ld     a2, 16(sp)
    ld     a1, 8(sp)
    ld     a0, 0(sp)
    addi   sp, sp, 64
    .cfi_adjust_cfa_offset -64

    SETUP_SAVE_ALL_CALLEE_SAVES_FRAME
    mv     a1, a2
    mv     a2, rSELF               # pass Thread::Current
    la     t6, artThrowArrayStoreException
    jalr zero, t6                 # artThrowArrayStoreException(Class*, Class*, Thread*)
END art_quick_aput_obj


// Macros taking opportunity of code similarities for downcalls.
.macro ONE_ARG_REF_DOWNCALL name, entrypoint, return, extend=0
    .extern \entrypoint
ENTRY \name
    SETUP_SAVE_REFS_ONLY_FRAME        # save callee saves in case of GC
    la      t6, \entrypoint
    mv      a1, rSELF                # pass Thread::Current
    jalr    t6                       # (field_idx, Thread*)
    .if     \extend
    slliw    a0, a0, 0               # sign-extend 32-bit result
    .endif
    \return                           # RETURN_IF_NO_EXCEPTION or RETURN_IF_ZERO
END \name
.endm

.macro TWO_ARG_REF_DOWNCALL name, entrypoint, return, extend=0
    .extern \entrypoint
ENTRY \name
    SETUP_SAVE_REFS_ONLY_FRAME        # save callee saves in case of GC
    la      t6, \entrypoint
    mv      a2, rSELF                # pass Thread::Current
    jalr    t6                       # (field_idx, Object*, Thread*) or
                                      # (field_idx, new_val, Thread*)
    
    .if     \extend
    slliw    a0, a0, 0                # sign-extend 32-bit result
    .endif
    \return                           # RETURN_IF_NO_EXCEPTION or RETURN_IF_ZERO
END \name
.endm

.macro THREE_ARG_REF_DOWNCALL name, entrypoint, return, extend=0
    .extern \entrypoint
ENTRY \name
    SETUP_SAVE_REFS_ONLY_FRAME        # save callee saves in case of GC
    la      t6, \entrypoint
    mv      a3, rSELF                # pass Thread::Current
    jalr    t6                       # (field_idx, Object*, new_val, Thread*)
    .if     \extend
    slliw     a0, a0, 0               # sign-extend 32-bit result
    .endif
    \return                           # RETURN_IF_NO_EXCEPTION or RETURN_IF_ZERO
END \name
.endm


    /*
     * Called by managed code to resolve a static/instance field and load/store a value.
     *
     * Note: Functions `art{Get,Set}<Kind>{Static,Instance}FromCompiledCode` are
     * defined with a macro in runtime/entrypoints/quick/quick_field_entrypoints.cc.
     */
ONE_ARG_REF_DOWNCALL art_quick_get_byte_static, artGetByteStaticFromCompiledCode, RETURN_IF_NO_EXCEPTION
ONE_ARG_REF_DOWNCALL art_quick_get_boolean_static, artGetBooleanStaticFromCompiledCode, RETURN_IF_NO_EXCEPTION
ONE_ARG_REF_DOWNCALL art_quick_get_short_static, artGetShortStaticFromCompiledCode, RETURN_IF_NO_EXCEPTION
ONE_ARG_REF_DOWNCALL art_quick_get_char_static, artGetCharStaticFromCompiledCode, RETURN_IF_NO_EXCEPTION
ONE_ARG_REF_DOWNCALL art_quick_get32_static, artGet32StaticFromCompiledCode, RETURN_IF_NO_EXCEPTION, 1
ONE_ARG_REF_DOWNCALL art_quick_get_obj_static, artGetObjStaticFromCompiledCode, RETURN_IF_NO_EXCEPTION
ONE_ARG_REF_DOWNCALL art_quick_get64_static, artGet64StaticFromCompiledCode, RETURN_IF_NO_EXCEPTION
TWO_ARG_REF_DOWNCALL art_quick_get_byte_instance, artGetByteInstanceFromCompiledCode, RETURN_IF_NO_EXCEPTION
TWO_ARG_REF_DOWNCALL art_quick_get_boolean_instance, artGetBooleanInstanceFromCompiledCode, RETURN_IF_NO_EXCEPTION
TWO_ARG_REF_DOWNCALL art_quick_get_short_instance, artGetShortInstanceFromCompiledCode, RETURN_IF_NO_EXCEPTION
TWO_ARG_REF_DOWNCALL art_quick_get_char_instance, artGetCharInstanceFromCompiledCode, RETURN_IF_NO_EXCEPTION
TWO_ARG_REF_DOWNCALL art_quick_get32_instance, artGet32InstanceFromCompiledCode, RETURN_IF_NO_EXCEPTION, 1
TWO_ARG_REF_DOWNCALL art_quick_get_obj_instance, artGetObjInstanceFromCompiledCode, RETURN_IF_NO_EXCEPTION
TWO_ARG_REF_DOWNCALL art_quick_get64_instance, artGet64InstanceFromCompiledCode, RETURN_IF_NO_EXCEPTION
TWO_ARG_REF_DOWNCALL art_quick_set8_static, artSet8StaticFromCompiledCode, RETURN_IF_ZERO
TWO_ARG_REF_DOWNCALL art_quick_set16_static, artSet16StaticFromCompiledCode, RETURN_IF_ZERO
TWO_ARG_REF_DOWNCALL art_quick_set32_static, artSet32StaticFromCompiledCode, RETURN_IF_ZERO
TWO_ARG_REF_DOWNCALL art_quick_set_obj_static, artSetObjStaticFromCompiledCode, RETURN_IF_ZERO
TWO_ARG_REF_DOWNCALL art_quick_set64_static, artSet64StaticFromCompiledCode, RETURN_IF_ZERO
THREE_ARG_REF_DOWNCALL art_quick_set8_instance, artSet8InstanceFromCompiledCode, RETURN_IF_ZERO
THREE_ARG_REF_DOWNCALL art_quick_set16_instance, artSet16InstanceFromCompiledCode, RETURN_IF_ZERO
THREE_ARG_REF_DOWNCALL art_quick_set32_instance, artSet32InstanceFromCompiledCode, RETURN_IF_ZERO
THREE_ARG_REF_DOWNCALL art_quick_set_obj_instance, artSetObjInstanceFromCompiledCode, RETURN_IF_ZERO
THREE_ARG_REF_DOWNCALL art_quick_set64_instance, artSet64InstanceFromCompiledCode, RETURN_IF_ZERO

// Macro to facilitate adding new allocation entrypoints.
.macro ONE_ARG_DOWNCALL name, entrypoint, return
    .extern \entrypoint
ENTRY \name
    SETUP_SAVE_REFS_ONLY_FRAME         # save callee saves in case of GC
    mv      a1, rSELF                 # pass Thread::Current
    jal     \entrypoint
    \return
END \name
.endm

// Macro to facilitate adding new allocation entrypoints.
.macro TWO_ARG_DOWNCALL name, entrypoint, return
    .extern \entrypoint
ENTRY \name
    SETUP_SAVE_REFS_ONLY_FRAME         # save callee saves in case of GC
    mv      a2, rSELF                 # pass Thread::Current
    jal     \entrypoint
    \return
END \name
.endm

.macro THREE_ARG_DOWNCALL name, entrypoint, return
    .extern \entrypoint
ENTRY \name
    SETUP_SAVE_REFS_ONLY_FRAME         # save callee saves in case of GC
    mv      a3, rSELF                 # pass Thread::Current
    jal     \entrypoint
    \return
END \name
.endm

.macro FOUR_ARG_DOWNCALL name, entrypoint, return
    .extern \entrypoint
ENTRY \name
    SETUP_SAVE_REFS_ONLY_FRAME         # save callee saves in case of GC
    mv      a4, rSELF                 # pass Thread::Current
    jal     \entrypoint
    \return
END \name
.endm

// Generate the allocation entrypoints for each allocator.
GENERATE_ALLOC_ENTRYPOINTS_FOR_NON_TLAB_ALLOCATORS
// Comment out allocators that have riscv64 specific asm.
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_OBJECT_RESOLVED(_region_tlab, RegionTLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_OBJECT_INITIALIZED(_region_tlab, RegionTLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_OBJECT_WITH_ACCESS_CHECK(_region_tlab, RegionTLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_STRING_OBJECT(_region_tlab, RegionTLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED(_region_tlab, RegionTLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED8(_region_tlab, RegionTLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED16(_region_tlab, RegionTLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED32(_region_tlab, RegionTLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED64(_region_tlab, RegionTLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_STRING_FROM_BYTES(_region_tlab, RegionTLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_STRING_FROM_CHARS(_region_tlab, RegionTLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_STRING_FROM_STRING(_region_tlab, RegionTLAB)

// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_OBJECT_RESOLVED(_tlab, TLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_OBJECT_INITIALIZED(_tlab, TLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_OBJECT_WITH_ACCESS_CHECK(_tlab, TLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_STRING_OBJECT(_tlab, TLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED(_tlab, TLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED8(_tlab, TLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED16(_tlab, TLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED32(_tlab, TLAB)
// GENERATE_ALLOC_ENTRYPOINTS_ALLOC_ARRAY_RESOLVED64(_tlab, TLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_STRING_FROM_BYTES(_tlab, TLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_STRING_FROM_CHARS(_tlab, TLAB)
GENERATE_ALLOC_ENTRYPOINTS_ALLOC_STRING_FROM_STRING(_tlab, TLAB)

// A hand-written override for:
//   GENERATE_ALLOC_ENTRYPOINTS_ALLOC_OBJECT_RESOLVED(_rosalloc, RosAlloc)
//   GENERATE_ALLOC_ENTRYPOINTS_ALLOC_OBJECT_INITIALIZED(_rosalloc, RosAlloc)
.macro ART_QUICK_ALLOC_OBJECT_ROSALLOC c_name, cxx_name, isInitialized
ENTRY_NO_GP \c_name
    # Fast path rosalloc allocation
    # a0: type
    # s1: Thread::Current
    # -----------------------------
    # t1: object size
    # t2: rosalloc run
    # t3: thread stack top offset
    # a4: thread stack bottom offset
    # v0: free list head
    #
    # a5, a6 : temps
    ld     t3, THREAD_LOCAL_ALLOC_STACK_TOP_OFFSET(s1)    # Check if thread local allocation stack
    ld     a4, THREAD_LOCAL_ALLOC_STACK_END_OFFSET(s1)    # has any room left.
    bgeu   t3, a4, .Lslow_path_\c_name

    lwu    t1, MIRROR_CLASS_OBJECT_SIZE_ALLOC_FAST_PATH_OFFSET(a0)  # Load object size (t1).
    li     a5, ROSALLOC_MAX_THREAD_LOCAL_BRACKET_SIZE      # Check if size is for a thread local
                                                            # allocation. Also does the initialized
                                                            # and finalizable checks.
    # When isInitialized == 0, then the class is potentially not yet initialized.
    # If the class is not yet initialized, the object size will be very large to force the branch
    # below to be taken.
    #
    # See InitializeClassVisitors in class-inl.h for more details.
    bltu   a5, t1, .Lslow_path_\c_name

    # Compute the rosalloc bracket index from the size. Since the size is already aligned we can
    # combine the two shifts together.
    srli   t1, t1, (ROSALLOC_BRACKET_QUANTUM_SIZE_SHIFT - POINTER_SIZE_SHIFT)

    add    t2, t1, s1
    ld     t2, (THREAD_ROSALLOC_RUNS_OFFSET - __SIZEOF_POINTER__)(t2)  # Load rosalloc run (t2).

    # Load the free list head (v0).
    # NOTE: this will be the return val.
    ld     t4, (ROSALLOC_RUN_FREE_LIST_OFFSET + ROSALLOC_RUN_FREE_LIST_HEAD_OFFSET)(t2)
    beqz   t4, .Lslow_path_\c_name

    # Load the next pointer of the head and update the list head with the next pointer.
    ld     a5, ROSALLOC_SLOT_NEXT_OFFSET(t4)
    sd     a5, (ROSALLOC_RUN_FREE_LIST_OFFSET + ROSALLOC_RUN_FREE_LIST_HEAD_OFFSET)(t2)

    # Store the class pointer in the header. This also overwrites the first pointer. The offsets are
    # asserted to match.

#if ROSALLOC_SLOT_NEXT_OFFSET != MIRROR_OBJECT_CLASS_OFFSET
#error "Class pointer needs to overwrite next pointer."
#endif

    POISON_HEAP_REF a0
    sw     a0, MIRROR_OBJECT_CLASS_OFFSET(t4)

    # Push the new object onto the thread local allocation stack and increment the thread local
    # allocation stack top.
    sw     t4, 0(t3)
    addi   t3, t3, COMPRESSED_REFERENCE_SIZE
    sd     t3, THREAD_LOCAL_ALLOC_STACK_TOP_OFFSET(s1)

    # Decrement the size of the free list.
    lw     a5, (ROSALLOC_RUN_FREE_LIST_OFFSET + ROSALLOC_RUN_FREE_LIST_SIZE_OFFSET)(t2)
    addiw a5, a5, -1 # @todo addiu  a5, a5, -1
    sw     a5, (ROSALLOC_RUN_FREE_LIST_OFFSET + ROSALLOC_RUN_FREE_LIST_SIZE_OFFSET)(t2)

.if \isInitialized == 0
    # This barrier is only necessary when the allocation also requires a class initialization check.
    #
    # If the class is already observably initialized, then new-instance allocations are protected
    # from publishing by the compiler which inserts its own StoreStore barrier.
    fence                        # Fence.
.endif
    mv      a0,   t4
    jr      ra 

.Lslow_path_\c_name:
    SETUP_SAVE_REFS_ONLY_FRAME
    mv     a0,   t4
    mv     a1 ,s1                              # Pass self as argument.
    jal    \cxx_name
    RETURN_IF_RESULT_IS_NON_ZERO_OR_DELIVER
END \c_name
.endm

ART_QUICK_ALLOC_OBJECT_ROSALLOC art_quick_alloc_object_resolved_rosalloc, artAllocObjectFromCodeResolvedRosAlloc, /* isInitialized */ 0
ART_QUICK_ALLOC_OBJECT_ROSALLOC art_quick_alloc_object_initialized_rosalloc, artAllocObjectFromCodeInitializedRosAlloc, /* isInitialized */ 1

// The common fast path code for art_quick_alloc_object_resolved/initialized_tlab
// and art_quick_alloc_object_resolved/initialized_region_tlab.
//
// a0: type, s1(rSELF): Thread::Current
// Need to preserve a0 to the slow path.
//
// If isInitialized=1 then the compiler assumes the object's class has already been initialized.
// If isInitialized=0 the compiler can only assume it's been at least resolved.
.macro ALLOC_OBJECT_RESOLVED_TLAB_FAST_PATH slowPathLabel isInitialized
    ld     t3, THREAD_LOCAL_POS_OFFSET(rSELF)         # Load thread_local_pos.
    ld     a2, THREAD_LOCAL_END_OFFSET(rSELF)         # Load thread_local_end.
    lwu    t0, MIRROR_CLASS_OBJECT_SIZE_ALLOC_FAST_PATH_OFFSET(a0)  # Load the object size.
    add    a3, t3, t0                                 # Add object size to tlab pos.

    # When isInitialized == 0, then the class is potentially not yet initialized.
    # If the class is not yet initialized, the object size will be very large to force the branch
    # below to be taken.
    #
    # See InitializeClassVisitors in class-inl.h for more details.
    bltu   a2, a3, \slowPathLabel                    # Check if it fits, overflow works since the
                                                       # tlab pos and end are 32 bit values.
    # "Point of no slow path". Won't go to the slow path from here on.
    sd     a3, THREAD_LOCAL_POS_OFFSET(rSELF)         # Store new thread_local_pos.
    ld     a2, THREAD_LOCAL_OBJECTS_OFFSET(rSELF)     # Increment thread_local_objects.
    addi   a2, a2, 1
    sd     a2, THREAD_LOCAL_OBJECTS_OFFSET(rSELF)
    POISON_HEAP_REF a0
    sw     a0, MIRROR_OBJECT_CLASS_OFFSET(t3)         # Store the class pointer.

.if \isInitialized == 0
    # This barrier is only necessary when the allocation also requires a class initialization check.
    #
    # If the class is already observably initialized, then new-instance allocations are protected
    # from publishing by the compiler which inserts its own StoreStore barrier.
    fence                                            # Fence.
.endif
    mv     a0,  t3
    jr     ra
.endm

// The common code for art_quick_alloc_object_resolved/initialized_tlab
// and art_quick_alloc_object_resolved/initialized_region_tlab.
.macro GENERATE_ALLOC_OBJECT_TLAB name, entrypoint, isInitialized
ENTRY_NO_GP \name
    # Fast path tlab allocation.
    # a0: type, s1(rSELF): Thread::Current.
    ALLOC_OBJECT_RESOLVED_TLAB_FAST_PATH .Lslow_path_\name, \isInitialized
.Lslow_path_\name:
    SETUP_SAVE_REFS_ONLY_FRAME                         # Save callee saves in case of GC.
    mv     a1, rSELF                                  # Pass Thread::Current.
    jal    \entrypoint                                 # (mirror::Class*, Thread*)    
    RETURN_IF_RESULT_IS_NON_ZERO_OR_DELIVER
END \name
.endm

GENERATE_ALLOC_OBJECT_TLAB art_quick_alloc_object_resolved_region_tlab, artAllocObjectFromCodeResolvedRegionTLAB, /* isInitialized */ 0
GENERATE_ALLOC_OBJECT_TLAB art_quick_alloc_object_initialized_region_tlab, artAllocObjectFromCodeInitializedRegionTLAB, /* isInitialized */ 1
GENERATE_ALLOC_OBJECT_TLAB art_quick_alloc_object_resolved_tlab, artAllocObjectFromCodeResolvedTLAB, /* isInitialized */ 0
GENERATE_ALLOC_OBJECT_TLAB art_quick_alloc_object_initialized_tlab, artAllocObjectFromCodeInitializedTLAB, /* isInitialized */ 1

// The common fast path code for art_quick_alloc_array_resolved/initialized_tlab
// and art_quick_alloc_array_resolved/initialized_region_tlab.
//
// a0: type, a1: component_count, a2: total_size, s1(rSELF): Thread::Current.
// Need to preserve a0 and a1 to the slow path.
.macro ALLOC_ARRAY_TLAB_FAST_PATH_RESOLVED_WITH_SIZE slowPathLabel
    li    a3, OBJECT_ALIGNMENT_MASK_TOGGLED64       # Apply alignemnt mask (addr + 7) & ~7.
    and    a2, a2, a3                               # The mask must be 64 bits to keep high
                                                       # bits in case of overflow.
    # Negative sized arrays are handled here since a1 holds a zero extended 32 bit value.
    # Negative ints become large 64 bit unsigned ints which will always be larger than max signed
    # 32 bit int. Since the max shift for arrays is 3, it can not become a negative 64 bit int.
    li    a3, MIN_LARGE_OBJECT_THRESHOLD
    bgeu  a2, a3, \slowPathLabel                    # Possibly a large object, go slow path.

    ld     t3, THREAD_LOCAL_POS_OFFSET(rSELF)       # Load thread_local_pos.
    ld     t1, THREAD_LOCAL_END_OFFSET(rSELF)       # Load thread_local_end.
    sub    t2, t1, t3                               # Compute the remaining buffer size.
    bltu   t2, a2, \slowPathLabel                   # Check tlab for space, note that we use
                                                    # (end - begin) to handle negative size
                                                    # arrays. It is assumed that a negative size
                                                    # will always be greater unsigned than region
                                                    # size.

    # "Point of no slow path". Won't go to the slow path from here on.
    add    a2, t3, a2                               # Add object size to tlab pos.
    sd     a2, THREAD_LOCAL_POS_OFFSET(rSELF)         # Store new thread_local_pos.
    ld     a2, THREAD_LOCAL_OBJECTS_OFFSET(rSELF)     # Increment thread_local_objects.
    addi   a2, a2, 1
    sd     a2, THREAD_LOCAL_OBJECTS_OFFSET(rSELF)
    POISON_HEAP_REF a0
    sw     a0, MIRROR_OBJECT_CLASS_OFFSET(t3)        # Store the class pointer.
    sw     a1, MIRROR_ARRAY_LENGTH_OFFSET(t3)        # Store the array length.

    mv     a0, t3            ### zhengxing: TBD
    jr     ra
.endm

.macro GENERATE_ALLOC_ARRAY_TLAB name, entrypoint, size_setup
ENTRY_NO_GP \name
    # Fast path array allocation for region tlab allocation.
    # a0: mirror::Class* type
    # a1: int32_t component_count
    # s1(rSELF): Thread::Current
    slli   a4, a1, 32                              # Create zero-extended component_count. Value
    srli   a4, a4, 32                              # in a1 is preserved in a case of slow path.

    \size_setup .Lslow_path_\name
    ALLOC_ARRAY_TLAB_FAST_PATH_RESOLVED_WITH_SIZE .Lslow_path_\name
.Lslow_path_\name:
    # a0: mirror::Class* type
    # a1: int32_t component_count
    # a2: Thread* self

    SETUP_SAVE_REFS_ONLY_FRAME                        # Save callee saves in case of GC.
    mv     a2, rSELF                                  # Pass Thread::Current.
    jal    \entrypoint
    RETURN_IF_RESULT_IS_NON_ZERO_OR_DELIVER
END \name
.endm

.macro COMPUTE_ARRAY_SIZE_UNKNOWN slow_path
    # Array classes are never finalizable or uninitialized, no need to check.
    lwu    a3, MIRROR_CLASS_COMPONENT_TYPE_OFFSET(a0) # Load component type.
    UNPOISON_HEAP_REF a3
    lw     a3, MIRROR_CLASS_OBJECT_PRIMITIVE_TYPE_OFFSET(a3)
    srli   a3, a3, PRIMITIVE_TYPE_SIZE_SHIFT_SHIFT   # Component size shift is in high 16 bits.
    sll    a2, a4, a3                               # Calculate data size.
                                                       # Add array data offset and alignment.
    addi   a2, a2, (MIRROR_INT_ARRAY_DATA_OFFSET + OBJECT_ALIGNMENT_MASK)
#if MIRROR_WIDE_ARRAY_DATA_OFFSET != MIRROR_INT_ARRAY_DATA_OFFSET + 4
#error Long array data offset must be 4 greater than int array data offset.
#endif

    addi   a3, a3, 1                                 # Add 4 to the length only if the component
    andi   a3, a3, 4                                 # size shift is 3 (for 64 bit alignment).
    add    a2, a2, a3
.endm

.macro COMPUTE_ARRAY_SIZE_8 slow_path
    # Add array data offset and alignment.
    addi   a2, a4, (MIRROR_INT_ARRAY_DATA_OFFSET + OBJECT_ALIGNMENT_MASK)
.endm

.macro COMPUTE_ARRAY_SIZE_16 slow_path
    slli   a2, a4, 1
    # Add array data offset and alignment.
    addi   a2, a2, (MIRROR_INT_ARRAY_DATA_OFFSET + OBJECT_ALIGNMENT_MASK)
.endm

.macro COMPUTE_ARRAY_SIZE_32 slow_path
    slli   a2, a4, 2
    # Add array data offset and alignment.
    addi   a2, a2, (MIRROR_INT_ARRAY_DATA_OFFSET + OBJECT_ALIGNMENT_MASK)
.endm

.macro COMPUTE_ARRAY_SIZE_64 slow_path
    slli   a2, a4, 3
    # Add array data offset and alignment.
    addi   a2, a2, (MIRROR_WIDE_ARRAY_DATA_OFFSET + OBJECT_ALIGNMENT_MASK)
.endm

GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved_region_tlab, artAllocArrayFromCodeResolvedRegionTLAB, COMPUTE_ARRAY_SIZE_UNKNOWN
GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved8_region_tlab, artAllocArrayFromCodeResolvedRegionTLAB, COMPUTE_ARRAY_SIZE_8
GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved16_region_tlab, artAllocArrayFromCodeResolvedRegionTLAB, COMPUTE_ARRAY_SIZE_16
GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved32_region_tlab, artAllocArrayFromCodeResolvedRegionTLAB, COMPUTE_ARRAY_SIZE_32
GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved64_region_tlab, artAllocArrayFromCodeResolvedRegionTLAB, COMPUTE_ARRAY_SIZE_64

GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved_tlab, artAllocArrayFromCodeResolvedTLAB, COMPUTE_ARRAY_SIZE_UNKNOWN
GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved8_tlab, artAllocArrayFromCodeResolvedTLAB, COMPUTE_ARRAY_SIZE_8
GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved16_tlab, artAllocArrayFromCodeResolvedTLAB, COMPUTE_ARRAY_SIZE_16
GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved32_tlab, artAllocArrayFromCodeResolvedTLAB, COMPUTE_ARRAY_SIZE_32
GENERATE_ALLOC_ARRAY_TLAB art_quick_alloc_array_resolved64_tlab, artAllocArrayFromCodeResolvedTLAB, COMPUTE_ARRAY_SIZE_64

    /*
     * Macro for resolution and initialization of indexed DEX file
     * constants such as classes and strings. $a0 is both input and
     * output.
     */
.macro ONE_ARG_SAVE_EVERYTHING_DOWNCALL name, entrypoint, runtime_method_offset = RUNTIME_SAVE_EVERYTHING_METHOD_OFFSET
    .extern \entrypoint
ENTRY_NO_GP \name
    SETUP_SAVE_EVERYTHING_FRAME \runtime_method_offset  # Save everything in case of GC.
    la      t6, \entrypoint
    move    a1, rSELF                # Pass Thread::Current (in delay slot).
    jalr    t6                       # (uint32_t index, Thread*)
    beqz    a0, 1f
    RESTORE_SAVE_EVERYTHING_FRAME 0  # Restore everything except $a0.
    jalr    zero, ra                 # Return on success.
1:
    DELIVER_PENDING_EXCEPTION_FRAME_READY
END \name
.endm

.macro ONE_ARG_SAVE_EVERYTHING_DOWNCALL_FOR_CLINIT name, entrypoint
    ONE_ARG_SAVE_EVERYTHING_DOWNCALL \name, \entrypoint, RUNTIME_SAVE_EVERYTHING_FOR_CLINIT_METHOD_OFFSET
.endm

    /*
     * Entry from managed code to resolve a method handle. On entry, A0 holds the method handle
     * index. On success the MethodHandle is returned, otherwise an exception is raised.
     */
ONE_ARG_SAVE_EVERYTHING_DOWNCALL art_quick_resolve_method_handle, artResolveMethodHandleFromCode

    /*
     * Entry from managed code to resolve a method type. On entry, A0 holds the method type index.
     * On success the MethodType is returned, otherwise an exception is raised.
     */
ONE_ARG_SAVE_EVERYTHING_DOWNCALL art_quick_resolve_method_type, artResolveMethodTypeFromCode

    /*
     * Entry from managed code to resolve a string, this stub will allocate a String and deliver an
     * exception on error. On success the String is returned. A0 holds the string index. The fast
     * path check for hit in strings cache has already been performed.
     */
ONE_ARG_SAVE_EVERYTHING_DOWNCALL art_quick_resolve_string, artResolveStringFromCode

    /*
     * Entry from managed code when uninitialized static storage, this stub will run the class
     * initializer and deliver the exception on error. On success the static storage base is
     * returned.
     */
ONE_ARG_SAVE_EVERYTHING_DOWNCALL_FOR_CLINIT art_quick_initialize_static_storage, artInitializeStaticStorageFromCode

    /*
     * Entry from managed code when dex cache misses for a type_idx.
     */
ONE_ARG_SAVE_EVERYTHING_DOWNCALL_FOR_CLINIT art_quick_resolve_type, artResolveTypeFromCode

    /*
     * Entry from managed code when type_idx needs to be checked for access and dex cache may also
     * miss.
     */
ONE_ARG_SAVE_EVERYTHING_DOWNCALL art_quick_resolve_type_and_verify_access, artResolveTypeAndVerifyAccessFromCode

    /*
     * Called by managed code when the value in rSUSPEND has been decremented to 0.
     */
       .extern artTestSuspendFromCode
ENTRY_NO_GP art_quick_test_suspend
    SETUP_SAVE_EVERYTHING_FRAME RUNTIME_SAVE_EVERYTHING_FOR_SUSPEND_CHECK_METHOD_OFFSET
                                              # save everything for stack crawl
    mv     a0, rSELF
    jal    artTestSuspendFromCode             # (Thread*)
  
    RESTORE_SAVE_EVERYTHING_FRAME
    jalr   zero, ra
END art_quick_test_suspend

    /*
     * Called by managed code that is attempting to call a method on a proxy class. On entry
     * r0 holds the proxy method; r1, r2 and r3 may contain arguments.
     */
    .extern artQuickProxyInvokeHandler
ENTRY art_quick_proxy_invoke_handler
    SETUP_SAVE_REFS_AND_ARGS_FRAME_WITH_METHOD_IN_A0
    mv      a2, rSELF             # pass Thread::Current
    mv      a3, sp               # pass $sp
    jal     artQuickProxyInvokeHandler  # (Method* proxy method, receiver, Thread*, SP)
    ld      a2, THREAD_EXCEPTION_OFFSET(rSELF) # load Thread::Current()->exception_
    bne     a2, zero, 1f
    RESTORE_SAVE_REFS_AND_ARGS_FRAME
    fmv.d.x f10, a0                # place return value to FP return value
    jalr    zero, ra
1:
    RESTORE_SAVE_REFS_AND_ARGS_FRAME
    DELIVER_PENDING_EXCEPTION
END art_quick_proxy_invoke_handler

    /*
     * Called to resolve an imt conflict.
     * a0 is the conflict ArtMethod.
     * t0 is a hidden argument that holds the target interface method's dex method index.
     */
ENTRY art_quick_imt_conflict_trampoline
    ld t6, ART_METHOD_JNI_OFFSET_64(a0)  // Load ImtConflictTable
    ld a0, 0(t6)                         // Load first entry in ImtConflictTable.
.Limt_table_iterate:
    // Branch if found. Benchmarks have shown doing a branch here is better.
    beq a0, t0, .Limt_table_found

    // If the entry is null, the interface method is not in the ImtConflictTable.
    beqz a0, .Lconflict_trampoline
    // Iterate over the entries of the ImtConflictTable.
    addi t6, t6, (2 * __SIZEOF_POINTER__)
    ld a0, 0(t6)
    j .Limt_table_iterate
.Limt_table_found:
    // We successfully hit an entry in the table. Load the target method
    // and jump to it.
    ld a0, __SIZEOF_POINTER__(t6)
    ld t6, ART_METHOD_QUICK_CODE_OFFSET_64(a0)
    jr      t6
.Lconflict_trampoline:
    // Call the runtime stub to populate the ImtConflictTable and jump to the
    // resolved method.
    move a0, t0  // Load interface method
    INVOKE_TRAMPOLINE_BODY artInvokeInterfaceTrampoline
END art_quick_imt_conflict_trampoline

   .extern artQuickResolutionTrampoline
ENTRY art_quick_resolution_trampoline
    SETUP_SAVE_REFS_AND_ARGS_FRAME
    mv      a2, rSELF             # pass Thread::Current
    mv      a3, sp               # pass $sp
    jal     artQuickResolutionTrampoline  # (Method* called, receiver, Thread*, SP)
    beq     a0, zero, 1f
    mv      t6, a0               # code pointer must be in $t6 to generate the global pointer
    ld      a0, 0(sp)            # load resolved method in $a0
                                   # artQuickResolutionTrampoline puts resolved method in *SP
    RESTORE_SAVE_REFS_AND_ARGS_FRAME
    jalr    zero, t6             # tail call to method
1:
    RESTORE_SAVE_REFS_AND_ARGS_FRAME
    DELIVER_PENDING_EXCEPTION
END art_quick_resolution_trampoline

    .extern artQuickGenericJniTrampoline
    .extern artQuickGenericJniEndTrampoline
ENTRY art_quick_generic_jni_trampoline
    SETUP_SAVE_REFS_AND_ARGS_FRAME_WITH_METHOD_IN_A0
    mv      s0, sp               # save $sp to fp(s0)
    .cfi_def_cfa_register s0

    # prepare for call to artQuickGenericJniTrampoline(Thread*, SP)
    // (Thread*, managed_sp, reserved_area)
    //    a0         a1            a2   <= C calling convention
    //  rSELF      fp(s0)          sp   <= where they are

    addi    sp, sp, -2047        # reserve space on the stack. -5120 bytes
    addi    sp, sp, -2047
    addi    sp, sp, -1026

    mv      a0, rSELF             # pass Thread::Current
    mv      a1, s0               # pass $sp
    mv      a2, sp               # pass $sp
    jal     artQuickGenericJniTrampoline   # (Thread*, SP)

    // The C call will have registered the complete save-frame on success.
    // The result of the call is:
    //     a0: pointer to native code, 0 on error.
    //     The bottom of the reserved area contains values for arg registers,
    //     hidden arg register and SP for out args for the call.

    // Check for error (class init check or locking for synchronized native method can throw).
    beq     a0, zero, .Lexception_in_native

    // Save the code pointer
    mv      t6, a0               # save the code ptr

    # Load parameters from stack into registers
    ld      a0,   0(sp)
    ld      a1,   8(sp)
    ld      a2,  16(sp)
    ld      a3,  24(sp)
    ld      a4,  32(sp)
    ld      a5,  40(sp)
    ld      a6,  48(sp)
    ld      a7,  56(sp)

    fld     f10,  64(sp)
    fld     f11,  72(sp)
    fld     f12, 80(sp)
    fld     f13, 88(sp)
    fld     f14, 96(sp)
    fld     f15, 104(sp)
    fld     f16, 112(sp)
    fld     f17, 120(sp)

    // Load hidden arg (T0) for @CriticalNative and SP for out args.
    ld      t0,  128(sp)

    // Apply the new SP for out args, releasing unneeded reserved area.
    ld      sp,  136(sp)

    jalr    t6                    # native call

    # result sign extension is handled in C code
    # prepare for call to artQuickGenericJniEndTrampoline(Thread*, result, result_f)
    mv      a1, a0
    mv      a0, rSELF             # pass Thread::Current
    fmv.x.d   a2, f10   
    jal     artQuickGenericJniEndTrampoline

    // Pending exceptions possible.
    ld      t0, THREAD_EXCEPTION_OFFSET(rSELF) # load Thread::Current()->exception_
    bne     t0, zero, .Lexception_in_native         # check for pending exceptions

    // Tear down the alloca.
    mv      sp, s0               # tear down the alloca (fp(s0) --> sp)
    .cfi_remember_state
    .cfi_def_cfa_register sp

    # tear dpown the callee-save frame
    RESTORE_SAVE_REFS_AND_ARGS_FRAME

    fmv.d.x f10, a0              # place return value to FP return value
    ret

    // Undo the unwinding information from above since it doesn't apply below.
    .cfi_restore_state
    .cfi_def_cfa s0, FRAME_SIZE_SAVE_REFS_AND_ARGS
.Lexception_in_native:
    ld      t0, THREAD_TOP_QUICK_FRAME_OFFSET(rSELF)
    addi    sp, t0, -1  // Remove the GenericJNI tag.
    # This will create a new save-all frame, required by the runtime.
    DELIVER_PENDING_EXCEPTION
END art_quick_generic_jni_trampoline

    .extern artQuickToInterpreterBridge
ENTRY art_quick_to_interpreter_bridge
    SETUP_SAVE_REFS_AND_ARGS_FRAME
    mv      a1, rSELF             # pass Thread::Current
    mv      a2, sp               # pass $sp
    jal     artQuickToInterpreterBridge    # (Method* method, Thread*, SP)

    ld      t0, THREAD_EXCEPTION_OFFSET(rSELF) # load Thread::Current()->exception_
    addi    sp, sp, REFS_AND_ARGS_MINUS_REFS_SIZE  # skip a0-a7 and f10-f17
    RESTORE_SAVE_REFS_ONLY_FRAME
    fmv.d.x f10, a0               # place return value to FP return value
    bne     t0, zero, 1f
    jalr    zero, ra
1:
    DELIVER_PENDING_EXCEPTION
END art_quick_to_interpreter_bridge


    .extern artInvokeObsoleteMethod
ENTRY art_invoke_obsolete_method_stub
    SETUP_SAVE_ALL_CALLEE_SAVES_FRAME
    mv      a1, rSELF                 # pass Thread::Current
    jal     artInvokeObsoleteMethod    # (Method* method, Thread* self)
END art_invoke_obsolete_method_stub

 .extern artInstrumentationMethodEntryFromCode
    .extern artInstrumentationMethodExitFromCode
ENTRY art_quick_instrumentation_entry
    SETUP_SAVE_REFS_AND_ARGS_FRAME
    # Preserve $a0 knowing there is a spare slot in kSaveRefsAndArgs.
    sd      a0, 8(sp)     # Save arg0.
    mv      a3, sp        # Pass $sp.
    mv      a2, rSELF      # pass Thread::Current
    jal     artInstrumentationMethodEntryFromCode  # (Method*, Object*, Thread*, SP)
    beq     a0, zero, .Ldeliver_instrumentation_entry_exception
                            # Deliver exception if we got nullptr as function.
    mv      t6, a0        # $t6 holds reference to code
    ld      a0, 8(sp)     # Restore arg0.
    RESTORE_SAVE_REFS_AND_ARGS_FRAME
    la      ra, art_quick_instrumentation_exit
    jalr    zero, 0(t6) # jr    t6  # jic    t6, 0          # call method, returning to art_quick_instrumentation_exit  todo: lyh
.Ldeliver_instrumentation_entry_exception:
    RESTORE_SAVE_REFS_AND_ARGS_FRAME
    DELIVER_PENDING_EXCEPTION
END art_quick_instrumentation_entry

    .hidden art_quick_instrumentation_exit
ENTRY_NO_GP art_quick_instrumentation_exit
    mv    ra, zero      # RA points here, so clobber with 0 for later checks.
    SETUP_SAVE_EVERYTHING_FRAME

    addi    a3, sp, 96    #offset 96   Pass fpr_res pointer ($fa0 in SAVE_EVERYTHING_FRAME).
    addi    a2, sp, 304   #offset 304  Pass gpr_res pointer ($a0 in SAVE_EVERYTHING_FRAME).
    mv      a1, sp        # Pass $sp.
    mv      a0, rSELF     # pass Thread::Current
    jal     artInstrumentationMethodExitFromCode  # (Thread*, SP, gpr_res*, fpr_res*)

    beq     a0, zero, .Ldo_deliver_instrumentation_exception
                            # Deliver exception if we got nullptr as function.
    bne    a1, zero, .Ldeoptimize

    # Normal return.
    sd      a0, (FRAME_SIZE_SAVE_EVERYTHING-8)(sp)  # Set return pc.
    RESTORE_SAVE_EVERYTHING_FRAME
    jalr    zero, ra  # jic     ra, 0         # todo: lyh
.Ldo_deliver_instrumentation_exception:
    DELIVER_PENDING_EXCEPTION_FRAME_READY
.Ldeoptimize:
    sd      a1, (FRAME_SIZE_SAVE_EVERYTHING-8)(sp)
                            # Fake a call from instrumentation return pc.
    RESTORE_SAVE_EVERYTHING_FRAME
    la      t6, art_quick_deoptimize
    jalr    zero, t6
END art_quick_instrumentation_exit

    /*
     * Instrumentation has requested that we deoptimize into the interpreter. The deoptimization
     * will long jump to the upcall with a special exception of -1.
     */

    .hidden art_quick_deoptimize
    .extern artDeoptimize
ENTRY_NO_GP_CFA art_quick_deoptimize, FRAME_SIZE_SAVE_EVERYTHING
    SETUP_SAVE_EVERYTHING_FRAME
    mv      a0, rSELF      # pass Thread::current
    jal     artDeoptimize   # artDeoptimize(Thread*)
    ebreak
END art_quick_deoptimize

    /*
     * Compiled code has requested that we deoptimize into the interpreter. The deoptimization
     * will long jump to the upcall with a special exception of -1.
     */
    .extern artDeoptimizeFromCompiledCode
ENTRY_NO_GP art_quick_deoptimize_from_compiled_code
    SETUP_SAVE_EVERYTHING_FRAME
    mv       a1, rSELF                       # pass Thread::current
    jal      artDeoptimizeFromCompiledCode    # (DeoptimizationKind, Thread*)
    ebreak;
END art_quick_deoptimize_from_compiled_code


/* java.lang.String.compareTo(String anotherString) */
ENTRY_NO_GP art_quick_string_compareto
/* $a0 holds address of "this" */
/* $a1 holds address of "anotherString" */
    mv     a2, zero
    mv     a3, zero                               # return 0 (it returns a2 - a3)
    beq    a0, a1, .Lstring_compareto_length_diff # this and anotherString are the same object

#if (STRING_COMPRESSION_FEATURE)
    lw     a4, MIRROR_STRING_COUNT_OFFSET(a0)     # 'count' field of this
    lw     a5, MIRROR_STRING_COUNT_OFFSET(a1)     # 'count' field of anotherString
    sra    a2, a4, 1                              # this.length()
    sra    a3, a5, 1                              # anotherString.length()
#else
    lw     a2, MIRROR_STRING_COUNT_OFFSET(a0)     # this.length()
    lw     a3, MIRROR_STRING_COUNT_OFFSET(a1)     # anotherString.length()
#endif
    
    sltu    t4, a2, a3                            #MINu   t2, a2, a3
    bnez    t4, .Llessthan
    mv      t2, a3  
    j       .Lmorethan
.Llessthan:
    mv      t2, a2

.Lmorethan:      
    # $t2 now holds min(this.length(),anotherString.length())

    # while min(this.length(),anotherString.length())-i != 0
    beqz    t2, .Lstring_compareto_length_diff # if $t2==0
                                               #     return (this.length() - anotherString.length())

#if (STRING_COMPRESSION_FEATURE)
    # Differ cases:
    andi      a6, a4, 0x01 # dext   a6, a4, 0, 1                    #todo: lyh
    beqz      a6, .Lstring_compareto_this_is_compressed
    andi      a6, a5, 0x01 # dext   a6, a5, 0, 1                      # In branch delay slot.
    beqz     a6, .Lstring_compareto_that_is_compressed
    j      .Lstring_compareto_both_not_compressed

.Lstring_compareto_this_is_compressed:
    beqz   a6, .Lstring_compareto_both_compressed
    /* If (this->IsCompressed() && that->IsCompressed() == false) */
.Lstring_compareto_loop_comparison_this_compressed:
    lb     t0, MIRROR_STRING_VALUE_OFFSET(a0)
    lh     t1, MIRROR_STRING_VALUE_OFFSET(a1)
    bne    t0, t1, .Lstring_compareto_char_diff
    addi   a0, a0, 1      # point at this.charAt(i++) - compressed
    addi   t2, t2, -1      # new value of min(this.length(),anotherString.length())-i todo:lyh
    addi   a1, a1, 2      # point at anotherString.charAt(i++) - uncompressed
    bnez   t2, .Lstring_compareto_loop_comparison_this_compressed    
    sub    a0, a2, a3    # return (this.length() - anotherString.length())
    jalr   zero, ra
    
.Lstring_compareto_that_is_compressed:
    lh     t0, MIRROR_STRING_VALUE_OFFSET(a0)
    lb     t1, MIRROR_STRING_VALUE_OFFSET(a1)
    bne    t0, t1, .Lstring_compareto_char_diff
    addi   a0, a0, 2      # point at this.charAt(i++) - uncompressed
    addi   t2, t2, -1      # new value of min(this.length(),anotherString.length())-i  todo:lyh
    addi   a1, a1, 1      # point at anotherString.charAt(i++) - compressed
    bnez   t2, .Lstring_compareto_that_is_compressed
    sub    a0, a2, a3    # return (this.length() - anotherString.length())
    jalr   zero, ra    

.Lstring_compareto_both_compressed:
    lb     t0, MIRROR_STRING_VALUE_OFFSET(a0)
    lb     t1, MIRROR_STRING_VALUE_OFFSET(a1)
    bne    t0, t1, .Lstring_compareto_char_diff
    addi   a0, a0, 1      # point at this.charAt(i++) - compressed
    addi   t2, t2, -1      # new value of min(this.length(),anotherString.length())-i  todo:lyh
    addi   a1, a1, 1      # point at anotherString.charAt(i++) - compressed
    bnez   t2, .Lstring_compareto_both_compressed
      
    sub   a0, a2, a3    # return (this.length() - anotherString.length())
    jalr   zero, ra
#endif

.Lstring_compareto_both_not_compressed:
    lh     t0, MIRROR_STRING_VALUE_OFFSET(a0)    # while this.charAt(i) == anotherString.charAt(i)
    lh     t1, MIRROR_STRING_VALUE_OFFSET(a1)
    bne    t0, t1, .Lstring_compareto_char_diff  # if this.charAt(i) != anotherString.charAt(i)
                            #     return (this.charAt(i) - anotherString.charAt(i))
    addi   a0, a0, 2      # point at this.charAt(i++)
    addi   t2, t2, -1      # new value of min(this.length(),anotherString.length())-i  todo:lyh
    addi   a1, a1, 2      # point at anotherString.charAt(i++)
    bnez   t2, .Lstring_compareto_both_not_compressed
    
.Lstring_compareto_length_diff:
    sub    a0, a2, a3    # return (this.length() - anotherString.length())
    jalr   zero, ra    

.Lstring_compareto_char_diff:
    sub    a0, t0, t1    # return (this.charAt(i) - anotherString.charAt(i))
    jalr   zero, ra
   
END art_quick_string_compareto

/* java.lang.String.indexOf(int ch, int fromIndex=0) */
ENTRY_NO_GP art_quick_indexof
/* $a0 holds address of "this" */
/* $a1 holds "ch" */
/* $a2 holds "fromIndex" */
#if (STRING_COMPRESSION_FEATURE)
    lw     a3, MIRROR_STRING_COUNT_OFFSET(a0)     # 'count' field of this
#else
    lw     t0, MIRROR_STRING_COUNT_OFFSET(a0)     # this.length()
#endif
    slt    t4, a2, zero      # if fromIndex < 0
    bnez   t4, .Lneqz        # seleqz $a2, $a2, $at    # fromIndex = 0;
    j      .Leqz
.Lneqz:
    li     a2, 0

.Leqz:
#if (STRING_COMPRESSION_FEATURE)
    srl    t0, a3, 1        # $a3 holds count (with flag) and $t0 holds actual length
#endif
    sub    t0,  t0, a2      # this.length() - fromIndex
    li     t5,  -1          #     return -1;
    blez   t0,  6f          # if this.length()-fromIndex <= 0

#if (STRING_COMPRESSION_FEATURE)
    andi   a3, a3, 1       # dext   $a3, $a3, 0, 1   # Extract compression flag.  todo: lyh
    beqz   a3, .Lstring_indexof_compressed
#endif

    sll    t5, a2, 1      # $a0 += $a2 * 2
    add    a0, a0, t5     #  "  ditto  "
    mv     t5, a2         # Set i to fromIndex.

1:
    lhu    t3, MIRROR_STRING_VALUE_OFFSET(a0)     # if this.charAt(i) == ch
    beq    t3, a1, 6f                             #     return i;
    addi   a0, a0, 2        # i++    
    addi   t0, t0, -1       # this.length() - i  
    addi   t5, t5, 1        # i++
    bnez   t0, 1b           # while this.length() - i > 0

    li     t5, -1           # if this.length() - i <= 0
                            #     return -1;

6:
    mv      a0,   t5
    jr      ra


#if (STRING_COMPRESSION_FEATURE)
.Lstring_indexof_compressed:
    mv   a4, a0         # Save a copy in $a4 to later compute result.
    add  a0, a0, a2     # $a0 += $a2

.Lstring_indexof_compressed_loop:
    lbu    t3, MIRROR_STRING_VALUE_OFFSET(a0)
    beq    t3, a1, .Lstring_indexof_compressed_matched
    addi   t0, t0, -1
    addi   a0, a0, 1            
    bgtz   t0, .Lstring_indexof_compressed_loop  

.Lstring_indexof_nomatch:
    li     a0, -1          # return -1;
    jalr   zero, ra

.Lstring_indexof_compressed_matched:
    sub    a0, a0, a4    # return (current - start);
    jalr   zero, ra
   
#endif
END art_quick_indexof

    /*
     * Create a function `name` calling the ReadBarrier::Mark routine,
     * getting its argument and returning its result through register
     * `reg`, saving and restoring all caller-save registers.
     */
.macro READ_BARRIER_MARK_REG name, reg
ENTRY \name
    // Null check so that we can load the lock word.
    bne   \reg, zero, .Lnot_null_\name
.Lret_rb_\name:
    jalr    zero, ra   # return
.Lnot_null_\name:
    // Check lock word for mark bit, if marked return.
    lw      t6, MIRROR_OBJECT_LOCK_WORD_OFFSET(\reg)

    slliw     t5, t6, 31 - LOCK_WORD_MARK_BIT_SHIFT     # Move mark bit to sign bit.
    bltz    t5, .Lret_rb_\name
#if (LOCK_WORD_STATE_SHIFT != 30) || (LOCK_WORD_STATE_FORWARDING_ADDRESS != 3)
    // The below code depends on the lock word state being in the highest bits
    // and the "forwarding address" state having all bits set.
#error "Unexpected lock word state shift or forwarding address state value."
#endif
    // Test that both the forwarding state bits are 1.
    slliw     t5, t6, 1
    and     t5, t5, t6                               # Sign bit = 1 IFF both bits are 1.
    bltz    t5, .Lret_forwarding_address\name
    # .set pop

    addi  sp, sp, -280
    .cfi_adjust_cfa_offset 280

    sd      ra, 272(sp)
    .cfi_rel_offset ra, 272
    sd      t5, 264(sp)
    .cfi_rel_offset t5, 264
    sd      t4, 256(sp)
    .cfi_rel_offset t4, 256
    sd      t3, 248(sp)
    .cfi_rel_offset t3, 248
    sd      t2, 240(sp)
    .cfi_rel_offset t2, 240
    sd      t1, 232(sp)
    .cfi_rel_offset t1, 232
    sd      t0, 224(sp)
    .cfi_rel_offset t0, 224
    sd      a7, 216(sp)
    .cfi_rel_offset a7, 216
    sd      a6, 208(sp)
    .cfi_rel_offset a6, 208
    sd      a5, 200(sp)
    .cfi_rel_offset a5, 200
    sd      a4, 192(sp)
    .cfi_rel_offset a4, 192
    sd      a3, 184(sp)
    .cfi_rel_offset a3, 184
    sd      a2, 176(sp)
    .cfi_rel_offset a2, 176
    sd      a1, 168(sp)
    .cfi_rel_offset a1, 168
    sd      a0, 160(sp)
    .cfi_rel_offset a0, 160

    la     t6, artReadBarrierMark   #todo：lyh

    fsd    f31, 152(sp)
    fsd    f30, 144(sp)
    fsd    f29, 136(sp)
    fsd    f28, 128(sp)
    fsd    f17, 120(sp)
    fsd    f16, 112(sp)
    fsd    f15, 104(sp)
    fsd    f14,  96(sp)
    fsd    f13,  88(sp)
    fsd    f12,  80(sp)
    fsd    f11,  72(sp)
    fsd    f10,  64(sp)
    fsd    f7,   56(sp)
    fsd    f6,   48(sp)
    fsd    f5,   40(sp)
    fsd    f4,   32(sp)
    fsd    f3,   24(sp)
    fsd    f2,   16(sp)
    fsd    f1,   8(sp)
    fsd    f0,   0(sp)

    .ifnc \reg, a0
      mv  a0, \reg           # pass obj from `reg` in a0
    .endif
    jalr    t6                 # v0 <- artReadBarrierMark(obj)

    ld      ra, 272(sp)
    .cfi_restore ra
    ld      t5, 264(sp)
    .cfi_restore t5
    ld      t4, 256(sp)
    .cfi_restore t4
    ld      t3, 248(sp)
    .cfi_restore t3
    ld      t2, 240(sp)
    .cfi_restore t2
    ld      t1, 232(sp)
    .cfi_restore t1
    ld      t0, 224(sp)
    .cfi_restore t0
    ld      a7, 216(sp)
    .cfi_restore a7
    ld      a6, 208(sp)
    .cfi_restore a6
    ld      a5, 200(sp)
    .cfi_restore a5
    ld      a4, 192(sp)
    .cfi_restore a4
    ld      a3, 184(sp)
    .cfi_restore a3
    ld      a2, 176(sp)
    .cfi_restore a2
    ld      a1, 168(sp)
    .cfi_restore a1

    .ifnc \reg, a0
      mv  \reg, a0           # `reg` <- a0
      ld    a0, 160(sp)
      .cfi_restore a0
    .endif

    fld    f31, 152(sp)
    fld    f30, 144(sp)
    fld    f29, 136(sp)
    fld    f28, 128(sp)
    fld    f17, 120(sp)
    fld    f16, 112(sp)
    fld    f15, 104(sp)
    fld    f14,  96(sp)
    fld    f13,  88(sp)
    fld    f12,  80(sp)
    fld    f11,   72(sp)
    fld    f10,   64(sp)
    fld    f7,   56(sp)
    fld    f6,   48(sp)
    fld    f5,   40(sp)
    fld    f4,   32(sp)
    fld    f3,   24(sp)
    fld    f2,   16(sp)
    fld    f1,   8(sp)
    fld    f0,   0(sp)

    addi    sp, sp, 280
    .cfi_adjust_cfa_offset -280
    jalr    zero, ra

.Lret_forwarding_address\name:
    // Shift left by the forwarding address shift. This clears out the state bits since they are
    // in the top 2 bits of the lock word.
    slliw     \reg, t6, LOCK_WORD_STATE_FORWARDING_ADDRESS_SHIFT
    slli     \reg, \reg, 32
    srli     \reg, \reg, 32  # Get address from \reg[31:0] and Make sure the address is zero-extended.
    jalr    zero, ra
END \name
.endm

// Note that art_quick_read_barrier_mark_regXX corresponds to register XX+1.
// ZERO, RA, SP, GP (registers 0 - 3) is reserved.
// TP/TR (register 4) is  reserved as thread register
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg01, t0
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg02, t1
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg03, t2
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg04, s0
// S1 (register 9) is reserved as thread register.
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg06, a0
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg07, a1
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg08, a2
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg09, a3
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg10, a4
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg11, a5
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg12, a6
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg13, a7
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg14, s2
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg15, s3
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg16, s4
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg17, s5
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg18, s6
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg19, s7
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg20, s8
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg21, s9
READ_BARRIER_MARK_REG art_quick_read_barrier_mark_reg22, s10
// S11 (register 27) is reserved as suspended register.
// T3, T4, T5, t6(registers 28 - 31) are reserved as temporary/scratch registers.


// Caller code:
// Short constant offset/index:
//  ld      $t6, pReadBarrierMarkReg00
//  beqzc   $t6, skip_call
//  jialc   $t6, thunk_disp
// skip_call:
//  lwu     `out`, ofs(`obj`)
// [dsubu   `out`, $zero, `out`
//  dext    `out`, `out`, 0, 32]  # Unpoison reference.
.macro BRB_FIELD_SHORT_OFFSET_ENTRY obj
    # Explicit null check. May be redundant (for array elements or when the field
    # offset is larger than the page size, 4KB).
    # $ra will be adjusted to point to lwu's stack map when throwing NPE.
    beq   \obj, zero, .Lintrospection_throw_npe
    la    t3, .Lintrospection_exits                  # $t3 = address of .Lintrospection_exits.  todo：lyh

    lw      t4, MIRROR_OBJECT_LOCK_WORD_OFFSET(\obj)
    slliw   t4, t4, 31 - LOCK_WORD_READ_BARRIER_STATE_SHIFT   # Move barrier state bit
                                                              # to sign bit.
    mv      t5, \obj                                   # Move `obj` to $t5 for common code.
    bltz    t4, .Lintrospection_field_array            # If gray, load reference, mark.

    fence
    jalr    zero, ra                                  # Otherwise, load-load barrier and return.
    ebreak                                               # Padding to 10 instructions.
    nop
    nop
.endm

// Caller code:
// Long constant offset/index:   | Variable index:
//  ld      $t6, pReadBarrierMarkReg00
//  beqz    $t6, skip_call       |  beqz    $t6, skip_call
//  daui    $t5, `obj`, ofs_hi   |  dlsa    $t5, `index`, `obj`, 2
//  jialc   $t6, thunk_disp      |  jialc   $t6, thunk_disp
// skip_call:                    | skip_call:
//  lwu     `out`, ofs_lo($t5)   |  lwu     `out`, ofs($t5)
// [dsubu   `out`, $zero, `out`  | [dsubu   `out`, $zero, `out`
//  dext    `out`, `out`, 0, 32] |  dext    `out`, `out`, 0, 32]  # Unpoison reference.
.macro BRB_FIELD_LONG_OFFSET_ENTRY obj
    # No explicit null check for variable indices or large constant indices/offsets
    # as it must have been done earlier.
    la    t3, .Lintrospection_exits                  # $t3 = address of .Lintrospection_exits. todo:lyh

    lw      t4, MIRROR_OBJECT_LOCK_WORD_OFFSET(\obj)
    slliw   t4, t4, 31 - LOCK_WORD_READ_BARRIER_STATE_SHIFT   # Move barrier state bit
                                                                # to sign bit.
    bltz   t4, .Lintrospection_field_array            # If gray, load reference, mark.

    fence
    jalr    zero, ra                                  # Otherwise, load-load barrier and return.

    ebreak                                               # Padding to 10 instructions.
    ebreak
    nop
    nop
    nop
    nop
.endm

.macro BRB_GC_ROOT_ENTRY root
    la    t3, .Lintrospection_exit_\root             # $t3 = exit point address.
    mv    t5, \root                                  # Move reference to $t5 for common code.
    bne   \root, zero, .Lintrospection_common
    nop
    jalr  zero, ra                                  # Return if null.
    nop                                               # padding to 6 bytes
    nop
.endm

.macro BRB_FIELD_EXIT out
.Lintrospection_exit_\out:
    mv      \out, t5                                   # Return reference in expected register.
    jalr    zero, ra
.endm

.macro BRB_FIELD_EXIT_BREAK
    ebreak
    ebreak
.endm

ENTRY_NO_GP art_quick_read_barrier_mark_introspection
    # Entry points for offsets/indices not fitting into int16_t and for variable indices.
    BRB_FIELD_LONG_OFFSET_ENTRY t0
    BRB_FIELD_LONG_OFFSET_ENTRY t1
    BRB_FIELD_LONG_OFFSET_ENTRY t2
    BRB_FIELD_LONG_OFFSET_ENTRY s0
    BRB_FIELD_LONG_OFFSET_ENTRY a0
    BRB_FIELD_LONG_OFFSET_ENTRY a1
    BRB_FIELD_LONG_OFFSET_ENTRY a2
    BRB_FIELD_LONG_OFFSET_ENTRY a3
    BRB_FIELD_LONG_OFFSET_ENTRY a4
    BRB_FIELD_LONG_OFFSET_ENTRY a5
    BRB_FIELD_LONG_OFFSET_ENTRY a6
    BRB_FIELD_LONG_OFFSET_ENTRY a7
    BRB_FIELD_LONG_OFFSET_ENTRY s2
    BRB_FIELD_LONG_OFFSET_ENTRY s3
    BRB_FIELD_LONG_OFFSET_ENTRY s4
    BRB_FIELD_LONG_OFFSET_ENTRY s5
    BRB_FIELD_LONG_OFFSET_ENTRY s6
    BRB_FIELD_LONG_OFFSET_ENTRY s7
    BRB_FIELD_LONG_OFFSET_ENTRY s8
    BRB_FIELD_LONG_OFFSET_ENTRY s9
    BRB_FIELD_LONG_OFFSET_ENTRY s10

    # Entry points for offsets/indices fitting into int16_t.
    BRB_FIELD_SHORT_OFFSET_ENTRY t0
    BRB_FIELD_SHORT_OFFSET_ENTRY t1
    BRB_FIELD_SHORT_OFFSET_ENTRY t2
    BRB_FIELD_SHORT_OFFSET_ENTRY s0
    BRB_FIELD_SHORT_OFFSET_ENTRY a0
    BRB_FIELD_SHORT_OFFSET_ENTRY a1
    BRB_FIELD_SHORT_OFFSET_ENTRY a2
    BRB_FIELD_SHORT_OFFSET_ENTRY a3
    BRB_FIELD_SHORT_OFFSET_ENTRY a4
    BRB_FIELD_SHORT_OFFSET_ENTRY a5
    BRB_FIELD_SHORT_OFFSET_ENTRY a6
    BRB_FIELD_SHORT_OFFSET_ENTRY a7
    BRB_FIELD_SHORT_OFFSET_ENTRY s2
    BRB_FIELD_SHORT_OFFSET_ENTRY s3
    BRB_FIELD_SHORT_OFFSET_ENTRY s4
    BRB_FIELD_SHORT_OFFSET_ENTRY s5
    BRB_FIELD_SHORT_OFFSET_ENTRY s6
    BRB_FIELD_SHORT_OFFSET_ENTRY s7
    BRB_FIELD_SHORT_OFFSET_ENTRY s8
    BRB_FIELD_SHORT_OFFSET_ENTRY s9
    BRB_FIELD_SHORT_OFFSET_ENTRY s10

    .global art_quick_read_barrier_mark_introspection_gc_roots
art_quick_read_barrier_mark_introspection_gc_roots:
    # Entry points for GC roots.
    BRB_GC_ROOT_ENTRY t0
    BRB_GC_ROOT_ENTRY t1
    BRB_GC_ROOT_ENTRY t2
    BRB_GC_ROOT_ENTRY s0
    BRB_GC_ROOT_ENTRY a0
    BRB_GC_ROOT_ENTRY a1
    BRB_GC_ROOT_ENTRY a2
    BRB_GC_ROOT_ENTRY a3
    BRB_GC_ROOT_ENTRY a4
    BRB_GC_ROOT_ENTRY a5
    BRB_GC_ROOT_ENTRY a6
    BRB_GC_ROOT_ENTRY a7
    BRB_GC_ROOT_ENTRY s2
    BRB_GC_ROOT_ENTRY s3
    BRB_GC_ROOT_ENTRY s4
    BRB_GC_ROOT_ENTRY s5
    BRB_GC_ROOT_ENTRY s6
    BRB_GC_ROOT_ENTRY s7     
    BRB_GC_ROOT_ENTRY s8
    BRB_GC_ROOT_ENTRY s9
    BRB_GC_ROOT_ENTRY s10

    .global art_quick_read_barrier_mark_introspection_end_of_entries
art_quick_read_barrier_mark_introspection_end_of_entries:

.Lintrospection_throw_npe:
    addi    ra, ra, 4         # Skip lwu, make $ra point to lwu's stack map.

    la      t6, art_quick_throw_null_pointer_exception
    jr      t6

    // Fields and array elements.

.Lintrospection_field_array:
    // Get the field/element address using $t5 and the offset from the lwu instruction.
    lh      t4, 0(ra)         # $ra points to lwu: $at = low 16 bits of field/element offset.
    addi    ra, ra, 4 + HEAP_POISON_INSTR_SIZE   # Skip lwu(+dsubu+dext).
    add     t5, t5, t4       # $t5 = field/element address.

    // Calculate the address of the exit point, store it in $t3 and load the reference into $t5.
    lb      t4, (-HEAP_POISON_INSTR_SIZE - 2)(ra)   # $ra-HEAP_POISON_INSTR_SIZE-4 points to
                                                      # "lwu `out`, ...".
    andi    t4, t4, 31        # Extract `out` from lwu.

    lw      t5, 0(t5)         # $t5 = reference.
    UNPOISON_HEAP_REF t5

    // Return if null reference.
    slli   t4, t4, 3          # $t3 = address of the exit point
    add    t3, t3, t4         # (BRB_FIELD_EXIT* macro is 8 bytes).

    bne    t5, zero, .Lintrospection_common

    // Early return through the exit point.
.Lintrospection_return_early:
    jalr    zero, t3          # Move $t5 to `out` and return.

    // Code common for GC roots, fields and array elements.

.Lintrospection_common:
    // Check lock word for mark bit, if marked return.
    lw      t6, MIRROR_OBJECT_LOCK_WORD_OFFSET(t5)
    slliw   t4, t6, 31 - LOCK_WORD_MARK_BIT_SHIFT     # Move mark bit to sign bit.
    bltz    t4, .Lintrospection_return_early
#if (LOCK_WORD_STATE_SHIFT != 30) || (LOCK_WORD_STATE_FORWARDING_ADDRESS != 3)
    // The below code depends on the lock word state being in the highest bits
    // and the "forwarding address" state having all bits set.
#error "Unexpected lock word state shift or forwarding address state value."
#endif
    // Test that both the forwarding state bits are 1.
    slliw   t4, t6, 1
    and     t4, t4, t6                               # Sign bit = 1 IFF both bits are 1.
    bge     t4, zero, .Lintrospection_mark

    # .set pop

    // Shift left by the forwarding address shift. This clears out the state bits since they are
    // in the top 2 bits of the lock word.
    slliw     t5, t6, LOCK_WORD_STATE_FORWARDING_ADDRESS_SHIFT
    slli      t5, t5,  32     # Make sure the address is zero-extended.
    srli      t5, t5,  32
    jalr    zero, t3          # Move $t5 to `out` and return.

.Lintrospection_mark:
    // Partially set up the stack frame preserving only $ra.
    addi  sp, sp, -280
    .cfi_adjust_cfa_offset 280
    sd      ra, 272(sp)
    .cfi_rel_offset ra, 272

    // Finalize the stack frame and call.
    sd      t5, 264(sp)
    .cfi_rel_offset t5, 264
    sd      t4, 256(sp)
    .cfi_rel_offset t4, 256
    sd      t3, 248(sp)             # Preserve the exit point address.
    .cfi_rel_offset t3, 248
    sd      t2, 240(sp)
    .cfi_rel_offset t2, 240
    sd      t1, 232(sp)
    .cfi_rel_offset t1, 232
    sd      t0, 224(sp)
    .cfi_rel_offset t0, 224
    sd      a7, 216(sp)
    .cfi_rel_offset a7, 216
    sd      a6, 208(sp)
    .cfi_rel_offset a6, 208
    sd      a5, 200(sp)
    .cfi_rel_offset a5, 200
    sd      a4, 192(sp)
    .cfi_rel_offset a4, 192
    sd      a3, 184(sp)
    .cfi_rel_offset a3, 184
    sd      a2, 176(sp)
    .cfi_rel_offset a2, 176
    sd      a1, 168(sp)
    .cfi_rel_offset a1, 168
    sd      a0, 160(sp)
    .cfi_rel_offset a0, 160

    la     t6, artReadBarrierMark

    fsd    f31, 152(sp)
    fsd    f30, 144(sp)
    fsd    f29, 136(sp)
    fsd    f28, 128(sp)
    fsd    f17, 120(sp)
    fsd    f16, 112(sp)
    fsd    f15, 104(sp)
    fsd    f14,  96(sp)
    fsd    f13,  88(sp)
    fsd    f12,  80(sp)
    fsd    f11,  72(sp)
    fsd    f10,  64(sp)
    fsd    f7,   56(sp)
    fsd    f6,   48(sp)
    fsd    f5,   40(sp)
    fsd    f4,   32(sp)
    fsd    f3,   24(sp)
    fsd    f2,   16(sp)
    fsd    f1,   8(sp)
    fsd    f0,   0(sp)

    mv    a0, t5            # Pass reference in $a0.
    jalr    t6                 # $v0 <- artReadBarrierMark(reference)

    ld      ra, 272(sp)
    .cfi_restore ra
    ld      t5, 264(sp)
    .cfi_restore t5
    ld      t4, 256(sp)
    .cfi_restore t4
    ld      t3, 248(sp)
    .cfi_restore t3
    ld      t2, 240(sp)
    .cfi_restore t2
    ld      t1, 232(sp)
    .cfi_restore t1
    ld      t0, 224(sp)
    .cfi_restore t0
    ld      a7, 216(sp)
    .cfi_restore a7
    ld      a6, 208(sp)
    .cfi_restore a6
    ld      a5, 200(sp)
    .cfi_restore a5
    ld      a4, 192(sp)
    .cfi_restore a4
    ld      a3, 184(sp)
    .cfi_restore a3
    ld      a2, 176(sp)
    .cfi_restore a2
    ld      a1, 168(sp)
    .cfi_restore a1

    mv    t5, a0
    ld    a0, 160(sp)
    .cfi_restore a0


    fld    f31, 152(sp)
    fld    f30, 144(sp)
    fld    f29, 136(sp)
    fld    f28, 128(sp)
    fld    f17, 120(sp)
    fld    f16, 112(sp)
    fld    f15, 104(sp)
    fld    f14,  96(sp)
    fld    f13,  88(sp)
    fld    f12,  80(sp)
    fld    f11,   72(sp)
    fld    f10,   64(sp)
    fld    f7,   56(sp)
    fld    f6,   48(sp)
    fld    f5,   40(sp)
    fld    f4,   32(sp)
    fld    f3,   24(sp)
    fld    f2,   16(sp)
    fld    f1,   8(sp)
    fld    f0,   0(sp)

    // Return through the exit point.
    addi    sp, sp, 280
    .cfi_adjust_cfa_offset -280
    jalr    zero, t3          # Move $t5 to `out` and return.

.Lintrospection_exits:
    BRB_FIELD_EXIT_BREAK 
    BRB_FIELD_EXIT_BREAK 
    BRB_FIELD_EXIT_BREAK 
    BRB_FIELD_EXIT_BREAK 
    BRB_FIELD_EXIT_BREAK 
    BRB_FIELD_EXIT t0
    BRB_FIELD_EXIT t1
    BRB_FIELD_EXIT t2
    BRB_FIELD_EXIT s0
    BRB_FIELD_EXIT_BREAK
    BRB_FIELD_EXIT a0
    BRB_FIELD_EXIT a1
    BRB_FIELD_EXIT a2
    BRB_FIELD_EXIT a3
    BRB_FIELD_EXIT a4
    BRB_FIELD_EXIT a5
    BRB_FIELD_EXIT a6
    BRB_FIELD_EXIT a7
    BRB_FIELD_EXIT s2
    BRB_FIELD_EXIT s3
    BRB_FIELD_EXIT s4
    BRB_FIELD_EXIT s5
    BRB_FIELD_EXIT s6
    BRB_FIELD_EXIT s7
    BRB_FIELD_EXIT s8
    BRB_FIELD_EXIT s9
    BRB_FIELD_EXIT s10
    BRB_FIELD_EXIT_BREAK
    BRB_FIELD_EXIT_BREAK
    BRB_FIELD_EXIT_BREAK
    BRB_FIELD_EXIT_BREAK
    BRB_FIELD_EXIT_BREAK
END art_quick_read_barrier_mark_introspection

   /*
     * Polymorphic method invocation.
     * On entry:
     *   a0 = unused
     *   a1 = receiver
     */
.extern artInvokePolymorphic
ENTRY art_quick_invoke_polymorphic
    SETUP_SAVE_REFS_AND_ARGS_FRAME
    mv      a0, a1               # Make $a0 the receiver
    mv      a1, rSELF             # Make $a1 an alias for the current Thread.
    mv      a2, sp               # Make $a3 a pointer to the saved frame context.
    jal     artInvokePolymorphic   # artInvokePolymorphic(receiver, Thread*, context)
    ld      t0, THREAD_EXCEPTION_OFFSET(rSELF) # load Thread::Current()->exception_
    addi    sp, sp, REFS_AND_ARGS_MINUS_REFS_SIZE  # skip a0-a7 and f12-f19
    RESTORE_SAVE_REFS_ONLY_FRAME
    fmv.d.x f10, a0                 # place return value to FP return value
    bne     t0, zero, 1f
    jalr      zero, ra
1:
    DELIVER_PENDING_EXCEPTION
END art_quick_invoke_polymorphic

    /*
     * InvokeCustom invocation.
     * On entry:
     *   a0 = call_site_idx
     */
.extern artInvokeCustom
ENTRY art_quick_invoke_custom
    SETUP_SAVE_REFS_AND_ARGS_FRAME
    mv      a1, rSELF             # Make $a1 an alias for the current Thread.
    mv      a2, sp
    jal     artInvokeCustom        # Call artInvokeCustom(call_site_idx, Thread*, context).
    ld      t0, THREAD_EXCEPTION_OFFSET(rSELF) # load Thread::Current()->exception_
    addi    sp, sp, REFS_AND_ARGS_MINUS_REFS_SIZE  # skip a0-a7 and f12-f19
    RESTORE_SAVE_REFS_ONLY_FRAME
    fmv.d.x f10, a0               # place return value to FP return value
    bne     t0, zero, 1f
    jalr     zero, ra
1:
    DELIVER_PENDING_EXCEPTION
END art_quick_invoke_custom
  # .set pop                        # todo：lyh

// Wrap ExecuteSwitchImpl in assembly method which specifies DEX PC for unwinding.
//  Argument 0: a0: The context pointer for ExecuteSwitchImpl.
//  Argument 1: a1: Pointer to the templated ExecuteSwitchImpl to call.
//  Argument 2: a2: The value of DEX PC (memory address of the methods bytecode).
ENTRY ExecuteSwitchImplAsm
    addi   sp, sp, -16         # save ra and s7
    .cfi_adjust_cfa_offset  16
    sd     ra, 8(sp)
    .cfi_rel_offset ra, 8
    sd     s7, 0(sp)
    .cfi_rel_offset s7, 0

    mv s7, a2                  # s7 = DEX PC
    CFI_DEFINE_DEX_PC_WITH_OFFSET(10 /* a0 */, 23 /* s7 */, 0)
    jalr ra, a1                     # Call the wrapped method.

    ld     s7, 0(sp)     # restore ra and s7
    .cfi_restore s7
    ld     ra, 8(sp)
    .cfi_restore ra
    addi   sp, sp, 16
    .cfi_adjust_cfa_offset -16

    jalr   zero, ra
END ExecuteSwitchImplAsm

    .extern artStringBuilderAppend
ENTRY art_quick_string_builder_append
  SETUP_SAVE_REFS_ONLY_FRAME          // save callee saves in case of GC
  addi    a1, sp, (FRAME_SIZE_SAVE_REFS_ONLY + __SIZEOF_POINTER__)  // pass args
  move    a2, rSELF                    // pass Thread::Current            
  la    t6, artStringBuilderAppend     // (uint32_t, const unit32_t*, Thread*)
  jalr  ra, t6

  RETURN_IF_RESULT_IS_NON_ZERO_OR_DELIVER
END art_quick_string_builder_append

// x0 contains the class, x8 contains the inline cache. x9-x15 can be used.
ENTRY art_quick_update_inline_cache
  ebreak;
  // TBD: Need implement it for Android 12
END art_quick_update_inline_cache

// On entry, method is at the bottom of the stack.
ENTRY art_quick_compile_optimized
  SETUP_SAVE_EVERYTHING_FRAME
  ld   a0, FRAME_SIZE_SAVE_EVERYTHING(sp) // pass ArtMethod
  move a1, rSELF                         // pass Thread::Current           
  la    t6, artCompileOptimized          // (ArtMethod*, Thread*)
  jalr  ra, t6

  RESTORE_SAVE_EVERYTHING_FRAME

  jalr   zero, ra
END art_quick_compile_optimized
