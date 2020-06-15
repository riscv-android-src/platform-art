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

    .balign 16

    /*
     * Jni dlsym lookup stub.
     */
    .extern artFindNativeMethod
ENTRY art_jni_dlsym_lookup_stub
    addi sp, sp, -80        # save a0-a7 and $ra
    .cfi_adjust_cfa_offset 80
    sd     ra, 64(sp)
    .cfi_rel_offset ra, 64
    sd     a7, 56(sp)
    .cfi_rel_offset a7, 56
    sd     a6, 48(sp)
    .cfi_rel_offset a6, 48
    sd     a5, 40(sp)
    .cfi_rel_offset a5, 40
    sd     a4, 32(sp)
    .cfi_rel_offset a4, 32
    sd     a3, 24(sp)
    .cfi_rel_offset a3, 24
    sd     a2, 16(sp)
    .cfi_rel_offset a2, 16
    sd     a1, 8(sp)
    .cfi_rel_offset a1, 8
    sd     a0, 0(sp)
    .cfi_rel_offset a0, 0
    move   a0, s1             # pass Thread::Current()

    la t6, artFindNativeMethod  # (Thread*)
    jalr ra, t6
    move t6, a0;               # put method code result in $t9 (t6 in riscv64)
    ld     a0, 0(sp)          # restore registers from stack
    .cfi_restore a0
    ld     a1, 8(sp)
    .cfi_restore a1
    ld     a2, 16(sp)
    .cfi_restore a2
    ld     a3, 24(sp)
    .cfi_restore a3
    ld     a4, 32(sp)
    .cfi_restore a4
    ld     a5, 40(sp)
    .cfi_restore a4
    ld     a6, 48(sp)
    .cfi_restore a6
    ld     a7, 56(sp)
    .cfi_restore a7
    ld     ra, 64(sp)
    .cfi_restore ra
    addi sp, sp, 80         # restore the stack
    .cfi_adjust_cfa_offset -80

    beq    t6, zero, .Lno_native_code_found
    nop
    jalr   zero, t6           # leaf call to method's code
    nop
.Lno_native_code_found:
    jalr   zero, ra
    nop
END art_jni_dlsym_lookup_stub
