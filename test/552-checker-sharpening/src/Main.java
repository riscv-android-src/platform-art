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

public class Main {

  public static void assertIntEquals(int expected, int result) {
    if (expected != result) {
      throw new Error("Expected: " + expected + ", found: " + result);
    }
  }

  public static void assertStringEquals(String expected, String result) {
    if (expected != null ? !expected.equals(result) : result != null) {
      throw new Error("Expected: " + expected + ", found: " + result);
    }
  }

  public static void assertClassEquals(Class<?> expected, Class<?> result) {
    if (expected != result) {
      throw new Error("Expected: " + expected + ", found: " + result);
    }
  }

  public static boolean doThrow = false;

  private static int $noinline$foo(int x) {
    if (doThrow) { throw new Error(); }
    return x;
  }

  /// CHECK-START: int Main.testSimple(int) sharpening (before)
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_via_method

  /// CHECK-START-ARM: int Main.testSimple(int) sharpening (after)
  /// CHECK-NOT:            ArmDexCacheArraysBase
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-ARM64: int Main.testSimple(int) sharpening (after)
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-MIPS: int Main.testSimple(int) sharpening (after)
  /// CHECK-NOT:            MipsDexCacheArraysBase
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-X86: int Main.testSimple(int) sharpening (after)
  /// CHECK-NOT:            X86ComputeBaseMethodAddress
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-X86_64: int Main.testSimple(int) sharpening (after)
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-ARM: int Main.testSimple(int) dex_cache_array_fixups_arm (after)
  /// CHECK:                ArmDexCacheArraysBase
  /// CHECK-NOT:            ArmDexCacheArraysBase

  /// CHECK-START-MIPS: int Main.testSimple(int) dex_cache_array_fixups_mips (after)
  /// CHECK:                MipsDexCacheArraysBase
  /// CHECK-NOT:            MipsDexCacheArraysBase

  /// CHECK-START-X86: int Main.testSimple(int) pc_relative_fixups_x86 (after)
  /// CHECK:                X86ComputeBaseMethodAddress
  /// CHECK-NOT:            X86ComputeBaseMethodAddress

  public static int testSimple(int x) {
    // This call should use PC-relative dex cache array load to retrieve the target method.
    return $noinline$foo(x);
  }

  /// CHECK-START: int Main.testDiamond(boolean, int) sharpening (before)
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_via_method

  /// CHECK-START-ARM: int Main.testDiamond(boolean, int) sharpening (after)
  /// CHECK-NOT:            ArmDexCacheArraysBase
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-ARM64: int Main.testDiamond(boolean, int) sharpening (after)
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-MIPS: int Main.testDiamond(boolean, int) sharpening (after)
  /// CHECK-NOT:            MipsDexCacheArraysBase
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-X86: int Main.testDiamond(boolean, int) sharpening (after)
  /// CHECK-NOT:            X86ComputeBaseMethodAddress
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-X86_64: int Main.testDiamond(boolean, int) sharpening (after)
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-ARM: int Main.testDiamond(boolean, int) dex_cache_array_fixups_arm (after)
  /// CHECK:                ArmDexCacheArraysBase
  /// CHECK-NOT:            ArmDexCacheArraysBase

  /// CHECK-START-ARM: int Main.testDiamond(boolean, int) dex_cache_array_fixups_arm (after)
  /// CHECK:                ArmDexCacheArraysBase
  /// CHECK-NEXT:           If

  /// CHECK-START-MIPS: int Main.testDiamond(boolean, int) dex_cache_array_fixups_mips (after)
  /// CHECK:                MipsDexCacheArraysBase
  /// CHECK-NOT:            MipsDexCacheArraysBase

  /// CHECK-START-MIPS: int Main.testDiamond(boolean, int) dex_cache_array_fixups_mips (after)
  /// CHECK:                MipsDexCacheArraysBase
  /// CHECK-NEXT:           If

  /// CHECK-START-X86: int Main.testDiamond(boolean, int) pc_relative_fixups_x86 (after)
  /// CHECK:                X86ComputeBaseMethodAddress
  /// CHECK-NOT:            X86ComputeBaseMethodAddress

  /// CHECK-START-X86: int Main.testDiamond(boolean, int) pc_relative_fixups_x86 (after)
  /// CHECK:                X86ComputeBaseMethodAddress
  /// CHECK-NEXT:           If

  public static int testDiamond(boolean negate, int x) {
    // These calls should use PC-relative dex cache array loads to retrieve the target method.
    // PC-relative bases used by ARM, MIPS and X86 should be pulled before the If.
    if (negate) {
      return $noinline$foo(-x);
    } else {
      return $noinline$foo(x);
    }
  }

  /// CHECK-START-X86: int Main.testLoop(int[], int) pc_relative_fixups_x86 (before)
  /// CHECK-NOT:            X86ComputeBaseMethodAddress

  /// CHECK-START-X86: int Main.testLoop(int[], int) pc_relative_fixups_x86 (after)
  /// CHECK:                X86ComputeBaseMethodAddress
  /// CHECK-NOT:            X86ComputeBaseMethodAddress

  /// CHECK-START-X86: int Main.testLoop(int[], int) pc_relative_fixups_x86 (after)
  /// CHECK:                InvokeStaticOrDirect
  /// CHECK-NOT:            InvokeStaticOrDirect

  /// CHECK-START-X86: int Main.testLoop(int[], int) pc_relative_fixups_x86 (after)
  /// CHECK:                ArrayLength
  /// CHECK-NEXT:           X86ComputeBaseMethodAddress
  /// CHECK-NEXT:           Goto
  /// CHECK:                begin_block
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-ARM: int Main.testLoop(int[], int) dex_cache_array_fixups_arm (before)
  /// CHECK-NOT:            ArmDexCacheArraysBase

  /// CHECK-START-ARM: int Main.testLoop(int[], int) dex_cache_array_fixups_arm (after)
  /// CHECK:                ArmDexCacheArraysBase
  /// CHECK-NOT:            ArmDexCacheArraysBase

  /// CHECK-START-ARM: int Main.testLoop(int[], int) dex_cache_array_fixups_arm (after)
  /// CHECK:                InvokeStaticOrDirect
  /// CHECK-NOT:            InvokeStaticOrDirect

  /// CHECK-START-ARM: int Main.testLoop(int[], int) dex_cache_array_fixups_arm (after)
  /// CHECK:                ArrayLength
  /// CHECK-NEXT:           ArmDexCacheArraysBase
  /// CHECK-NEXT:           Goto
  /// CHECK:                begin_block
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  /// CHECK-START-MIPS: int Main.testLoop(int[], int) dex_cache_array_fixups_mips (before)
  /// CHECK-NOT:            MipsDexCacheArraysBase

  /// CHECK-START-MIPS: int Main.testLoop(int[], int) dex_cache_array_fixups_mips (after)
  /// CHECK:                MipsDexCacheArraysBase
  /// CHECK-NOT:            MipsDexCacheArraysBase

  /// CHECK-START-MIPS: int Main.testLoop(int[], int) dex_cache_array_fixups_mips (after)
  /// CHECK:                InvokeStaticOrDirect
  /// CHECK-NOT:            InvokeStaticOrDirect

  /// CHECK-START-MIPS: int Main.testLoop(int[], int) dex_cache_array_fixups_mips (after)
  /// CHECK:                ArrayLength
  /// CHECK-NEXT:           MipsDexCacheArraysBase
  /// CHECK-NEXT:           Goto
  /// CHECK:                begin_block
  /// CHECK:                InvokeStaticOrDirect method_load_kind:dex_cache_pc_relative

  public static int testLoop(int[] array, int x) {
    // PC-relative bases used by ARM, MIPS and X86 should be pulled before the loop.
    for (int i : array) {
      x += $noinline$foo(i);
    }
    return x;
  }

  /// CHECK-START-X86: int Main.testLoopWithDiamond(int[], boolean, int) pc_relative_fixups_x86 (before)
  /// CHECK-NOT:            X86ComputeBaseMethodAddress

  /// CHECK-START-X86: int Main.testLoopWithDiamond(int[], boolean, int) pc_relative_fixups_x86 (after)
  /// CHECK:                If
  /// CHECK:                begin_block
  /// CHECK:                ArrayLength
  /// CHECK-NEXT:           X86ComputeBaseMethodAddress
  /// CHECK-NEXT:           Goto

  /// CHECK-START-ARM: int Main.testLoopWithDiamond(int[], boolean, int) dex_cache_array_fixups_arm (before)
  /// CHECK-NOT:            ArmDexCacheArraysBase

  /// CHECK-START-ARM: int Main.testLoopWithDiamond(int[], boolean, int) dex_cache_array_fixups_arm (after)
  /// CHECK:                If
  /// CHECK:                begin_block
  /// CHECK:                ArrayLength
  /// CHECK-NEXT:           ArmDexCacheArraysBase
  /// CHECK-NEXT:           Goto

  /// CHECK-START-MIPS: int Main.testLoopWithDiamond(int[], boolean, int) dex_cache_array_fixups_mips (before)
  /// CHECK-NOT:            MipsDexCacheArraysBase

  /// CHECK-START-MIPS: int Main.testLoopWithDiamond(int[], boolean, int) dex_cache_array_fixups_mips (after)
  /// CHECK:                If
  /// CHECK:                begin_block
  /// CHECK:                ArrayLength
  /// CHECK-NEXT:           MipsDexCacheArraysBase
  /// CHECK-NEXT:           Goto

  public static int testLoopWithDiamond(int[] array, boolean negate, int x) {
    // PC-relative bases used by ARM, MIPS and X86 should be pulled before the loop
    // but not outside the if.
    if (array != null) {
      for (int i : array) {
        if (negate) {
          x += $noinline$foo(-i);
        } else {
          x += $noinline$foo(i);
        }
      }
    }
    return x;
  }

  /// CHECK-START: java.lang.String Main.$noinline$getBootImageString() sharpening (before)
  /// CHECK:                LoadString load_kind:DexCacheViaMethod

  /// CHECK-START-X86: java.lang.String Main.$noinline$getBootImageString() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadString load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}}

  /// CHECK-START-X86_64: java.lang.String Main.$noinline$getBootImageString() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadString load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}}

  /// CHECK-START-ARM: java.lang.String Main.$noinline$getBootImageString() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadString load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}}

  /// CHECK-START-ARM64: java.lang.String Main.$noinline$getBootImageString() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadString load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}}

  /// CHECK-START-MIPS: java.lang.String Main.$noinline$getBootImageString() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadString load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}}

  public static String $noinline$getBootImageString() {
    // Prevent inlining to avoid the string comparison being optimized away.
    if (doThrow) { throw new Error(); }
    // Empty string is known to be in the boot image.
    return "";
  }

  /// CHECK-START: java.lang.String Main.$noinline$getNonBootImageString() sharpening (before)
  /// CHECK:                LoadString load_kind:DexCacheViaMethod

  /// CHECK-START-X86: java.lang.String Main.$noinline$getNonBootImageString() sharpening (after)
  /// CHECK:                LoadString load_kind:DexCachePcRelative

  /// CHECK-START-X86: java.lang.String Main.$noinline$getNonBootImageString() pc_relative_fixups_x86 (after)
  /// CHECK-DAG:            X86ComputeBaseMethodAddress
  /// CHECK-DAG:            LoadString load_kind:DexCachePcRelative

  /// CHECK-START-X86_64: java.lang.String Main.$noinline$getNonBootImageString() sharpening (after)
  /// CHECK:                LoadString load_kind:DexCachePcRelative

  /// CHECK-START-ARM: java.lang.String Main.$noinline$getNonBootImageString() sharpening (after)
  /// CHECK:                LoadString load_kind:DexCachePcRelative

  /// CHECK-START-ARM: java.lang.String Main.$noinline$getNonBootImageString() dex_cache_array_fixups_arm (after)
  /// CHECK-DAG:            ArmDexCacheArraysBase
  /// CHECK-DAG:            LoadString load_kind:DexCachePcRelative

  /// CHECK-START-ARM64: java.lang.String Main.$noinline$getNonBootImageString() sharpening (after)
  /// CHECK:                LoadString load_kind:DexCachePcRelative

  /// CHECK-START-MIPS: java.lang.String Main.$noinline$getNonBootImageString() sharpening (after)
  /// CHECK:                LoadString load_kind:DexCachePcRelative

  /// CHECK-START-MIPS: java.lang.String Main.$noinline$getNonBootImageString() dex_cache_array_fixups_mips (after)
  /// CHECK-DAG:            MipsDexCacheArraysBase
  /// CHECK-DAG:            LoadString load_kind:DexCachePcRelative

  public static String $noinline$getNonBootImageString() {
    // Prevent inlining to avoid the string comparison being optimized away.
    if (doThrow) { throw new Error(); }
    // This string is not in the boot image.
    return "non-boot-image-string";
  }

  /// CHECK-START: java.lang.Class Main.$noinline$getStringClass() sharpening (before)
  /// CHECK:                LoadClass load_kind:DexCacheViaMethod class_name:java.lang.String

  /// CHECK-START-X86: java.lang.Class Main.$noinline$getStringClass() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadClass load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}} class_name:java.lang.String

  /// CHECK-START-X86_64: java.lang.Class Main.$noinline$getStringClass() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadClass load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}} class_name:java.lang.String

  /// CHECK-START-ARM: java.lang.Class Main.$noinline$getStringClass() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadClass load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}} class_name:java.lang.String

  /// CHECK-START-ARM64: java.lang.Class Main.$noinline$getStringClass() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadClass load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}} class_name:java.lang.String

  /// CHECK-START-MIPS: java.lang.Class Main.$noinline$getStringClass() sharpening (after)
  // Note: load kind depends on PIC/non-PIC
  // TODO: Remove DexCacheViaMethod when read barrier config supports BootImageAddress.
  /// CHECK:                LoadClass load_kind:{{BootImageAddress|DexCachePcRelative|DexCacheViaMethod}} class_name:java.lang.String

  public static Class<?> $noinline$getStringClass() {
    // Prevent inlining to avoid the string comparison being optimized away.
    if (doThrow) { throw new Error(); }
    // String class is known to be in the boot image.
    return String.class;
  }

  /// CHECK-START: java.lang.Class Main.$noinline$getOtherClass() sharpening (before)
  /// CHECK:                LoadClass load_kind:DexCacheViaMethod class_name:Other

  /// CHECK-START-X86: java.lang.Class Main.$noinline$getOtherClass() sharpening (after)
  /// CHECK:                LoadClass load_kind:DexCachePcRelative class_name:Other

  /// CHECK-START-X86: java.lang.Class Main.$noinline$getOtherClass() pc_relative_fixups_x86 (after)
  /// CHECK-DAG:            X86ComputeBaseMethodAddress
  /// CHECK-DAG:            LoadClass load_kind:DexCachePcRelative class_name:Other

  /// CHECK-START-X86_64: java.lang.Class Main.$noinline$getOtherClass() sharpening (after)
  /// CHECK:                LoadClass load_kind:DexCachePcRelative class_name:Other

  /// CHECK-START-ARM: java.lang.Class Main.$noinline$getOtherClass() sharpening (after)
  /// CHECK:                LoadClass load_kind:DexCachePcRelative class_name:Other

  /// CHECK-START-ARM: java.lang.Class Main.$noinline$getOtherClass() dex_cache_array_fixups_arm (after)
  /// CHECK-DAG:            ArmDexCacheArraysBase
  /// CHECK-DAG:            LoadClass load_kind:DexCachePcRelative class_name:Other

  /// CHECK-START-ARM64: java.lang.Class Main.$noinline$getOtherClass() sharpening (after)
  /// CHECK:                LoadClass load_kind:DexCachePcRelative class_name:Other

  /// CHECK-START-MIPS: java.lang.Class Main.$noinline$getOtherClass() sharpening (after)
  /// CHECK:                LoadClass load_kind:DexCachePcRelative class_name:Other

  /// CHECK-START-MIPS: java.lang.Class Main.$noinline$getOtherClass() dex_cache_array_fixups_mips (after)
  /// CHECK-DAG:            MipsDexCacheArraysBase
  /// CHECK-DAG:            LoadClass load_kind:DexCachePcRelative class_name:Other

  public static Class<?> $noinline$getOtherClass() {
    // Prevent inlining to avoid the string comparison being optimized away.
    if (doThrow) { throw new Error(); }
    // Other class is not in the boot image.
    return Other.class;
  }

  public static void main(String[] args) {
    assertIntEquals(1, testSimple(1));
    assertIntEquals(1, testDiamond(false, 1));
    assertIntEquals(-1, testDiamond(true, 1));
    assertIntEquals(3, testLoop(new int[]{ 2 }, 1));
    assertIntEquals(8, testLoop(new int[]{ 3, 4 }, 1));
    assertIntEquals(1, testLoopWithDiamond(null, false, 1));
    assertIntEquals(3, testLoopWithDiamond(new int[]{ 2 }, false, 1));
    assertIntEquals(-6, testLoopWithDiamond(new int[]{ 3, 4 }, true, 1));
    assertStringEquals("", $noinline$getBootImageString());
    assertStringEquals("non-boot-image-string", $noinline$getNonBootImageString());
    assertClassEquals(String.class, $noinline$getStringClass());
    assertClassEquals(Other.class, $noinline$getOtherClass());
  }
}

class Other {
}
