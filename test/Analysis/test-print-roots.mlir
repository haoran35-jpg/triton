// RUN: triton-opt %s -mlir-disable-threading -test-print-roots -o /dev/null 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared  = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

// CHECK-LABEL: function @two_args
// Two funcargs, splat + addptr by constant. Offsets must be precise.
// CHECK-DAG: %arg0 roots={%arg0} off=0
// CHECK-DAG: %arg1 roots={%arg1} off=0
// CHECK-DAG: %0 roots={%arg0} off=0
// CHECK-DAG: %1 roots={%arg1} off=0
tt.func @two_args(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %sb = tt.splat %B : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  tt.return
}

// CHECK-LABEL: function @const_addptr
// Two splatted ptrs at constant offsets +128 / +129 from same root.
// CHECK-DAG: %1 roots={%arg0} off=128
// CHECK-DAG: %2 roots={%arg0} off=129
tt.func @const_addptr(%A: !tt.ptr<f16>) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c128 = arith.constant dense<128> : tensor<128xi32, #blocked>
  %c1   = arith.constant dense<1>   : tensor<128xi32, #blocked>
  %p128 = tt.addptr %sa, %c128 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
  %p129 = tt.addptr %p128, %c1 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
  tt.return
}

// CHECK-LABEL: function @loop_carried
// scf.for iter_arg %p advances by 128 each iteration. Loop-lifting must
// derive: lifted(%p) = 0 + dim_0 * 128, so getOffset on the load's ptr
// yields a dim_0-dependent affine.  (iv itself stays <bot>; that's OK.)
// CHECK-DAG: %arg5 roots={%arg0} off=d0 * 128
// CHECK-DAG: %3 roots={%arg0} off=d0 * 128 + 128
// CHECK: edge (tt.load{{.*}}, tt.store{{.*}}, dist=1, lat=400, blocking=false) kind=mem-loop-carried
tt.func @loop_carried(%A: !tt.ptr<f16>, %lb: index, %ub: index, %step: index) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c128 = arith.constant dense<128> : tensor<128xi32, #blocked>
  %res = scf.for %iv = %lb to %ub step %step iter_args(%p = %sa) -> (tensor<128x!tt.ptr<f16>, #blocked>) {
    %v0 = tt.load %p : tensor<128x!tt.ptr<f16>, #blocked>
    %next = tt.addptr %p, %c128 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %next, %v0 : tensor<128x!tt.ptr<f16>, #blocked>
    scf.yield %next : tensor<128x!tt.ptr<f16>, #blocked>
  }
  tt.return
}

// CHECK-LABEL: function @same_iter_dep
// Two memory ops at same address in same iteration: must report SameIter.
// CHECK: edge (tt.load{{.*}}, tt.store{{.*}}, dist=0, lat=400, blocking=false) kind=mem-same-iter
tt.func @same_iter_dep(%A: !tt.ptr<f16>) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %v0 = tt.load %sa : tensor<128x!tt.ptr<f16>, #blocked>
  tt.store %sa, %v0 : tensor<128x!tt.ptr<f16>, #blocked>
  tt.return
}

// CHECK-LABEL: function @no_dep_different_roots
// Two memory ops with disjoint roots: no mem-alias edge should be emitted.
// (SSA / iter-carried edges may exist and are not what we're checking here.)
// CHECK-NOT: kind=mem-
tt.func @no_dep_different_roots(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %sb = tt.splat %B : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %v0 = tt.load %sa : tensor<128x!tt.ptr<f16>, #blocked>
  tt.store %sb, %v0 : tensor<128x!tt.ptr<f16>, #blocked>
  tt.return
}

// CHECK-LABEL: function @iv_in_addptr
// IV used directly in addptr offset:  off = iv * 128.  iv-fixup substitutes
// dim for iv so the load/store within the loop differ by a known constant.
//   load  ptr off=d0*128         covers [k*128,    k*128+128)
//   store ptr off=d0*128 + 64    covers [k*128+64, k*128+192)
//   Same iter overlap = 64 elements ⇒ tile-aware reports SameIter.
// CHECK: edge (tt.load{{.*}}, tt.store{{.*}}, dist=0, lat=400, blocking=false) kind=mem-same-iter
tt.func @iv_in_addptr(%A: !tt.ptr<f16>, %lb: index, %ub: index, %step: index) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c128_idx = arith.constant 128 : index
  %c64_idx  = arith.constant 64  : index
  scf.for %iv = %lb to %ub step %step {
    %iv_off = arith.muli %iv, %c128_idx : index
    %iv_i32 = arith.index_cast %iv_off : index to i32
    %iv_vec = tt.splat %iv_i32 : i32 -> tensor<128xi32, #blocked>
    %p_load = tt.addptr %sa, %iv_vec : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %p_load : tensor<128x!tt.ptr<f16>, #blocked>
    %c64_i32 = arith.constant dense<64> : tensor<128xi32, #blocked>
    %p_store = tt.addptr %p_load, %c64_i32 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %p_store, %v : tensor<128x!tt.ptr<f16>, #blocked>
  }
  tt.return
}

// CHECK-LABEL: function @loop_carried_delta_2
// iter_arg %p advances by 128 per iter (slope α=128).  load reads %p;
// store writes %p + 256.  Per-iter slopes match → diff = 256 (constant).
// Cross-iter solve:  store(k) writes addr  k*128 + 256
//                    load(k+2) reads addr (k+2)*128 = k*128 + 256
// ⇒ loop-carried with δ = 2.
// CHECK: edge (tt.load{{.*}}, tt.store{{.*}}, dist=2, lat=400, blocking=false) kind=mem-loop-carried
tt.func @loop_carried_delta_2(%A: !tt.ptr<f16>, %lb: index, %ub: index, %step: index) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c128 = arith.constant dense<128> : tensor<128xi32, #blocked>
  %c256 = arith.constant dense<256> : tensor<128xi32, #blocked>
  %res = scf.for %iv = %lb to %ub step %step iter_args(%p = %sa) -> (tensor<128x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x!tt.ptr<f16>, #blocked>
    %p_store = tt.addptr %p, %c256 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %p_store, %v : tensor<128x!tt.ptr<f16>, #blocked>
    %next = tt.addptr %p, %c128 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    scf.yield %next : tensor<128x!tt.ptr<f16>, #blocked>
  }
  tt.return
}

// CHECK-LABEL: function @loop_carried_delta_3
// α=64, store writes %p + 192, tile W=128.  Interval-overlap window:
//   |192 - 64·δ| < 128  ⇒  δ ∈ {2, 3, 4}.
// The tightest (smallest) binding δ = 2 — partial overlap of 64 elements at
// δ=2, full alias at δ=3, partial again at δ=4.  Scheduler must respect δ=2.
// CHECK: edge (tt.load{{.*}}, tt.store{{.*}}, dist=2, lat=400, blocking=false) kind=mem-loop-carried
tt.func @loop_carried_delta_3(%A: !tt.ptr<f16>, %lb: index, %ub: index, %step: index) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c64  = arith.constant dense<64>  : tensor<128xi32, #blocked>
  %c192 = arith.constant dense<192> : tensor<128xi32, #blocked>
  %res = scf.for %iv = %lb to %ub step %step iter_args(%p = %sa) -> (tensor<128x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x!tt.ptr<f16>, #blocked>
    %p_store = tt.addptr %p, %c192 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %p_store, %v : tensor<128x!tt.ptr<f16>, #blocked>
    %next = tt.addptr %p, %c64 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    scf.yield %next : tensor<128x!tt.ptr<f16>, #blocked>
  }
  tt.return
}

// CHECK-LABEL: function @loop_carried_delta_3_clean
// α=128, store writes %p + 384, tile W=128.  Window:
//   |384 - 128·δ| < 128  ⇒  257 ≤ 128·δ ≤ 511  ⇒  δ ∈ {3}.
// Clean δ=3 because adjacent δ=2 / δ=4 fall outside the overlap window
// (tile gap 128 exactly matches the slope).
// CHECK: edge (tt.load{{.*}}, tt.store{{.*}}, dist=3, lat=400, blocking=false) kind=mem-loop-carried
tt.func @loop_carried_delta_3_clean(%A: !tt.ptr<f16>, %lb: index, %ub: index, %step: index) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c128 = arith.constant dense<128> : tensor<128xi32, #blocked>
  %c384 = arith.constant dense<384> : tensor<128xi32, #blocked>
  %res = scf.for %iv = %lb to %ub step %step iter_args(%p = %sa) -> (tensor<128x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x!tt.ptr<f16>, #blocked>
    %p_store = tt.addptr %p, %c384 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %p_store, %v : tensor<128x!tt.ptr<f16>, #blocked>
    %next = tt.addptr %p, %c128 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    scf.yield %next : tensor<128x!tt.ptr<f16>, #blocked>
  }
  tt.return
}

// CHECK-LABEL: function @no_dep_wide_gap
// α=128, β=4096, W=128:  smallest |4096 - 128·δ| < 128 is at δ=32 (exact match
// → 128 overlap).  But the test sticks to small iter counts; we don't filter
// by trip count, so we still emit δ=32 here.  Demonstrates large clean δ.
// CHECK: edge (tt.load{{.*}}, tt.store{{.*}}, dist=32, lat=400, blocking=false) kind=mem-loop-carried
tt.func @no_dep_wide_gap(%A: !tt.ptr<f16>, %lb: index, %ub: index, %step: index) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c128  = arith.constant dense<128>  : tensor<128xi32, #blocked>
  %c4096 = arith.constant dense<4096> : tensor<128xi32, #blocked>
  %res = scf.for %iv = %lb to %ub step %step iter_args(%p = %sa) -> (tensor<128x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x!tt.ptr<f16>, #blocked>
    %p_store = tt.addptr %p, %c4096 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %p_store, %v : tensor<128x!tt.ptr<f16>, #blocked>
    %next = tt.addptr %p, %c128 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    scf.yield %next : tensor<128x!tt.ptr<f16>, #blocked>
  }
  tt.return
}

// CHECK-LABEL: function @truly_no_dep
// α=256 (advance c256 per iter), store writes %p + 128, tile W=128.
// Window |128 - 256·δ| < 128:
//   δ=0: |128|=128, NOT < 128 → load[k*256, k*256+128) and store at
//        [k*256+128, k*256+256) are adjacent, NO overlap.
//   δ=1: |128-256|=128, NOT < 128 → also adjacent the other way.
//   δ ≥ 2 or δ ≤ -1: further apart.
// No integer δ in window ⇒ truly No mem-alias edge.
// (SSA / iter-carried edges may exist and are not what we're checking here.)
// CHECK-NOT: kind=mem-
tt.func @truly_no_dep(%A: !tt.ptr<f16>, %lb: index, %ub: index, %step: index) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c256 = arith.constant dense<256> : tensor<128xi32, #blocked>
  %c128 = arith.constant dense<128> : tensor<128xi32, #blocked>
  %res = scf.for %iv = %lb to %ub step %step iter_args(%p = %sa) -> (tensor<128x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x!tt.ptr<f16>, #blocked>
    %p_store = tt.addptr %p, %c128 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %p_store, %v : tensor<128x!tt.ptr<f16>, #blocked>
    %next = tt.addptr %p, %c256 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    scf.yield %next : tensor<128x!tt.ptr<f16>, #blocked>
  }
  tt.return
}

// CHECK-LABEL: function @async_blocking
// async_copy_global_to_local reads the same global addr as a subsequent
// tt.store; pairing yields a SameIter edge with blocking=true on the async
// side.  The regular tt.load → tt.store pair below it stays blocking=false.
// CHECK-DAG: edge (ttg.async_copy_global_to_local{{.*}}, tt.store{{.*}}, dist=0, lat={{.*}}, blocking=true) kind=mem-same-iter
// CHECK-DAG: edge (tt.load{{.*}}, tt.store{{.*}}, dist=0, lat=400, blocking=false) kind=mem-same-iter
tt.func @async_blocking(%A: !tt.ptr<f16>) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
  %tok = ttg.async_copy_global_to_local %sa, %buf : tensor<128x!tt.ptr<f16>, #blocked> -> <128xf16, #shared, #ttg.shared_memory, mutable>
  %v = tt.load %sa : tensor<128x!tt.ptr<f16>, #blocked>
  tt.store %sa, %v : tensor<128x!tt.ptr<f16>, #blocked>
  tt.return
}

// CHECK-LABEL: function @nested_loop
// Outer iter_arg %p0 advances by 128 per outer iter; inner iter_arg %p1
// (init = %p0) advances by 1 per inner iter.  Load/store happen inside the
// inner body using %p1 and (%p1 + 1).  Diff = 1, tile W=128.
//   Same-iter:    |1 - 1·0| = 1 < 128  ⇒  127-element overlap.
// Tightest binding is SameIter (also has δ=1 exact-alias, but δ=0 dominates).
// CHECK: edge (tt.load{{.*}}, tt.store{{.*}}, dist=0, lat=400, blocking=false) kind=mem-same-iter
tt.func @nested_loop(%A: !tt.ptr<f16>, %lb: index, %ub: index, %step: index) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %c128 = arith.constant dense<128> : tensor<128xi32, #blocked>
  %c1   = arith.constant dense<1>   : tensor<128xi32, #blocked>
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%p0 = %sa) -> (tensor<128x!tt.ptr<f16>, #blocked>) {
    %r1 = scf.for %j = %lb to %ub step %step iter_args(%p1 = %p0) -> (tensor<128x!tt.ptr<f16>, #blocked>) {
      %v = tt.load %p1 : tensor<128x!tt.ptr<f16>, #blocked>
      %next1 = tt.addptr %p1, %c1 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
      tt.store %next1, %v : tensor<128x!tt.ptr<f16>, #blocked>
      scf.yield %next1 : tensor<128x!tt.ptr<f16>, #blocked>
    }
    %next0 = tt.addptr %r1, %c128 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    scf.yield %next0 : tensor<128x!tt.ptr<f16>, #blocked>
  }
  tt.return
}

// CHECK-LABEL: function @async_commit_group_blocking
// async_copy at N-1, async_commit_group at N, async_wait at N+1 (skipped),
// local_load at N+2 (target).
// Expected: prev(async_copy) -> next_next(local_load), dist=0, blocking=true.
// CHECK: edge (ttg.async_copy_global_to_local{{.*}}, ttg.local_load{{.*}}, dist=0, lat={{.*}}, blocking=true) kind=ssa
tt.func @async_commit_group_blocking(%A: !tt.ptr<f16>) {
  %sa  = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %buf = ttg.local_alloc : ()
      -> !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
  %tok = ttg.async_copy_global_to_local %sa, %buf
      : tensor<128x!tt.ptr<f16>, #blocked>
        -> <128xf16, #shared, #ttg.shared_memory, mutable>
  ttg.async_commit_group
  ttg.async_wait { num = 0 : i32 }
  %tile = ttg.local_load %buf
      : !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
        -> tensor<128xf16, #blocked>
  tt.return
}

// CHECK-LABEL: function @mbarrier_wait_blocking
// Any op at N-1, wait_barrier at N, local_load at N+1 (target).
// Expected: prev(init_barrier) -> next(local_load), dist=0, blocking=true.
// CHECK: edge (ttng.init_barrier{{.*}}, ttg.local_load{{.*}}, dist=0, lat=1, blocking=true) kind=ssa
tt.func @mbarrier_wait_blocking(%A: !tt.ptr<f16>) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %buf_data = ttg.local_alloc : ()
      -> !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
  %barrier = ttg.local_alloc : ()
      -> !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
  %c0 = arith.constant 0 : i32
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
  %tile = ttg.local_load %buf_data
      : !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
        -> tensor<128xf16, #blocked>
  tt.store %sa, %tile : tensor<128x!tt.ptr<f16>, #blocked>
  tt.return
}

// CHECK-LABEL: function @warp_group_dot_wait_blocking
// Any op at N-1, warp_group_dot_wait at N, local_load at N+1 (target).
// Expected: prev(constant) -> next(local_load), dist=0, blocking=true.
// CHECK: edge (arith.constant{{.*}}, ttg.local_load{{.*}}, dist=0, lat=1, blocking=true) kind=ssa
tt.func @warp_group_dot_wait_blocking(%A: !tt.ptr<f16>) {
  %sa   = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %buf = ttg.local_alloc : ()
      -> !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
  %acc  = arith.constant dense<0.0> : tensor<128xf32, #blocked>
  %acc_wait = ttng.warp_group_dot_wait %acc { pendings = 0 : i32 }
      : tensor<128xf32, #blocked>
  %tile = ttg.local_load %buf
      : !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
        -> tensor<128xf16, #blocked>
  tt.store %sa, %tile : tensor<128x!tt.ptr<f16>, #blocked>
  tt.return
}

// CHECK-LABEL: function @async_commit_group_no_next2
// async_copy at N-1, async_commit_group at N, nothing at N+2.
// No edge should be emitted when next_next is null.
// CHECK-NOT: blocking=true
tt.func @async_commit_group_no_next2(%A: !tt.ptr<f16>) {
  %sa  = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %buf = ttg.local_alloc : ()
      -> !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
  %tok = ttg.async_copy_global_to_local %sa, %buf
      : tensor<128x!tt.ptr<f16>, #blocked>
        -> <128xf16, #shared, #ttg.shared_memory, mutable>
  ttg.async_commit_group
  tt.return
}

// CHECK-LABEL: function @tc_gen5_commit_blocking
// tc_gen5_commit at N should create a bridge edge from N-1 to N+2.
// Expected: prev(init_barrier) -> next2(local_load), dist=0, blocking=true.
// CHECK: edge (ttng.init_barrier{{.*}}, ttg.local_load{{.*}}, dist=0, lat=1, blocking=true) kind=ssa
tt.func @tc_gen5_commit_blocking(%A: !tt.ptr<f16>) {
  %sa = tt.splat %A : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
  %buf_data = ttg.local_alloc : ()
      -> !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
  %barrier = ttg.local_alloc : ()
      -> !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
  ttng.tc_gen5_commit %barrier : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
  %c0 = arith.constant 0 : i32
  %tile = ttg.local_load %buf_data
      : !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
        -> tensor<128xf16, #blocked>
  tt.store %sa, %tile : tensor<128x!tt.ptr<f16>, #blocked>
  tt.return
}
}
