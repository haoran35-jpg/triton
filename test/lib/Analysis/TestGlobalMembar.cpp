#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/GlobalMembar.h"
#include "triton/Analysis/TwillTypes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

using namespace mlir;

namespace {

static bool isPtrLike(Type t) {
  if (isa<triton::PointerType>(t))
    return true;
  if (auto rt = dyn_cast<RankedTensorType>(t))
    return isa<triton::PointerType>(rt.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// test-print-roots: emit the lattice `(roots, offset)` for every interesting
// Value, plus mayDepend results between every pair of memory ops (tt.load /
// tt.store / tt.atomic_*).
//===----------------------------------------------------------------------===//
struct TestGlobalMembarPass
    : public PassWrapper<TestGlobalMembarPass,
                         OperationPass<triton::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGlobalMembarPass);

  StringRef getArgument() const final { return "test-print-roots"; }
  StringRef getDescription() const final {
    return "print (roots, offset) per Value and mayDepend pairs for memory ops";
  }

  static std::string nameOf(Value v, AsmState &state) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    v.printAsOperand(ss, state);
    return s;
  }

  static std::string nameOfOp(Operation *op, AsmState &state) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << op->getName().getStringRef();
    if (op->getNumResults() > 0) {
      ss << "(";
      op->getResult(0).printAsOperand(ss, state);
      ss << ")";
    } else if (op->getNumOperands() > 0) {
      ss << "[ptr=";
      op->getOperand(0).printAsOperand(ss, state);
      ss << "]";
    }
    return s;
  }


  void runOnOperation() override {
    Operation *funcOp = getOperation();

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    triton::AffineSymbolTable syms(funcOp->getContext());
    auto *analysis = solver->load<triton::RootOffsetAnalysis>(&syms);
    if (failed(solver->initializeAndRun(funcOp)))
      return signalPassFailure();

    triton::LiftedMap lifted;
    triton::runLoopLifting(funcOp, *analysis, syms, lifted);

    AsmState state(funcOp->getParentOfType<ModuleOp>());

    auto emitVal = [&](Value v, Location loc) {
      InFlightDiagnostic diag = emitRemark(loc);
      diag << nameOf(v, state) << " ";
      const triton::OffsetInfo *info = triton::getOffsetInfo(v, *analysis);
      if (!info) {
        diag << "<no-lattice>";
        return;
      }
      // roots
      diag << "roots=";
      if (info->isUnknownRoots) {
        diag << "<top>";
      } else {
        diag << "{";
        SmallVector<std::string> names;
        for (Value r : info->roots)
          names.push_back(nameOf(r, state));
        std::sort(names.begin(), names.end());
        llvm::interleaveComma(names, diag, [&](StringRef n) { diag << n; });
        diag << "}";
      }
      // offset (use lifted helper for ptr-like values)
      diag << " off=";
      if (isPtrLike(v.getType())) {
        AffineExpr off =
            triton::getLiftedOffset(v, *analysis, syms, lifted);
        if (off) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          off.print(ss);
          diag << s;
        } else {
          diag << "<top>";
        }
      } else {
        switch (info->offsetState) {
        case triton::OffsetState::Bottom:
          diag << "<bot>";
          break;
        case triton::OffsetState::Top:
          diag << "<top>";
          break;
        case triton::OffsetState::Known: {
          std::string s;
          llvm::raw_string_ostream ss(s);
          info->offset.print(ss);
          diag << s;
          break;
        }
        }
      }
    };


    if (auto func = dyn_cast<FunctionOpInterface>(funcOp))
      emitRemark(func.getLoc()) << "function @" << func.getName();

    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            emitVal(arg, op->getLoc());
      for (Value r : op->getResults())
        emitVal(r, op->getLoc());
    });

    // Unified edge dump: SSA def-use + scf.for iter_arg carry + mem alias,
    // each tagged with kind.
    for (const triton::DepEdge &e : triton::collectEdges(funcOp)) {
      InFlightDiagnostic diag = emitRemark(e.v->getLoc());
      diag << "edge (" << nameOfOp(e.u, state) << ", " << nameOfOp(e.v, state)
           << ", dist=" << e.dist << ", lat=" << e.lat
           << ", blocking=" << (e.blocking ? "true" : "false")
           << ") kind=" << triton::edgeKindName(e.kind);
    }

    // Per-iteration static address analysis for each tt.load.
    twill::LoadAccessTable loads;
    triton::analyzeLoadAccesses(funcOp, *analysis, syms, lifted, loads);
    for (const twill::LoadAccess &la : loads.loads) {
      InFlightDiagnostic diag = emitRemark(funcOp->getLoc());
      diag << "load " << la.opName << " tile=" << la.tileWidth
           << " base=" << la.constBase << (la.baseOpaque ? "+opaque" : "")
           << (la.affineInIv ? "" : " (iv-invariant/unknown)")
           << " strides[outer..inner]={";
      for (size_t d = 0; d < la.ivStrides.size(); ++d)
        diag << (d ? "," : "") << la.ivStrides[d];
      diag << "} trips={";
      for (size_t d = 0; d < la.ivTrip.size(); ++d)
        diag << (d ? "," : "") << la.ivTrip[d];
      diag << "} addr[innermost i]=";
      for (int i = 0; i < 4; ++i)
        diag << (i ? "," : "") << (la.baseOpaque ? "base+" : "")
             << la.addressAt(i);
    }
  }
};

//===----------------------------------------------------------------------===//
// test-twill-schedule: end-to-end wiring
//   collectEdges  ->  op-id map  ->  ILPProblem (solveILP)  ->  SMTProblem
//   (solveSMT)  ->  print op[] / opw[]
//===----------------------------------------------------------------------===//
struct TestTwillSchedulePass
    : public PassWrapper<TestTwillSchedulePass,
                         OperationPass<triton::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTwillSchedulePass);

  StringRef getArgument() const final { return "test-twill-schedule"; }
  StringRef getDescription() const final {
    return "collectEdges -> ILP -> SMT, printing schedule + warp assignment";
  }

  // Functional-unit class for the resource-reservation table (RRT).
  //   0 = TENSOR  (tensor-core / MMA pipe)
  //   1 = MEM     (global / shared / tensor-memory data movement = LSU)
  //   2 = ALU     (everything else: address math, allocs, sync, arith)
  enum FU { FU_TENSOR = 0, FU_MEM = 1, FU_ALU = 2, FU_COUNT = 3 };
  static int classifyFU(Operation *op) {
    if (isa<triton::nvidia_gpu::TCGen5MMAOp, triton::nvidia_gpu::WarpGroupDotOp,
            triton::DotOp>(op))
      return FU_TENSOR;
    if (isa<triton::LoadOp, triton::StoreOp, triton::gpu::LocalLoadOp,
            triton::gpu::LocalStoreOp, triton::gpu::AsyncCopyGlobalToLocalOp,
            triton::nvidia_gpu::TMEMLoadOp, triton::nvidia_gpu::TMEMStoreOp>(op))
      return FU_MEM;
    return FU_ALU;
  }
  static const char *fuName(int f) {
    switch (f) {
    case FU_TENSOR:
      return "TENSOR";
    case FU_MEM:
      return "MEM";
    default:
      return "ALU";
    }
  }

  void runOnOperation() override {
    Operation *funcOp = getOperation();

    // 1) Pick the first loop; its body ops become scheduling units.
    scf::ForOp target;
    funcOp->walk([&](scf::ForOp forOp) {
      if (!target)
        target = forOp;
    });
    if (!target) {
      emitRemark(funcOp->getLoc()) << "twill: no scf.for found, skipping";
      return;
    }

    AsmState state(funcOp->getParentOfType<ModuleOp>());
    llvm::DenseMap<Operation *, int> opId;
    SmallVector<Operation *> opById;
    for (Operation &op : target.getBody()->without_terminator()) {
      if (op.getNumRegions() != 0)
        continue;
      if (op.hasTrait<OpTrait::ConstantLike>())
        continue;
      opId[&op] = (int)opById.size();
      opById.push_back(&op);
    }
    const int V = (int)opById.size();
    if (V == 0) {
      emitRemark(target.getLoc()) << "twill: empty loop body, skipping";
      return;
    }

    // 2) collectEdges -> twill edges, keeping only loop-body endpoints.
    //    Dedup on (u,v,dist,lat); blocking flag is OR-ed.
    std::map<std::tuple<int, int, int, int>, bool> uniq;
    for (const triton::DepEdge &e : triton::collectEdges(funcOp)) {
      auto itu = opId.find(e.u);
      auto itv = opId.find(e.v);
      if (itu == opId.end() || itv == opId.end())
        continue;
      auto key = std::make_tuple(itu->second, itv->second, e.dist, (int)e.lat);
      uniq[key] = uniq[key] || e.blocking;
    }
    // Scale raw pipeline latencies (e.g. 400-cycle global loads) down to a
    // compact range so the time-indexed ILP/SMT horizon L stays tractable.
    auto scaleLat = [](int lat) { return std::max(1, (lat + 16) / 32); };
    std::vector<twill::Edge> edges;
    std::vector<bool> blocking;
    for (auto &[k, blk] : uniq) {
      auto [u, v, dist, lat] = k;
      edges.push_back({u, v, scaleLat(lat), dist});
      blocking.push_back(blk);
    }

    // Makespan upper bound: a fully serial schedule places every op after the
    // sum of all edge latencies, plus V cycles of single-cycle issue slack.
    // With Iu=1 the iteration does not overlap itself, so II = L = makespan.
    int sumLat = V;
    for (const auto &e : edges)
      sumLat += e.d;
    const int II = sumLat;
    const int L = II;

    llvm::outs() << "== twill schedule for @"
                 << cast<FunctionOpInterface>(funcOp).getName() << " ==\n";
    llvm::outs() << "V=" << V << " |E|=" << edges.size() << " II=" << II
                 << " L=" << L << "\n";
    for (int v = 0; v < V; ++v) {
      llvm::outs() << "  op[" << v << "] = ";
      opById[v]->getName().print(llvm::outs());
      if (opById[v]->getNumResults())
        opById[v]->getResult(0).printAsOperand(llvm::outs(), state);
      llvm::outs() << "\n";
    }
    for (size_t k = 0; k < edges.size(); ++k)
      llvm::outs() << "  edge (" << edges[k].u << "->" << edges[k].v
                   << ", d=" << edges[k].d << ", delta=" << edges[k].delta
                   << (blocking[k] ? ", BLOCKING" : "") << ")\n";

    // 3) Step-1 ILP, with the resource-reservation table (RRT) turned on.
    //    Each op issues into one functional-unit class and occupies it for one
    //    cycle (cycles(v)=1); the per-cycle issue width is cap(f).  The long
    //    pipeline latency stays on the dependence edges (e.d), not in cycles.
    twill::ILPProblem ilp;
    ilp.V = V;
    ilp.edges = edges;
    ilp.cycles.assign(V, 1);
    ilp.F = FU_COUNT;
    ilp.cap = {/*TENSOR*/ 1, /*MEM*/ 1, /*ALU*/ 2};
    ilp.RRT.assign(V, std::vector<std::vector<int>>(ilp.F));
    for (int v = 0; v < V; ++v) {
      int f = classifyFU(opById[v]);
      ilp.RRT[v][f] = {1}; // busy for 1 cycle at the issue slot
    }
    ilp.M = 0;
    ilp.II = II;
    ilp.L = L;
    ilp.deriveIu();

    llvm::outs() << "\n== functional units (cap: TENSOR=" << ilp.cap[0]
                 << " MEM=" << ilp.cap[1] << " ALU=" << ilp.cap[2] << ") ==\n";
    for (int v = 0; v < V; ++v)
      llvm::outs() << "  op[" << v << "]  fu=" << fuName(classifyFU(opById[v]))
                   << "\n";
    twill::Schedule sched = twill::solveILP(ilp);
    if (!sched.solved) {
      llvm::outs() << "[ILP] infeasible\n";
      return;
    }
    llvm::outs() << "\n== op[] issue time (ILP) ==\n";
    for (int v = 0; v < V; ++v) {
      int t = -1;
      for (int tt = 0; tt <= L; ++tt) {
        auto it = sched.op.find({v, 0, tt});
        if (it != sched.op.end() && it->second) {
          t = tt;
          break;
        }
      }
      llvm::outs() << "  op[" << v << "]  t=" << t << "\n";
    }

    // 4) Step-2 SMT
    twill::SMTProblem smt;
    smt.sched = sched;
    smt.II = II;
    smt.L = L;
    smt.Iu = ilp.Iu;
    smt.V = V;
    smt.edges = edges;
    smt.blocking = blocking;
    smt.cycles.assign(V, 1);
    smt.regs.assign(V, 1);
    smt.variable_latency.assign(V, false);
    for (int v = 0; v < V; ++v)
      if (triton::isAsyncIssuer(opById[v]))
        smt.variable_latency[v] = true;
    smt.spillcost.assign(V, 0);

    int numWarps = 4;
    if (auto mod = funcOp->getParentOfType<ModuleOp>())
      if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
        numWarps = attr.getInt();
    smt.W = std::max(numWarps, 2);
    smt.W_vl = 0;
    smt.reg_limit = 1 << 20;

    twill::WarpAssignment wa = twill::solveSMT(smt);
    if (!wa.solved) {
      llvm::outs() << "[SMT] UNSAT\n";
      return;
    }
    llvm::outs() << "\n== opw[] warp assignment (SMT) ==\n";
    for (int v = 0; v < V; ++v) {
      llvm::outs() << "  op[" << v << "]  warp=" << wa.assign[v]
                   << (smt.variable_latency[v] ? "  (variable_latency)" : "")
                   << (wa.assign[v] == smt.W_vl ? "  [W_vl]" : "") << "\n";
    }
  }
};

//===----------------------------------------------------------------------===//
// test-twill-cache-joint: end-to-end wiring of the joint cache/RR/warp solver.
//   collectEdges        -> per-op dependence (u, iter-distance)
//   analyzeLoadAccesses -> per-load Line[v,i] = lineBase + i*lineStride
//   => CacheRRIterProblem -> solveTwillCacheJoint -> print warp/start/lat/hit.
//===----------------------------------------------------------------------===//
struct TestTwillCacheJointPass
    : public PassWrapper<TestTwillCacheJointPass,
                         OperationPass<triton::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTwillCacheJointPass);

  StringRef getArgument() const final { return "test-twill-cache-joint"; }
  StringRef getDescription() const final {
    return "collectEdges + analyzeLoadAccesses -> solveTwillCacheJoint";
  }

  // Each load is modelled as
  // touching a single representative line line(v,i)=lineBase+i*lineStride.
  static constexpr long kLine = 32;

  // Scaled latencies keep the SMT timeline T small enough to stay tractable.
  static int staticLatOf(Operation *op) {
    if (isa<triton::nvidia_gpu::TCGen5MMAOp,
            triton::nvidia_gpu::WarpGroupDotOp, triton::DotOp>(op))
      return 8;
    if (isa<triton::StoreOp, triton::gpu::LocalLoadOp,
            triton::gpu::LocalStoreOp, triton::nvidia_gpu::TMEMLoadOp,
            triton::nvidia_gpu::TMEMStoreOp>(op))
      return 4;
    return 1;
  }

  // Latency-bearing ops that are worth scheduling as their own units.  For very
  // large loop bodies (e.g. attention's KV loop, ~70 ops) we schedule only
  // these and fold the address arithmetic into transitive dependence edges.
  static bool isHeavyOp(Operation *op) {
    return isa<triton::DotOp, triton::nvidia_gpu::WarpGroupDotOp,
               triton::nvidia_gpu::TCGen5MMAOp, triton::ReduceOp,
               triton::LoadOp, triton::StoreOp, triton::gpu::LocalLoadOp,
               triton::gpu::LocalStoreOp,
               triton::gpu::AsyncCopyGlobalToLocalOp,
               triton::gpu::AsyncCommitGroupOp, triton::gpu::AsyncWaitOp,
               triton::nvidia_gpu::TMEMLoadOp, triton::nvidia_gpu::TMEMStoreOp>(
        op);
  }

  void runOnOperation() override {
    Operation *funcOp = getOperation();

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    triton::AffineSymbolTable syms(funcOp->getContext());
    auto *analysis = solver->load<triton::RootOffsetAnalysis>(&syms);
    if (failed(solver->initializeAndRun(funcOp)))
      return signalPassFailure();
    triton::LiftedMap lifted;
    triton::runLoopLifting(funcOp, *analysis, syms, lifted);

    // 1) Pick the innermost loop (the one with no nested scf.for) so its body
    //    holds the loads whose per-iteration address we model.
    scf::ForOp target;
    funcOp->walk([&](scf::ForOp forOp) {
      bool hasInner = false;
      forOp.getBody()->walk([&](scf::ForOp inner) {
        if (inner != forOp)
          hasInner = true;
      });
      if (!hasInner)
        target = forOp;
    });
    if (!target) {
      emitRemark(funcOp->getLoc()) << "twill: no scf.for found, skipping";
      return;
    }

    AsmState state(funcOp->getParentOfType<ModuleOp>());

    // Full loop-body op list (including region-bearing reduces) used to
    // reconnect dependences transitively when only heavy ops are scheduled.
    SmallVector<Operation *> allOps;
    llvm::DenseMap<Operation *, int> allId;
    for (Operation &op : target.getBody()->without_terminator()) {
      if (op.hasTrait<OpTrait::ConstantLike>())
        continue;
      allId[&op] = (int)allOps.size();
      allOps.push_back(&op);
    }

    // Schedule only the latency-bearing ("heavy") ops; light ops are folded
    // into the dependence edges via the transitive reconnection below.
    llvm::DenseMap<Operation *, int> opId;
    SmallVector<Operation *> opById;
    for (Operation *op : allOps) {
      if (!isHeavyOp(op))
        continue;
      opId[op] = (int)opById.size();
      opById.push_back(op);
    }
    const int V = (int)opById.size();
    if (V == 0) {
      emitRemark(target.getLoc()) << "twill: empty loop body, skipping";
      return;
    }

    // 2) analyzeLoadAccesses -> LoadAccess per tt.load
    twill::LoadAccessTable loads;
    triton::analyzeLoadAccesses(funcOp, *analysis, syms, lifted, loads);
    SmallVector<Operation *> loadOps;
    funcOp->walk([&](triton::LoadOp l) { loadOps.push_back(l.getOperation()); });
    llvm::DenseMap<Operation *, const twill::LoadAccess *> laOf;
    llvm::DenseMap<Operation *, int> loadIdx; // global tt.load walk-order index
    for (size_t k = 0; k < loadOps.size(); ++k) {
      loadIdx[loadOps[k]] = (int)k;
      if (k < loads.loads.size())
        laOf[loadOps[k]] = &loads.loads[k];
    }

    // Per-load profiled hit/miss latencies, keyed by global load walk order and
    // scaled to keep the time-indexed SMT horizon T tractable.
    StringRef fnName = cast<FunctionOpInterface>(funcOp).getName();
    auto loadLatFor = [&](int idx, int &fast, int &slow) {
      (void)idx;
      fast = 10; // softmax test file profile hit ~50000 cyc / 5000
      slow = 14; //                  miss ~70000 cyc / 5000
    };

    twill::CacheRRIterProblem P;
    P.I = 2;
    P.S = 2;
    // Set-associative abstract cache: 4 sets × 8 ways (per-set round-robin
    // victim). Set index of each load = line % numSets.
    P.numSets = 4;
    P.numWays = 8;
    P.issueWidth = 1;
    int numWarps = 2;
    if (auto mod = funcOp->getParentOfType<ModuleOp>())
      if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
        numWarps = attr.getInt();
    P.W = std::min(std::max(numWarps, 2), 3);

    bool isAttn =
        fnName.contains("tem_fused") || fnName.contains("attention");

    // regs(v): per-thread register footprint of an op's result, derived from
    // its tile size.  A value's tensor is distributed over all threads in the
    // CTA (threadsPerWarp=32 × num-warps); a 32-bit register holds 32 bits, so
    //   perThread = ceil(totalElems · elemBits / (threadsPerCTA · 32)).
    auto regsOf = [&](Operation *op) -> int {
      if (op->getNumResults() == 0)
        return 0;
      auto rtt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!rtt)
        return 1; // scalar value ≈ 1 register
      long elems = 1;
      for (int64_t d : rtt.getShape())
        elems *= d;
      Type et = rtt.getElementType();
      int bits = et.isIntOrFloat() ? et.getIntOrFloatBitWidth() : 32;
      long threads = 32L * std::max(1, numWarps);
      long perThread = (elems * (long)bits + threads * 32 - 1) / (threads * 32);
      return (int)std::max<long>(1, perThread);
    };

    P.ops.assign(V, {});
    int dotOrd = 0, llOrd = 0;
    for (int v = 0; v < V; ++v) {
      Operation *op = opById[v];
      twill::CacheRRIterProblem::OpInfo &o = P.ops[v];
      o.staticLat = staticLatOf(op);
      o.blocking = triton::isAsyncIssuer(op) ||
                   isa<triton::gpu::AsyncWaitOp>(op); // holds its warp
      // Cross-warp spill cost: a consumer on a different warp than this
      // producer pays extra cycles to read the value; tensors cost more than
      // scalars, ops with no result cost nothing.
      if (op->getNumResults() == 0)
        o.spillcost = 0;
      else if (isa<RankedTensorType>(op->getResult(0).getType()))
        o.spillcost = 3;
      else
        o.spillcost = 1;

      // Per-thread register footprint (for the per-warp reg-limit constraint).
      o.footprint = regsOf(op);

      if (isAttn) {
        // Profiled attention op cycles, scaled /1000 (attention_op_cycles.txt):
        //   dot QK 2037, dot PV 3534, reduces ~150-231, local_load smem hit 65
        if (isa<triton::DotOp>(op)) {
          o.staticLat = (dotOrd++ == 0) ? 2 : 4; // QK then PV
        } else if (isa<triton::ReduceOp>(op)) {
          o.staticLat = 1;
        } else if (isa<triton::gpu::LocalLoadOp>(op)) {
          o.isLoad = true;
          o.fastLat = 1;             // smem hit
          o.slowLat = 3;             // stall waiting on cp.async
          o.lineBase = 100 + llOrd;  // each smem buffer its own line
          o.lineStride = 0;          // reused across iterations (hit-able)
          ++llOrd;
        } else if (isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                       triton::gpu::AsyncCommitGroupOp,
                       triton::gpu::AsyncWaitOp>(op)) {
          o.staticLat = 1;
        }
      }

      if (isa<triton::LoadOp>(op)) {
        o.isLoad = true;
        loadLatFor(loadIdx.count(op) ? loadIdx[op] : -1, o.fastLat, o.slowLat);
        auto it = laOf.find(op);
        if (it != laOf.end()) {
          const twill::LoadAccess *la = it->second;
          o.lineBase = (int)(la->constBase / kLine);
          o.lineStride = la->stride == 0 ? 0 : std::max<long>(1, la->stride / kLine);
        } else {
          o.lineBase = v;
          o.lineStride = 1;
        }
      }
    }

    // 4) deps kept only between scheduled ops(load, mma, reduce)
    std::vector<std::vector<std::pair<int, int>>> adj(allOps.size());
    for (const triton::DepEdge &e : triton::collectEdges(funcOp)) {
      auto iu = allId.find(e.u), iv = allId.find(e.v);
      if (iu == allId.end() || iv == allId.end())
        continue;
      adj[iu->second].push_back({iv->second, e.dist});
    }
    std::map<std::tuple<int, int, int>, bool> seen;
    auto addDep = [&](int cons, int prod, int dist) {
      if (cons == prod)
        return;
      auto key = std::make_tuple(cons, prod, dist);
      if (seen.count(key))
        return;
      seen[key] = true;
      P.ops[cons].deps.push_back({prod, dist});
    };
    for (int s = 0; s < (int)allOps.size(); ++s) {
      auto sit = opId.find(allOps[s]);
      if (sit == opId.end())
        continue; // s is not a scheduled op
      // BFS from heavy source s through non-heavy nodes to heavy sinks.
      std::vector<char> vis(allOps.size(), 0);
      std::vector<std::pair<int, int>> stk(adj[s].begin(), adj[s].end());
      while (!stk.empty()) {
        auto [n, d] = stk.back();
        stk.pop_back();
        if (opId.count(allOps[n])) {       // reached a scheduled (heavy) op
          addDep(opId[allOps[n]], sit->second, d);
          continue;                        // boundary: stop expanding past it
        }
        if (vis[n])
          continue;
        vis[n] = 1;
        for (auto [w, dw] : adj[n])
          stk.push_back({w, d + dw});
      }
    }

    // Timeline bound: I iterations of the intra-iteration latency sum + slack.
    long latSum = 0;
    for (int v = 0; v < V; ++v)
      latSum += P.ops[v].isLoad ? P.ops[v].slowLat : P.ops[v].staticLat;
    P.T = std::min<long>(90, latSum * P.I + 8);

    // ---- report inputs ----
    llvm::outs() << "== twill-cache-joint for @"
                 << cast<FunctionOpInterface>(funcOp).getName() << " ==\n";
    llvm::outs() << "V=" << V << " I=" << P.I << " W=" << P.W << " S=" << P.S
                 << " T=" << P.T << "\n";
    for (int v = 0; v < V; ++v) {
      llvm::outs() << "  op[" << v << "] ";
      opById[v]->getName().print(llvm::outs());
      if (P.ops[v].isLoad)
        llvm::outs() << "  LOAD line=" << P.ops[v].lineBase << "+i*"
                     << P.ops[v].lineStride << " (fast=" << P.ops[v].fastLat
                     << "/slow=" << P.ops[v].slowLat << ")";
      else
        llvm::outs() << "  lat=" << P.ops[v].staticLat;
      if (P.ops[v].blocking)
        llvm::outs() << "  [BLOCKING]";
      if (P.ops[v].spillcost)
        llvm::outs() << "  spill=" << P.ops[v].spillcost;
      if (!P.ops[v].deps.empty()) {
        llvm::outs() << "  deps={";
        for (auto [u, d] : P.ops[v].deps)
          llvm::outs() << u << "(d=" << d << ") ";
        llvm::outs() << "}";
      }
      llvm::outs() << "\n";
    }

    // 5) solve and print, in two configurations for comparison:
    //      (a) cache model ON   : loads may hit or miss, decided
    //                             jointly with the schedule.
    //      (b) cache model OFF  : original Twill, every load pays its fixed DRAM
    //                             (slow) latency
    auto solveAndPrint = [&](twill::CacheRRIterProblem Q, const char *tag) {
      twill::CacheRRIterResult R = twill::solveTwillCacheJoint(Q);
      llvm::outs() << "\n==== " << tag << " ====\n";
      if (!R.solved) {
        llvm::outs() << "[twill-cache-joint] UNSAT (raise T?)\n";
        return;
      }
      llvm::outs() << "makespan=" << R.makespan << "  misses=" << R.misses
                   << "\n  op-instance   warp Start Done Lat  cache\n";
      for (int i = 0; i < Q.I; ++i)
        for (int v = 0; v < V; ++v) {
          int st = R.start[v][i], lt = R.lat[v][i];
          llvm::outs() << "  v" << v << "/i" << i << "  w" << R.warp[v][i]
                       << "  " << st << "  " << (st + lt) << "  " << lt;
          if (Q.ops[v].isLoad)
            llvm::outs() << "  line=" << Q.line(v, i)
                         << (R.hit[v][i] ? " HIT" : " MISS");
          llvm::outs() << "\n";
        }
    };

    // Per-warp register budget: Σ_{v,i} live[v,i,t]·opw[v,w]·regs(v) ≤ regLimit.
    P.regLimit = 255;
    P.optimizeTimeoutMs = 240000;
    llvm::outs() << "== per-warp register limit = " << P.regLimit
                 << " (regs(v) from tile size) ==\n";
    for (int v = 0; v < V; ++v)
      llvm::outs() << "  regs(op[" << v << "]) = " << P.ops[v].footprint << "\n";

    solveAndPrint(P, "cache model ON (joint hit/miss, feasibility)");

    // Optimal makespan: incremental descent on the feasibility solver
    twill::CacheRRIterProblem Popt = P;
    Popt.minimizeMakespan = true;
    solveAndPrint(Popt, "cache model ON + min makespan (incremental descent)");

    // No-cache baseline
    twill::CacheRRIterProblem P0 = P;
    for (auto &o : P0.ops)
      if (o.isLoad)
        o.fastLat = o.slowLat;
    solveAndPrint(P0, "cache model OFF (original Twill, fixed load latency)");

    twill::CacheRRIterProblem P0opt = P0;
    P0opt.minimizeMakespan = true;
    solveAndPrint(P0opt, "cache model OFF + min makespan (incremental descent)");
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestGlobalMembarPass() {
  PassRegistration<TestGlobalMembarPass>();
  PassRegistration<TestTwillSchedulePass>();
  PassRegistration<TestTwillCacheJointPass>();
}
} // namespace test
} // namespace mlir
