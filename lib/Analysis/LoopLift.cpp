#include "triton/Analysis/GlobalMembar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

#define DEBUG_TYPE "global-membar-lift"

namespace mlir::triton {
enum mem_ops {
  //global mem to shared mem
  // arith ops accessing
};
enum def_use_blocking {

};

static bool isPtrLikeT(Type t) {
  if (isa<triton::PointerType>(t))
    return true;
  if (auto rt = dyn_cast<RankedTensorType>(t))
    return isa<triton::PointerType>(rt.getElementType());
  return false;
}

static AffineExpr knownOffset(const OffsetLattice *lat) {
  if (!lat)
    return AffineExpr();
  const auto &info = lat->getValue();
  if (info.offsetState != OffsetState::Known)
    return AffineExpr();
  return info.offset;
}

static AffineExpr getOffsetFromAnalysis(Value v, RootOffsetAnalysis &analysis) {
  return knownOffset(analysis.getLatticeElement(v));
}

static AffineExpr computeDelta(Value iter_arg, Value yielded,
                               RootOffsetAnalysis &analysis,
                               AffineSymbolTable &syms, LiftedMap &lifted,
                               MLIRContext *ctx) {
  if (yielded == iter_arg)
    return mlir::getAffineConstantExpr(0, ctx);

  Operation *def = yielded.getDefiningOp();
  if (!def)
    return AffineExpr();

  if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
    AffineExpr base =
        computeDelta(iter_arg, addptr.getPtr(), analysis, syms, lifted, ctx);
    if (!base)
      return AffineExpr();
    AffineExpr off = getOffsetFromAnalysis(addptr.getOffset(), analysis);
    if (!off)
      return AffineExpr();
    return base + off;
  }
  if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
          triton::ReshapeOp, triton::BitcastOp>(def)) {
    return computeDelta(iter_arg, def->getOperand(0), analysis, syms, lifted,
                        ctx);
  }
  // Inner scf.for result: try to express its offset wrt outer iter_arg using
  // the lifted map (inner lifting must have populated `lifted[result]`).
  if (auto innerFor = dyn_cast<scf::ForOp>(def)) {
    if (auto it = lifted.table.find(yielded); it != lifted.table.end()) {
      AffineExpr resultOff = it->second;
      AffineExpr baseOff = getLiftedOffset(iter_arg, analysis, syms, lifted);
      if (resultOff && baseOff)
        return resultOff - baseOff;
    }
    return AffineExpr();
  }
  return AffineExpr();
}


void runLoopLifting(Operation *root, RootOffsetAnalysis &analysis,
                    AffineSymbolTable &syms, LiftedMap &out) {
  MLIRContext *ctx = root->getContext();

  SmallVector<scf::ForOp, 8> fors;
  root->walk<WalkOrder::PreOrder>([&](scf::ForOp f) { fors.push_back(f); });

  for (scf::ForOp forOp : fors) {
    AffineExpr dim_iv = syms.getOrAllocDimFor(forOp.getInductionVar());

    Operation *yieldOp = forOp.getBody()->getTerminator();
    auto inits = forOp.getInits();
    auto iterArgs = forOp.getRegionIterArgs();

    for (auto [init, iter, yielded] :
         llvm::zip(inits, iterArgs, yieldOp->getOperands())) {
      if (!isPtrLikeT(iter.getType()))
        continue; // only ptr iter_args participate in offset analysis


      AffineExpr initOff = getLiftedOffset(init, analysis, syms, out);
      if (!initOff)
        initOff = syms.getOrAllocSymFor(init);

      AffineExpr delta = computeDelta(iter, yielded, analysis, syms, out, ctx);
      if (!delta) {
        AffineExpr fallback = syms.getOrAllocSymFor(iter);
        out.table.try_emplace(iter, fallback);
        LLVM_DEBUG(llvm::dbgs()
                   << "loop-lift: Δ unknown, fallback sym for " << iter
                   << "\n");
        continue;
      }
      // lifted(iter_arg) = init.offset + dim_iv * Δ
      AffineExpr lifted = initOff + dim_iv * delta;
      out.table.try_emplace(iter, lifted);
      LLVM_DEBUG({
        llvm::dbgs() << "loop-lift: " << iter << " -> ";
        lifted.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });

      unsigned idx = std::distance(iterArgs.begin(), llvm::find(iterArgs, iter));
      if (idx < forOp.getNumResults()) {
        Value res = forOp.getResult(idx);
        out.table.try_emplace(res, lifted + delta);
      }
    }
  }
}

AffineExpr getLiftedOffset(Value v, RootOffsetAnalysis &analysis,
                           AffineSymbolTable &syms, LiftedMap &lifted) {
  if (auto it = lifted.table.find(v); it != lifted.table.end())
    return it->second;

  if (AffineExpr e = getOffsetFromAnalysis(v, analysis))
    return e;

  Operation *def = v.getDefiningOp();
  if (!def)
    return AffineExpr();

  if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
    AffineExpr base =
        getLiftedOffset(addptr.getPtr(), analysis, syms, lifted);
    if (!base)
      return AffineExpr();
    AffineExpr off = getOffsetFromAnalysis(addptr.getOffset(), analysis);
    if (!off)
      return AffineExpr();
    AffineExpr result = base + off;
    lifted.table.try_emplace(v, result);
    return result;
  }
  if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
          triton::ReshapeOp, triton::BitcastOp>(def)) {
    AffineExpr inner =
        getLiftedOffset(def->getOperand(0), analysis, syms, lifted);
    if (inner)
      lifted.table.try_emplace(v, inner);
    return inner;
  }
  return AffineExpr();
}

const OffsetInfo *getOffsetInfo(Value v, RootOffsetAnalysis &analysis) {
  const OffsetLattice *lat = analysis.getLatticeElement(v);
  return lat ? &lat->getValue() : nullptr;
}

namespace {
struct LinearForm {
  int64_t alpha = 0;       // coefficient of dim
  int64_t beta = 0;        // constant term
  unsigned dimPos = 0;     // valid iff alpha != 0
  bool hasDim = false;     // false iff alpha == 0
  bool hasSymbol = false;
};

// Recursive helper.  Accumulates (alpha for each dim, beta).
struct LinearAccum {
  llvm::SmallDenseMap<unsigned, int64_t> dimCoeffs; // only one dim is supported
  // the one dimension 4 * d0 where 4 is mapped to alpha, d0 is mapped to dimPos
  //beta is mapped to beta
  int64_t beta = 0;
  bool hasSymbol = false;
  bool nonLinear = false;
};

static void collectLinear(AffineExpr e, int64_t scale, LinearAccum &acc) {
  if (acc.nonLinear) return;
  if (auto c = dyn_cast<AffineConstantExpr>(e)) {
    // 4
    acc.beta += scale * c.getValue();
    return;
  }
  if (auto d = dyn_cast<AffineDimExpr>(e)) {
    // 4 * d0
    acc.dimCoeffs[d.getPosition()] += scale;
    return;
  }
  if (auto s = dyn_cast<AffineSymbolExpr>(e)) {
    // d0, init
    acc.hasSymbol = true; // we don't model symbols in the linear form
    return;
  }
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin) { acc.nonLinear = true; return; }
  switch (bin.getKind()) {
  case AffineExprKind::Add:
  // if add, both terms keep going. Hereitage the scale.
    collectLinear(bin.getLHS(), scale, acc);
    collectLinear(bin.getRHS(), scale, acc);
    return;
  case AffineExprKind::Mul: {
    // if mul,
    auto lc = dyn_cast<AffineConstantExpr>(bin.getLHS());
    auto rc = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (lc) {
      // 4 * d0
      // d0 keeps recursing, 4 timed with scale becomes the true scale
      collectLinear(bin.getRHS(), scale * lc.getValue(), acc);
      return;
    }
    if (rc) {
      // d0 * 4
      //d0 keeps recursing, 4 timed with scale becomes the true scale
      collectLinear(bin.getLHS(), scale * rc.getValue(), acc);
      return;
    }
    acc.nonLinear = true;
    return;
  }
  default:
    acc.nonLinear = true;
    return;
  }
}

static std::optional<LinearForm> matchLinearOverDim(AffineExpr expr) {
  LinearAccum acc;
  collectLinear(expr, 1, acc);
  if (acc.nonLinear)
    // not supporting non linear expression
    return std::nullopt;

  LinearForm out;
  out.beta = acc.beta;
  out.hasSymbol = acc.hasSymbol;

  llvm::SmallVector<std::pair<unsigned, int64_t>> nz;
  for (auto &kv : acc.dimCoeffs)
    if (kv.second != 0)
      nz.push_back({kv.first, kv.second});
  if (nz.size() > 1)
    return std::nullopt;
  if (!nz.empty()) {
    // here for nz size >= 2 has been fitered out
    out.alpha = nz[0].second;
    out.dimPos = nz[0].first;
    out.hasDim = true;
  }
  return out;
}
} // namespace


static bool rootsDisjoint(const OffsetInfo &a, const OffsetInfo &b) {
  if (a.isUnknownRoots || b.isUnknownRoots)
    return false;
  for (Value v : a.roots)
    if (b.roots.contains(v))
      return false;
  return true;
}

static int64_t tileWidthOf(Value v) {
  Type t = v.getType();
  if (auto rt = dyn_cast<RankedTensorType>(t)) {
    int64_t n = 1;
    for (int64_t d : rt.getShape())
      n *= d;
    return std::max<int64_t>(n, 1);
  }
  return 1;
}

// Find the smallest non-negative integer δ satisfying |β − α·δ| < W.  Returns
// -1 when no such δ exists.  Models the interval-overlap criterion:
//   A covers   [α·k_A + β_A, α·k_A + β_A + W)
//   B covers   [α·k_B + β_B, α·k_B + β_B + W)
//   They alias iff |β_diff − α·δ| ≤ W − 1     (δ = k_A − k_B, β_diff = β_B − β_A)
static int64_t smallestOverlapDelta(int64_t alpha, int64_t beta, int64_t W) {
  if (W <= 0)
    return -1;
  if (alpha == 0) {
    // Loop-invariant addresses: dep iff intervals overlap regardless of iter.
    return std::abs(beta) <= W - 1 ? 0 : -1;
  }
  // Rewrite as α·δ ∈ [β − W + 1, β + W − 1].
  int64_t lo = beta - W + 1;
  int64_t hi = beta + W - 1;
  if (alpha < 0) {
    // Solving for δ flips the inequality.
    alpha = -alpha;
    int64_t tmp = lo;
    lo = -hi;
    hi = -tmp;
  }
  // α > 0.  δ ∈ [⌈lo/α⌉, ⌊hi/α⌋].
  int64_t dMin = llvm::divideCeilSigned(lo, alpha);
  int64_t dMax = llvm::divideFloorSigned(hi, alpha);
  if (dMin < 0)
    dMin = 0;
  if (dMin > dMax)
    return -1;
  return dMin;
}
/*
maybe: cannot get offsetinfo for any of the two pointers. Any root unknown. cannot get ofsetinfo
for any of the two pointer offset. Has symbo. Has dim.
no: roots disjoint. delta = 0.
loop carrier: delta > 0.
same iter: delta = 0.
*/
DepResult mayDepend(Operation *A, unsigned aPtrIdx, Operation *B,
                    unsigned bPtrIdx, RootOffsetAnalysis &analysis,
                    AffineSymbolTable &syms, LiftedMap &lifted) {
  Value pa = A->getOperand(aPtrIdx);
  Value pb = B->getOperand(bPtrIdx);

  const OffsetInfo *ia = getOffsetInfo(pa, analysis);
  const OffsetInfo *ib = getOffsetInfo(pb, analysis);
  if (!ia || !ib)
    return {DepKind::Maybe};

  if (rootsDisjoint(*ia, *ib))
    return {DepKind::No};
  if (ia->isUnknownRoots || ib->isUnknownRoots)
    return {DepKind::Maybe};

  AffineExpr oa = getLiftedOffset(pa, analysis, syms, lifted);
  AffineExpr ob = getLiftedOffset(pb, analysis, syms, lifted);
  if (!oa || !ob)
    return {DepKind::Maybe};

  unsigned numDims = syms.getNumDims();
  unsigned numSyms = syms.getNumSymbols();
  AffineExpr diff =
      mlir::simplifyAffineExpr(ob - oa, numDims, numSyms);

  DepResult r;
  r.diff = diff;

  auto ld = matchLinearOverDim(diff);
  if (!ld || ld->hasSymbol) {
    r.kind = DepKind::Maybe;
    return r;
  }
  if (ld->hasDim) {
    r.kind = DepKind::Maybe;
    return r;
  }

  // diff is a pure constant β.
  int64_t betaDiff = ld->beta;
  int64_t W = std::min(tileWidthOf(pa), tileWidthOf(pb));

  // Per-iter slope: take it from oa (matchLinearOverDim allows trailing syms).
  auto la = matchLinearOverDim(oa);
  int64_t alpha = (la && la->hasDim) ? la->alpha : 0;

  int64_t delta = smallestOverlapDelta(alpha, betaDiff, W);
  if (delta < 0) {
    r.kind = DepKind::No;
    return r;
  }
  if (delta == 0) {
    r.kind = DepKind::SameIter;
    r.delta = 0;
    return r;
  }
  r.kind = DepKind::LoopCarried;
  r.delta = delta;
  return r;
}


bool isBlockingOp(Operation *op) {
  //blocking ops incluldes
  // for Amphere: commit: cp.async.commit_group
  // for Hopper: mbarrier.wait
  // for Wgmma Hopper: wgmma.wait_group.sync.aligned N
  // for Blackwell: tcgen05.commit_group

  //for commit group, we count all the async above the blocking op as
  // ops being dependent. The first op below blocking op as the one dependent.
  // for mbarrier,in TMA, we count the op above blockign op as the one being
  // dependent, the one below as the one dependent.
  if(isa<triton::nvidia_gpu::TCGen5CommitOp, triton::gpu::AsyncCommitGroupOp,
    triton::nvidia_gpu::WarpGroupDotWaitOp, riton::nvidia_gpu::WaitBarrierOp>) {
      return true;
    }

}


static vector<Operation *> getPrevInBlock(Operation *op) {
  if (!op || !op->getBlock())
    return nullptr;
  auto it = Block::iterator(op);
  if (it == op->getBlock()->begin())
    return nullptr;
  vector<Operation *> prevAsyncOps;
  while(it != op->getBlock()->begin()) {
    if(isa<triton::nvidia_gpu::TCGen5CommitOp>) {
      if(isa<triton::nvidia_gpu::TCGen5MMAOp>) {
        prevAsyncOps.push_back(it);
        it--;
      }
      else break;
    } else if (isa<triton::nvidia_gpu::WarpGroupDotWaitOp>) {
      if(isa<triton::nvidia_gpu::WarpGroupDotOp>) {
        prevAsyncOps.push_back(it);
        it--;
      }
      else break;
    } else if(isa<triton::nvidia_gpu::WaitBarrierOp>) {
      if(isa<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>) {
        prevAsyncOps.push_back(it);
        it--;
      }
      else break;

    } else if(isa<triton::gpu::AsyncCommitGroupOp>) {
      if(isa<triton::gpu::AsyncCopyGlobalToLocalOp>) {
        prevAsyncOps.push_back(it);
        it--;
      }
      else break;
    }
  }
  return prevAsyncOps;
}



// collection of all blocking realted ops
bool isAsyncIssuer(Operation *op) {
  // TritonGPU cp.async family (Ampere+).
  if (isa<triton::gpu::AsyncCopyGlobalToLocalOp,
          triton::gpu::AsyncCommitGroupOp>(op))
    return true;

  // Hopper TMA family (cp.async.bulk).
  if (isa<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp,
          triton::nvidia_gpu::AsyncTMAGatherOp,
          triton::nvidia_gpu::AsyncTMAScatterOp,
          triton::nvidia_gpu::AsyncTMAReduceOp>(op))
    return true;

  // Hopper warp-group MMA: runs in background, needs wgmma.wait_group.
  if (isa<triton::nvidia_gpu::WarpGroupDotOp>(op))
    return true;

  // sm100 tcgen05 MMA family.
  if (isa<triton::nvidia_gpu::TCGen5MMAOp,
          triton::nvidia_gpu::TCGen5MMAScaledOp,
          triton::nvidia_gpu::TCGen5CommitOp>(op))
    return true;

  return false;
}



static Operation *getNextInBlock(Operation *op) {
  if (!op || !op->getBlock())
    return nullptr;
  auto it = Block::iterator(op);
  if (it == op->getBlock()->begin())
    return nullptr;
  if(isa<triton::nvidia_gpu::TCGen5CommitOp>) {
    it++;
    return it++;
  } else if (isa<triton::nvidia_gpu::WarpGroupDotWaitOp>) {
    return it++;
  } else if(isa<triton::nvidia_gpu::WaitBarrierOp>) {
    return it++;

  } else if(isa<triton::gpu::AsyncCommitGroupOp>) {
    it++;
    return it++;
  }
  return nullptr;
}


auto pushBlockingBridgeEdge = [&](std::vector<DepEdge> &edges, Operation *u, Operation *v) {
  if (!u || !v)
    return;
  DepEdge e;
  e.u = u;
  e.v = v;
  e.dist = 0;
  e.lat = defaultLatency(u);
  e.blocking = true;
  e.kind = EdgeKind::BLOCKING;
  edges.push_back(e);
};

bool edgeIsBlocking(Operation *u, SmallVector<DepEdge> &edges) {
  if(isBlockingOp(u)) {
    vector<Operation *> prevAsyncOps = getPrevInBlock(u);
    Operation *next = getNextInBlock(u);
    for(auto prev_op : prevAsyncOps) {
      pushBlockingBridgeEdge(edges,prev_op, next);
    }
    rturn true;
  }
  return false;
}

const char *edgeKindName(EdgeKind k) {
  switch (k) {
  case EdgeKind::SSA:             return "ssa";
  case EdgeKind::LOOPSSA:     return "loop-ssa";
  case EdgeKind::MemSameIter:     return "mem-same-iter";
  case EdgeKind::MemLoopCarried:  return "mem-loop-carried";
  case EdgeKind::MemMaybe:        return "mem-maybe";
  }
  return "?";
}

unsigned defaultLatency(Operation *op) {
  if (!op)
    return 1;
  if (isa<triton::LoadOp>(op))      return 400;
  if (isa<triton::StoreOp>(op))     return 8;
  if (isa<triton::AtomicRMWOp>(op)) return 200;
  if (isa<triton::AtomicCASOp>(op)) return 200;
  if (isa<triton::gpu::AsyncCopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(op))
    return 400;
  if (isa<triton::nvidia_gpu::WarpGroupDotOp>(op))
    return 64;
  if (isa<triton::nvidia_gpu::TCGen5MMAOp,
          triton::nvidia_gpu::TCGen5MMAScaledOp>(op))
    return 64;
  return 1;
}

// Ops we ask RootOffsetAnalysis about (operand 0 is a global ptr we can model).
static bool isMemAliasOp(Operation *op) {
  return isa<triton::LoadOp, triton::StoreOp,
             triton::AtomicRMWOp, triton::AtomicCASOp,
             triton::gpu::AsyncCopyGlobalToLocalOp,
             triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
             triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(op);
}

std::vector<DepEdge> collectEdges(Operation *funcOp) {
  std::vector<DepEdge> edges;
  if (!funcOp)
    return edges;
  MLIRContext *ctx = funcOp->getContext();

  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  AffineSymbolTable syms(ctx);
  auto *analysis = solver->load<RootOffsetAnalysis>(&syms);
  if (failed(solver->initializeAndRun(funcOp)))
    return edges;
  LiftedMap lifted;
  runLoopLifting(funcOp, *analysis, syms, lifted);

  auto pushEdge = [&](Operation *u, Operation *v, int dist, EdgeKind kind) {
    if (!u || !v)
      return;
    DepEdge e;
    e.u = u;
    e.v = v;
    e.dist = dist;
    e.lat = defaultLatency(u);
    e.blocking = edgeIsBlocking(u, v);
    e.kind = kind;
    edges.push_back(e);
  };
 // def-use 1: out of for loop
  funcOp->walk([&](Operation *consumer) {
    for (Value operand : consumer->getOperands()) {
      if (Operation *producer = operand.getDefiningOp())
        pushEdge(producer, consumer, /*dist=*/0, EdgeKind::SSA);
    }
  });

  // def-use 2: in for loop
  funcOp->walk([&](scf::ForOp forOp) {
    Operation *term = forOp.getBody()->getTerminator();
    auto yieldOp = dyn_cast<scf::YieldOp>(term);
    if (!yieldOp)
      return;
    auto iterArgs = forOp.getRegionIterArgs();
    auto yieldVals = yieldOp.getOperands();
    for (auto [iterArg, yieldVal] : llvm::zip(iterArgs, yieldVals)) {
      Operation *producer = yieldVal.getDefiningOp();
      if (!producer)
        continue;
      for (Operation *user : iterArg.getUsers())
        pushEdge(producer, user, /*dist=*/1, EdgeKind::LOOPSSA);
    }
  });

  SmallVector<Operation *> memOps;
  funcOp->walk([&](Operation *op) {
    if (isMemAliasOp(op))
      memOps.push_back(op);
  });
  for (size_t i = 0; i < memOps.size(); ++i) {
    for (size_t j = i + 1; j < memOps.size(); ++j) {
      Operation *A = memOps[i], *B = memOps[j];
      auto dep = mayDepend(A, /*aPtrIdx=*/0, B, /*bPtrIdx=*/0, *analysis, syms,
                           lifted);
      if (dep.kind == DepKind::No)
        continue;
      // Order in program order (u → v).
      Operation *u = A, *v = B;
      if (A->getBlock() == B->getBlock() && !A->isBeforeInBlock(B))
        std::swap(u, v);
      EdgeKind kind = EdgeKind::MemMaybe;
      switch (dep.kind) {
      case DepKind::SameIter:    kind = EdgeKind::MemSameIter; break;
      case DepKind::LoopCarried: kind = EdgeKind::MemLoopCarried; break;
      case DepKind::Maybe:       kind = EdgeKind::MemMaybe; break;
      case DepKind::No:          continue;
      }
      pushEdge(u, v, dep.delta, kind);
    }
  }

  funcOp->walk([&](Operation *op) {
    if(isBlockingOp(op)) {
      edgeIsBlocking(op, edges);
    }
  });

  return edges;
}

} // namespace mlir::triton
