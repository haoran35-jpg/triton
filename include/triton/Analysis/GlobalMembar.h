#ifndef TRITON_ANALYSIS_GLOBAL_MEMBAR_H
#define TRITON_ANALYSIS_GLOBAL_MEMBAR_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <vector>

namespace mlir::triton {

class AffineSymbolTable {
public:
  explicit AffineSymbolTable(MLIRContext *ctx) : ctx(ctx) {}

  // dim index for `iv` (BlockArgument of scf.for).
  AffineExpr getOrAllocDimFor(Value iv);

  AffineExpr getConstant(int64_t c) const;

  AffineExpr lookupDim(Value iv) const;

  MLIRContext *getContext() const { return ctx; }
  unsigned getNumDims() const { return nextDim; }
  unsigned getNumSymbols() const { return nextSym; }

private:
  MLIRContext *ctx;
  llvm::DenseMap<Value, AffineExpr> dimMap;
  llvm::DenseMap<Value, AffineExpr> symMap;
  unsigned nextDim = 0;
  unsigned nextSym = 0;
};

enum class OffsetState : uint8_t { Bottom, Known, Top };

struct OffsetInfo {
  llvm::SmallSetVector<Value, 4> roots;
  bool isUnknownRoots = false;
  OffsetState offsetState = OffsetState::Bottom;
  AffineExpr offset;

  static OffsetInfo getUnknownRoots() {
    OffsetInfo r;
    r.isUnknownRoots = true;
    return r;
  }
  static OffsetInfo getRoot(Value v) {
    OffsetInfo r;
    r.roots.insert(v);
    return r;
  }
  static OffsetInfo getKnownOffset(AffineExpr e) {
    OffsetInfo r;
    r.offsetState = OffsetState::Known;
    r.offset = e;
    return r;
  }
  static OffsetInfo getTopOffset() {
    OffsetInfo r;
    r.offsetState = OffsetState::Top;
    return r;
  }
  static OffsetInfo getTop() {
    OffsetInfo r;
    r.isUnknownRoots = true;
    r.offsetState = OffsetState::Top;
    return r;
  }

  bool operator==(const OffsetInfo &o) const {
    if (isUnknownRoots != o.isUnknownRoots)
      return false;
    if (!isUnknownRoots && roots != o.roots)
      return false;
    if (offsetState != o.offsetState)
      return false;
    if (offsetState == OffsetState::Known && offset != o.offset)
      return false;
    return true;
  }

  static OffsetInfo join(const OffsetInfo &a, const OffsetInfo &b) {
    OffsetInfo r;
    if (a.isUnknownRoots || b.isUnknownRoots) {
      r.isUnknownRoots = true;
    } else {
      r.roots = a.roots;
      for (Value v : b.roots)
        r.roots.insert(v);
    }
    if (a.offsetState == OffsetState::Top ||
        b.offsetState == OffsetState::Top) {
      r.offsetState = OffsetState::Top;
    } else if (a.offsetState == OffsetState::Bottom) {
      r.offsetState = b.offsetState;
      r.offset = b.offset;
    } else if (b.offsetState == OffsetState::Bottom) {
      r.offsetState = a.offsetState;
      r.offset = a.offset;
    } else {
      if (a.offset == b.offset) {
        r.offsetState = OffsetState::Known;
        r.offset = a.offset;
      } else {
        r.offsetState = OffsetState::Top;
      }
    }
    return r;
  }

  void print(raw_ostream &os) const {
    os << "roots=";
    if (isUnknownRoots) {
      os << "<top>";
    } else {
      os << "{";
      llvm::interleaveComma(roots, os,
                            [&](Value v) { v.printAsOperand(os, {}); });
      os << "}";
    }
    os << " off=";
    switch (offsetState) {
    case OffsetState::Bottom:
      os << "<bot>";
      break;
    case OffsetState::Top:
      os << "<top>";
      break;
    case OffsetState::Known:
      offset.print(os);
      break;
    }
  }
};

using OffsetLattice = mlir::dataflow::Lattice<OffsetInfo>;

class RootOffsetAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<OffsetLattice> {
public:
  RootOffsetAnalysis(DataFlowSolver &solver, AffineSymbolTable *syms)
      : SparseForwardDataFlowAnalysis(solver), syms(syms) {}

  using SparseForwardDataFlowAnalysis::getLatticeElement;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<const OffsetLattice *> operands,
                 ArrayRef<OffsetLattice *> results) override;

  void setToEntryState(OffsetLattice *lat) override;

private:
  AffineSymbolTable *syms;
};

struct LiftedMap {
  llvm::DenseMap<Value, AffineExpr> table;
};

void runLoopLifting(Operation *root, RootOffsetAnalysis &analysis,
                    AffineSymbolTable &syms, LiftedMap &out);

AffineExpr getLiftedOffset(Value v, RootOffsetAnalysis &analysis,
                           AffineSymbolTable &syms, LiftedMap &lifted);

const OffsetInfo *getOffsetInfo(Value v, RootOffsetAnalysis &analysis);

enum class DepKind { No, SameIter, LoopCarried, Maybe };
struct DepResult {
  DepKind kind = DepKind::Maybe;
  int64_t delta = 0;
  AffineExpr diff;
};

DepResult mayDepend(Operation *A, unsigned aPtrIdx, Operation *B,
                    unsigned bPtrIdx, RootOffsetAnalysis &analysis,
                    AffineSymbolTable &syms, LiftedMap &lifted);

bool isAsyncIssuer(Operation *op);

bool edgeIsBlocking(Operation *u, Operation *v);

enum class EdgeKind : uint8_t {
  SSA,             // direct SSA def-use
  LOOPSSA,     // scf.for yield → iter_arg user (δ ≥ 1)
  MemSameIter,     // mem alias, same iter (dist = 0)
  MemLoopCarried,  // mem alias, cross iter (dist = δ)
  MemMaybe,        // mem alias could not be proven absent
  BLOCKING,
};

const char *edgeKindName(EdgeKind k);

struct DepEdge {
  Operation *u = nullptr;
  Operation *v = nullptr;
  int dist = 0;         // iteration distance δ
  unsigned lat = 0;     // cycle latency
  bool blocking = false;
  EdgeKind kind = EdgeKind::SSA;
};

unsigned defaultLatency(Operation *op);

std::vector<DepEdge> collectEdges(Operation *funcOp);

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_GLOBAL_MEMBAR_H
