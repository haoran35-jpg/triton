#include "triton/Analysis/GlobalMembar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::triton {

AffineExpr AffineSymbolTable::getOrAllocDimFor(Value iv) {
  if (auto it = dimMap.find(iv); it != dimMap.end())
    return it->second;
  AffineExpr e = mlir::getAffineDimExpr(nextDim++, ctx);
  dimMap.try_emplace(iv, e);
  return e;
}
AffineExpr AffineSymbolTable::getOrAllocSymFor(Value v) {
  if (auto it = symMap.find(v); it != symMap.end())
    return it->second;
  AffineExpr e = mlir::getAffineSymbolExpr(nextSym++, ctx);
  symMap.try_emplace(v, e);
  return e;
}
AffineExpr AffineSymbolTable::getConstant(int64_t c) const {
  return mlir::getAffineConstantExpr(c, ctx);
}
AffineExpr AffineSymbolTable::lookupDim(Value iv) const {
  auto it = dimMap.find(iv);
  return it == dimMap.end() ? AffineExpr() : it->second;
}


static bool isPtrLikeType(Type t) {
  if (isa<triton::PointerType>(t))
    return true;
  if (auto rt = dyn_cast<RankedTensorType>(t))
    return isa<triton::PointerType>(rt.getElementType());
  return false;
}
static bool isIntLikeType(Type t) {
  if (t.isIntOrIndex())
    return true;
  if (auto rt = dyn_cast<RankedTensorType>(t))
    return rt.getElementType().isIntOrIndex();
  return false;
}

static std::optional<int64_t> tryGetConstInt(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto i = dyn_cast<IntegerAttr>(cst.getValue()))
      return i.getInt();
    if (auto d = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
      if (d.isSplat())
        return d.getSplatValue<APInt>().getSExtValue();
    }
  }
  return std::nullopt;
}

LogicalResult RootOffsetAnalysis::visitOperation(
    Operation *op, ArrayRef<const OffsetLattice *> operands,
    ArrayRef<OffsetLattice *> results) {
  if (results.empty())
    return success();

  if (op->getNumResults() != 1) {
    for (OffsetLattice *r : results)
      propagateIfChanged(r, r->join(OffsetInfo::getTop()));
    return success();
  }

  OffsetLattice *out = results[0];
  Type rt = op->getResult(0).getType();
  MLIRContext *ctx = op->getContext();

  auto cstAffine = [&](int64_t c) {
    return mlir::getAffineConstantExpr(c, ctx);
  };
  auto symAffine = [&](Value v) { return syms->getOrAllocSymFor(v); };

  auto opLat = [&](unsigned i) -> OffsetInfo {
    Value v = dyn_cast<Value>(operands[i]->getAnchor());
    if (v) {
      if (auto bb = dyn_cast<BlockArgument>(v)) {
        if (auto forOp = dyn_cast<scf::ForOp>(bb.getOwner()->getParentOp())) {
          if (bb.getArgNumber() == 0) {
            // induction variable: must be Known(dim_iv).
            OffsetInfo info = operands[i]->getValue();
            info.offsetState = OffsetState::Known;
            info.offset = syms->getOrAllocDimFor(v);
            return info;
          }
        }
      }
    }
    return operands[i]->getValue();
  };


  if (isPtrLikeType(rt)) {
    OffsetInfo info;
    if (auto addptr = dyn_cast<triton::AddPtrOp>(op)) {
      OffsetInfo p = opLat(0);
      OffsetInfo o = opLat(1);
      info.isUnknownRoots = p.isUnknownRoots;
      info.roots = p.roots;
      if (p.offsetState == OffsetState::Top ||
          o.offsetState == OffsetState::Top) {
        info.offsetState = OffsetState::Top;
      } else if (p.offsetState == OffsetState::Bottom ||
                 o.offsetState == OffsetState::Bottom) {
        info.offsetState = OffsetState::Bottom;
      } else {
        info.offsetState = OffsetState::Known;
        info.offset = p.offset + o.offset;
      }
    } else if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
                   triton::ReshapeOp, triton::BitcastOp,
                   triton::MakeTensorDescOp>(op)) {
      info = opLat(0);
    } else if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      info = OffsetInfo::join(opLat(1), opLat(2));
    } else if (isa<triton::IntToPtrOp>(op)) {
      info = OffsetInfo::getTop();
    } else {
      info = OffsetInfo::getTop();
    }
    propagateIfChanged(out, out->join(info));
    return success();
  }

  if (isIntLikeType(rt)) {
    OffsetInfo info;
    info.isUnknownRoots = false; // n/a

    auto setKnown = [&](AffineExpr e) {
      info.offsetState = OffsetState::Known;
      info.offset = e;
    };
    auto setTop = [&]() { info.offsetState = OffsetState::Top; };
    auto fwd = [&](unsigned i) {
      const OffsetInfo &x = opLat(i);
      info.offsetState = x.offsetState;
      info.offset = x.offset;
    };

    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      if (auto c = tryGetConstInt(cst.getResult()))
        setKnown(cstAffine(*c));
      else
        setTop();
    } else if (auto addi = dyn_cast<arith::AddIOp>(op)) {
      OffsetInfo a = opLat(0);
      OffsetInfo b = opLat(1);
      if (a.offsetState == OffsetState::Top ||
          b.offsetState == OffsetState::Top)
        setTop();
      else if (a.offsetState == OffsetState::Bottom ||
               b.offsetState == OffsetState::Bottom)
        ;
      else
        setKnown(a.offset + b.offset);
    } else if (auto subi = dyn_cast<arith::SubIOp>(op)) {
      OffsetInfo a = opLat(0);
      OffsetInfo b = opLat(1);
      if (a.offsetState == OffsetState::Top ||
          b.offsetState == OffsetState::Top)
        setTop();
      else if (a.offsetState == OffsetState::Bottom ||
               b.offsetState == OffsetState::Bottom)
        ;
      else
        setKnown(a.offset - b.offset);
    } else if (auto muli = dyn_cast<arith::MulIOp>(op)) {
      auto lhsC = tryGetConstInt(op->getOperand(0));
      auto rhsC = tryGetConstInt(op->getOperand(1));
      OffsetInfo a = opLat(0);
      OffsetInfo b = opLat(1);
      if (lhsC && b.offsetState == OffsetState::Known)
        setKnown(b.offset * cstAffine(*lhsC));
      else if (rhsC && a.offsetState == OffsetState::Known)
        setKnown(a.offset * cstAffine(*rhsC));
      else if (lhsC && rhsC)
        setKnown(cstAffine(*lhsC * *rhsC));
      else
        setTop();
    } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp,
                   arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
      fwd(0);
    } else if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
                   triton::ReshapeOp>(op)) {
      fwd(0);
    } else if (isa<triton::MakeRangeOp>(op)) {
      auto mk = cast<triton::MakeRangeOp>(op);
      AffineExpr lane = symAffine(mk.getResult());
      setKnown(lane + cstAffine(mk.getStart()));
    } else {

      setKnown(symAffine(op->getResult(0)));
    }
    propagateIfChanged(out, out->join(info));
    return success();
  }

  return success();
}

void RootOffsetAnalysis::setToEntryState(OffsetLattice *lat) {
  OffsetInfo info;
  Value v = dyn_cast<Value>(lat->getAnchor());
  if (!v) {
    propagateIfChanged(lat, lat->join(info));
    return;
  }
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg) {
    propagateIfChanged(lat, lat->join(info));
    return;
  }
  // get the parent of block arg:forop or function op
  Operation *parent = arg.getOwner()->getParentOp();

  if (isa_and_nonnull<FunctionOpInterface>(parent)) {
    // only functoion parameters inside
    Type t = arg.getType();
    if (isPtrLikeType(t)) {
      info = OffsetInfo::getRoot(arg);
      info.offsetState = OffsetState::Known;
      //t.getContext(): get MLIR context by taking advantage of t
      //also, all function parameters are root themselves
      info.offset = mlir::getAffineConstantExpr(0, t.getContext());
    } else if (isIntLikeType(t)) {
      //similar to iterator and MakeRangeOp, set a symbol for the int parameter in
      //signature. becaus statistically analysis can not know actual value of function
      //int paramter.
      info.offsetState = OffsetState::Known;
      info.offset = syms->getOrAllocSymFor(arg);
    }
  }
  propagateIfChanged(lat, lat->join(info));
}

} // namespace mlir::triton
