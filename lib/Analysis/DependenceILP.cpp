
#include "triton/Analysis/TwillTypes.h"

#include "CbcModel.hpp"
#include "CoinPackedMatrix.hpp"
#include "OsiClpSolverInterface.hpp"

#include <cstdio>
#include <map>
#include <tuple>
#include <vector>

namespace twill {
//map for both op[v,i,t] and live[v,i,t]

namespace {
class VarMap {
public:
  enum Kind { Op = 0, Live = 1 };

  int alloc(Kind k, int v, int i, int t) {
    auto key = std::make_tuple(static_cast<int>(k), v, i, t);
    auto [it, inserted] = idx_.try_emplace(key, ncols_);
    if (inserted) {
      ncols_++;
      keys_.push_back(key);
    }
    return it->second;
  }

  int find(Kind k, int v, int i, int t) const {
    auto it = idx_.find(std::make_tuple(static_cast<int>(k), v, i, t));
    return it == idx_.end() ? -1 : it->second;
  }

  int size() const { return ncols_; }
  const std::vector<std::tuple<int, int, int, int>> &keys() const {
    return keys_;
  }

private:
  std::map<std::tuple<int, int, int, int>, int> idx_;
  std::vector<std::tuple<int, int, int, int>> keys_;
  int ncols_ = 0;
};
} // namespace


Schedule solveILP(const ILPProblem &P) {
  VarMap V;

  // -------- create op[v,i,t] (COMPLETION: only if t + cycles[v] ≤ L) -------
  // -------- create live[v,i,t] for t ∈ [0, L]                       -------
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      for (int t = 0; t + P.cycles[v] <= P.L; t++)
        V.alloc(VarMap::Op, v, i, t);
      for (int t = 0; t <= P.L; t++)
        V.alloc(VarMap::Live, v, i, t);
    }
  }

  const int ncols = V.size();

  OsiClpSolverInterface solver;
  solver.setLogLevel(0);

  std::vector<double> lb(ncols, 0.0), ub(ncols, 1.0), obj(ncols, 0.0);
  for (const auto &k : V.keys()) {
    auto [kind, v, i, t] = k;
    if (kind == VarMap::Live)
      obj[V.find(VarMap::Live, v, i, t)] = 1.0;
  }


  std::vector<CoinBigIndex> colStart(ncols + 1, 0);
  solver.loadProblem(ncols, /*numrows*/ 0, colStart.data(),
                     /*index*/ nullptr, /*value*/ nullptr, lb.data(), ub.data(),
                     obj.data(), /*rowlb*/ nullptr, /*rowub*/ nullptr);
  for (int c = 0; c < ncols; c++)
    solver.setInteger(c);

  auto addRow = [&](const std::vector<int> &cols,
                    const std::vector<double> &coefs, double lo, double hi) {
    if (cols.empty())
      return;
    solver.addRow(static_cast<int>(cols.size()), cols.data(), coefs.data(), lo,
                  hi);
  };
  const double NEG_INF = -1e30;

  // -------- UNIQUENESS:  Σ_t op[v,i,t] = 1 --------
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      std::vector<int> cols;
      std::vector<double> coefs;
      for (int t = 0; t + P.cycles[v] <= P.L; t++) {
        int c = V.find(VarMap::Op, v, i, t);
        if (c >= 0) {
          cols.push_back(c);
          coefs.push_back(1.0);
        }
      }
      addRow(cols, coefs, 1.0, 1.0);
    }
  }

  // -------- CONSISTENCY:  op[v,0,t] ⇒ op[v,i,t+i·II] --------
  //          encoded as:   op[v,0,t] - op[v,i,t+i·II] ≤ 0
  for (int v = 0; v < P.V; v++) {
    for (int i = 1; i < P.Iu; i++) {
      for (int t = 0; t + P.cycles[v] <= P.L; t++) {
        int c0 = V.find(VarMap::Op, v, 0, t);
        if (c0 < 0)
          continue;
        int tp = t + i * P.II;
        int ci = V.find(VarMap::Op, v, i, tp);
        if (ci < 0) {
          addRow({c0}, {1.0}, 0.0, 0.0);
        } else {
          addRow({c0, ci}, {1.0, -1.0}, NEG_INF, 0.0);
        }
      }
    }
  }

  // -------- DEPENDENCE: op[u,i,t] + op[v,i+δ,t'] ≤ 1  for t'∈[0, t+d) -----
  for (const Edge &e : P.edges) {
    for (int i = 0; i < P.Iu; i++) {
      int j = i + e.delta;
      if (j < 0 || j >= P.Iu)
        continue;
      for (int t = 0; t + P.cycles[e.u] <= P.L; t++) {
        int cu = V.find(VarMap::Op, e.u, i, t);
        if (cu < 0)
          continue;
        for (int tp = 0; tp < t + e.d; tp++) {
          int cv = V.find(VarMap::Op, e.v, j, tp);
          if (cv < 0 || cv == cu)
            continue;
          addRow({cu, cv}, {1.0, 1.0}, NEG_INF, 1.0);
        }
      }
    }
  }

  // -------- CAPACITY: Σ op[v,i,t-c] · RRT[v][f][c] ≤ cap(f) --------
  if (P.F > 0) {
    for (int t = 0; t <= P.L; t++) {
      for (int f = 0; f < P.F; f++) {
        std::vector<int> cols;
        std::vector<double> coefs;
        for (int v = 0; v < P.V; v++) {
          if (v >= (int)P.RRT.size() || f >= (int)P.RRT[v].size())
            continue;
          for (int c = 0; c < (int)P.RRT[v][f].size(); c++) {
            int use = P.RRT[v][f][c];
            if (use == 0)
              continue;
            int ts = t - c;
            if (ts < 0)
              continue;
            for (int i = 0; i < P.Iu; i++) {
              int col = V.find(VarMap::Op, v, i, ts);
              if (col < 0)
                continue;
              cols.push_back(col);
              coefs.push_back(static_cast<double>(use));
            }
          }
        }
        addRow(cols, coefs, NEG_INF, static_cast<double>(P.cap[f]));
      }
    }
  }

  // -------- MEMORY CAPACITY: Σ live[v,i,t] · footprint(v,m) ≤ capacity(m) --
  if (P.M > 0) {
    for (int t = 0; t <= P.L; t++) {
      for (int m = 0; m < P.M; m++) {
        std::vector<int> cols;
        std::vector<double> coefs;
        for (int v = 0; v < P.V; v++) {
          if (v >= (int)P.footprint.size() || m >= (int)P.footprint[v].size())
            continue;
          int fp = P.footprint[v][m];
          if (fp == 0)
            continue;
          for (int i = 0; i < P.Iu; i++) {
            int col = V.find(VarMap::Live, v, i, t);
            if (col < 0)
              continue;
            cols.push_back(col);
            coefs.push_back(static_cast<double>(fp));
          }
        }
        addRow(cols, coefs, NEG_INF, static_cast<double>(P.capacity_mem[m]));
      }
    }
  }

  // -------- INIT: live[v, Iu-1, L] = 1 iff ∃ (v,u,_,δ>0) ∈ E --------
  std::vector<bool> hasLoopCarriedOut(P.V, false);
  for (const Edge &e : P.edges)
    if (e.delta > 0)
      hasLoopCarriedOut[e.u] = true;
  for (int v = 0; v < P.V; v++) {
    int col = V.find(VarMap::Live, v, P.Iu - 1, P.L);
    if (col < 0)
      continue;
    double val = hasLoopCarriedOut[v] ? 1.0 : 0.0;
    addRow({col}, {1.0}, val, val);
  }

  // -------- LIVEPROP-1:  live[v,i,t] + op[v,i,t] + live[v,i,t-1] ≤ 2 ------
  // -------- LIVEPROP-2:  live[v,i,t] - op[v,i,t] - live[v,i,t-1] ≤ 0 ------
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      for (int t = 1; t <= P.L; t++) {
        int cL = V.find(VarMap::Live, v, i, t);
        int cLm = V.find(VarMap::Live, v, i, t - 1);
        int cO = V.find(VarMap::Op, v, i, t);
        if (cL < 0 || cLm < 0)
          continue;
        if (cO >= 0) {
          addRow({cL, cO, cLm}, {1.0, 1.0, 1.0}, NEG_INF, 2.0);    // LP-1
          addRow({cL, cO, cLm}, {1.0, -1.0, -1.0}, NEG_INF, 0.0);  // LP-2
        } else {
          // op[v,i,t] ≡ 0 (doesn't exist); LP-1 trivial, LP-2:
          //   live[t] - live[t-1] ≤ 0
          addRow({cL, cLm}, {1.0, -1.0}, NEG_INF, 0.0);
        }
      }
    }
  }

  // -------- DEADPROP-1 / DEADPROP-2 --------
  //   For each (v,u,_,δ) ∈ E:
  //     DP-1:  op[u,i+δ,t] - live[v,i,t] - live[v,i,t-1] ≤ 0   (per edge)
  //     DP-2:  live[v,i,t-1] - live[v,i,t] - Σ_u op[u,i+δ,t] ≤ 0  (aggregated)
  std::vector<std::vector<std::pair<int, int>>> outs(P.V); // v -> [(u, δ)]
  for (const Edge &e : P.edges)
    outs[e.u].push_back({e.v, e.delta});

  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      for (int t = 1; t <= P.L; t++) {
        int cL = V.find(VarMap::Live, v, i, t);
        int cLm = V.find(VarMap::Live, v, i, t - 1);
        if (cL < 0 || cLm < 0)
          continue;

        // DP-1: per-consumer
        for (auto [u, delta] : outs[v]) {
          int j = i + delta;
          if (j < 0 || j >= P.Iu)
            continue;
          int cU = V.find(VarMap::Op, u, j, t);
          if (cU < 0)
            continue;
          addRow({cU, cL, cLm}, {1.0, -1.0, -1.0}, NEG_INF, 0.0);
        }

        // DP-2: aggregated
        std::vector<int> cols = {cLm, cL};
        std::vector<double> coefs = {1.0, -1.0};
        for (auto [u, delta] : outs[v]) {
          int j = i + delta;
          if (j < 0 || j >= P.Iu)
            continue;
          int cU = V.find(VarMap::Op, u, j, t);
          if (cU < 0)
            continue;
          cols.push_back(cU);
          coefs.push_back(-1.0);
        }
        addRow(cols, coefs, NEG_INF, 0.0);
      }
    }
  }

  printf("[ILP] built: vars=%d rows=%d  → branch-and-bound...\n",
         solver.getNumCols(), solver.getNumRows());
  CbcModel model(solver);
  model.setLogLevel(0);
  model.branchAndBound();

  Schedule out;
  if (!model.isProvenOptimal()) {
    printf("[ILP] no feasible solution (II=%d, L=%d, Iu=%d)\n", P.II, P.L,
           P.Iu);
    return out;
  }
  out.solved = true;
  const double *sol = model.bestSolution();
  for (const auto &k : V.keys()) {
    auto [kind, vid, iid, tid] = k;
    int col = V.find(static_cast<VarMap::Kind>(kind), vid, iid, tid);
    int val = sol[col] > 0.5 ? 1 : 0;
    if (kind == VarMap::Op)
      out.op[{vid, iid, tid}] = val;
    else
      out.live[{vid, iid, tid}] = val;
  }
  return out;
}

} // namespace twill

#ifndef TWILL_NO_MAIN

namespace {

void dumpILP(const twill::ILPProblem &P, const twill::Schedule &S) {
  if (!S.solved) {
    printf("UNSAT / no solution\n");
    return;
  }

  printf("\n== op[v,i,t] = 1 ==\n");
  for (auto &[k, val] : S.op) {
    if (!val)
      continue;
    auto [v, i, t] = k;
    printf("  op[v=%d i=%d t=%d]   (occupies t..t+%d-1)\n", v, i, t,
           P.cycles[v]);
  }

  printf("\n== live[v,i,t] = 1 ==\n");
  for (auto &[k, val] : S.live) {
    if (!val)
      continue;
    auto [v, i, t] = k;
    printf("  live[v=%d i=%d t=%d]\n", v, i, t);
  }

  printf("\n== timeline (rows: v×i, cols: t∈[0..L]) ==\n");
  printf("v  i | ");
  for (int t = 0; t <= P.L; t++)
    printf("%2d ", t);
  printf("\n");
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      printf("%-2d %-2d| ", v, i);
      for (int t = 0; t <= P.L; t++) {
        auto itOp = S.op.find({v, i, t});
        bool isOp = (itOp != S.op.end()) && itOp->second;
        auto itLv = S.live.find({v, i, t});
        bool isLv = (itLv != S.live.end()) && itLv->second;
        const char *mark = isOp ? " X " : (isLv ? " = " : " . ");
        printf("%s", mark);
      }
      printf("\n");
    }
  }
}

} // namespace

int main() {
  using namespace twill;
  setvbuf(stdout, nullptr, _IONBF, 0);

  ILPProblem P;

  P.V = 3;
  P.cycles = {1, 1, 1};
  P.edges = {
      {0, 1, 1, 0},
      {1, 2, 1, 0},
      {2, 0, 1, 1},
  };

  // ----- Functional units (RRT) -----
  // single ALU, cap=1; every op uses ALU for 1 cycle
  P.F = 1;
  P.cap = {1};
  P.RRT.assign(P.V, std::vector<std::vector<int>>(P.F));
  P.RRT[0][0] = {1};
  P.RRT[1][0] = {1};
  P.RRT[2][0] = {1};

  // ----- Memory (footprint) -----
  // single register file, cap=4; every live value costs 1 slot
  P.M = 1;
  P.capacity_mem = {4};
  P.footprint.assign(P.V, std::vector<int>(P.M, 1));

  // ----- Schedule parameters -----
  P.II = 3;
  P.L = 6;
  P.deriveIu();

  printf("[Twill ILP demo]\n");
  printf("  V=%d  |E|=%zu  F=%d  M=%d  II=%d  L=%d  Iu=%d\n", P.V,
         P.edges.size(), P.F, P.M, P.II, P.L, P.Iu);
  printf("  edges (u→v, d, δ):");
  for (auto &e : P.edges)
    printf(" (%d→%d,d=%d,δ=%d)", e.u, e.v, e.d, e.delta);
  printf("\n  cycles:");
  for (int c : P.cycles)
    printf(" %d", c);
  printf("\n  cap(f):");
  for (int c : P.cap)
    printf(" %d", c);
  printf("\n  capacity(m):");
  for (int c : P.capacity_mem)
    printf(" %d", c);
  printf("\n");

  Schedule S = solveILP(P);
  dumpILP(P, S);
  return 0;
}

#endif // TWILL_NO_MAIN
