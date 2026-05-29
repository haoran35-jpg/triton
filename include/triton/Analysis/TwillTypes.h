//===- TwillTypes.h - Shared types for Twill modulo scheduler -------------===//
//
// Common data types used by both Step-1 ILP (`DependenceILP.cpp`) and
// Step-2 SMT (`DependenceSMT.cpp`).  Putting them here lets the SMT demo
// directly consume the `Schedule` produced by the ILP demo, instead of
// hand-copying the values.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_TWILL_TYPES_H
#define TRITON_ANALYSIS_TWILL_TYPES_H

#include <map>
#include <tuple>
#include <vector>

namespace twill {

//===----------------------------------------------------------------------===//
// Dependence edge:  e = (u, v, d, δ)
//   u, v  : producer / consumer op id
//   d     : cycle latency      (clock-cycle delay producer → consumer)
//   delta : iteration latency  (iteration distance, δ ≥ 0)
//===----------------------------------------------------------------------===//
struct Edge {
  int u;
  int v;
  int d;
  int delta;
};

// Step-1 ILP result (also serves as a Step-2 SMT input)
struct Schedule {
  std::map<std::tuple<int, int, int>, int> op;
  std::map<std::tuple<int, int, int>, int> live;
  bool solved = false;
};


// Step-1 ILP input

struct ILPProblem {
  // ---- dependence graph ----
  int V = 0;
  std::vector<Edge> edges;

  // ---- per-op statistics ----
  std::vector<int> cycles;                          // cycles(v)
  std::vector<std::vector<std::vector<int>>> RRT;   // RRT(v)[f][c]
  std::vector<std::vector<int>> footprint;          // footprint(v, m)

  // ---- hardware capacities ----
  int F = 0;                       // number of functional units
  int M = 0;                       // number of memory classes
  std::vector<int> cap;            // cap(f)
  std::vector<int> capacity_mem;   // capacity(m)

  // ---- schedule parameters ----
  int II = 1;                      // initiation interval
  int L = 1;                       // schedule length upper bound (= T)
  int Iu = 1;                      // = ⌈L/II⌉

  static int ceilDiv(int a, int b) { return (a + b - 1) / b; }
  void deriveIu() { Iu = ceilDiv(L, II); }
};


// Step-2 SMT input
struct SMTProblem {
  // ---- From Step-1 ILP  ----
  Schedule sched;
  int II = 1;
  int L = 1;
  int Iu = 1;

  // ---- Block-level dependence graph ----
  int V = 0;
  std::vector<Edge> edges;
  std::vector<bool> blocking;          // size == edges.size()

  // ---- Per-op statistics ----
  std::vector<int> cycles;             // cycles(v)
  std::vector<int> regs;               // regs(v)
  std::vector<bool> variable_latency;  // variable_latency(v)
  std::vector<int> spillcost;          // spillcost(v)

  // ---- Hardware ----
  int W = 1;         // total # of warps
  int W_vl = 0;      // warp id reserved for variable-latency ops
  int reg_limit = 0; // per-warp register budget
};

// Step-2 SMT result

struct WarpAssignment {
  bool solved = false;
  std::vector<int> assign; // assign[v] = w
  Schedule sched;
};

// Solver entry points (defined in DependenceILP.cpp / DependenceSMT.cpp)

Schedule solveILP(const ILPProblem &P);
WarpAssignment solveSMT(const SMTProblem &P);

} // namespace twill

#endif // TRITON_ANALYSIS_TWILL_TYPES_H
