// Step-2 SMT
//   op  [v][i][t]
//   live[v][i][t]
//   opw [v][w]

#include "triton/Analysis/TwillTypes.h"

#include <z3++.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace twill {


WarpAssignment solveSMT(const SMTProblem &P) {
  WarpAssignment out;
  out.assign.assign(P.V, -1);

  z3::context ctx;
  z3::solver s(ctx);

  auto inOpRange = [&](int v, int t) {
    return t >= 0 && t + P.cycles[v] <= P.L;
  };
  auto inLiveRange = [&](int t) { return t >= 0 && t <= P.L; };

  // ---- op[v][i][t] : created only when t + cycles[v] ≤ L --------------------
  std::vector<std::vector<std::vector<z3::expr>>> opv(
      P.V,
      std::vector<std::vector<z3::expr>>(
          P.Iu, std::vector<z3::expr>(P.L + 1, ctx.bool_val(false))));
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      for (int t = 0; t <= P.L; t++) {
        if (!inOpRange(v, t))
          continue;
        char nm[64];
        snprintf(nm, sizeof(nm), "op_%d_%d_%d", v, i, t);
        opv[v][i][t] = ctx.bool_const(nm);
      }
    }
  }

  // ---- live[v][i][t] : created for all t ∈ [0, L] ---------------------------
  std::vector<std::vector<std::vector<z3::expr>>> liv(
      P.V,
      std::vector<std::vector<z3::expr>>(
          P.Iu, std::vector<z3::expr>(P.L + 1, ctx.bool_val(false))));
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      for (int t = 0; t <= P.L; t++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "live_%d_%d_%d", v, i, t);
        liv[v][i][t] = ctx.bool_const(nm);
      }
    }
  }

  // ---- opw[v][w] ------------------------------------------------------------
  std::vector<std::vector<z3::expr>> opw(P.V);
  for (int v = 0; v < P.V; v++) {
    opw[v].reserve(P.W);
    for (int w = 0; w < P.W; w++) {
      char nm[64];
      snprintf(nm, sizeof(nm), "opw_%d_%d", v, w);
      opw[v].push_back(ctx.bool_const(nm));
    }
  }


  // -------- WARP UNIQUENESS:  Σ_w opw[v,w] = 1 ------------------------------
  for (int v = 0; v < P.V; v++) {
    z3::expr_vector lits(ctx);
    std::vector<int> coeffs;
    for (int w = 0; w < P.W; w++) {
      lits.push_back(opw[v][w]);
      coeffs.push_back(1);
    }
    s.add(z3::pbeq(lits, coeffs.data(), 1));
  }

  // -------- VARIABLE LATENCY:  variable_latency(v) ⇔ opw[v, W_vl] -----------
  for (int v = 0; v < P.V; v++) {
    if (P.variable_latency[v])
      s.add(opw[v][P.W_vl]);
    else
      s.add(!opw[v][P.W_vl]);
  }

  // -------- REGISTER LIMIT --------------------------------------------------
  //   ∀t,w  Σ_{v,i} (live[v,i,t] ∧ opw[v,w]) · regs[v] ≤ reg_limit
  for (int t = 0; t <= P.L; t++) {
    if (!inLiveRange(t))
      continue;
    for (int w = 0; w < P.W; w++) {
      z3::expr sum = ctx.int_val(0);
      bool any = false;
      for (int v = 0; v < P.V; v++) {
        int r = P.regs[v];
        if (r == 0)
          continue;
        for (int i = 0; i < P.Iu; i++) {
          sum = sum + z3::ite(liv[v][i][t] && opw[v][w],
                              ctx.int_val(r), ctx.int_val(0));
          any = true;
        }
      }
      if (any)
        s.add(sum <= ctx.int_val(P.reg_limit));
    }
  }

  // -------- CROSS-WARP SPILLS ----------------------------------------------
  //   ∀(u,v,d,δ)∈E, t, i, w' ≠ w, s ∈ [0, spillcost(u))
  //     op[u,i,t] ∧ opw[u,w] ∧ opw[v,w'] ⇒ ¬op[v, i+δ, t+d+s]
  for (const Edge &e : P.edges) {
    if (P.spillcost[e.u] <= 0)
      continue;
    for (int i = 0; i < P.Iu; i++) {
      int j = i + e.delta;
      if (j < 0 || j >= P.Iu)
        continue;
      for (int t = 0; t <= P.L; t++) {
        if (!inOpRange(e.u, t))
          continue;
        for (int sd = 0; sd < P.spillcost[e.u]; sd++) {
          int tp = t + e.d + sd;
          if (!inOpRange(e.v, tp))
            continue;
          for (int w = 0; w < P.W; w++) {
            for (int wp = 0; wp < P.W; wp++) {
              if (wp == w)
                continue;
              s.add(z3::implies(
                  opv[e.u][i][t] && opw[e.u][w] && opw[e.v][wp],
                  !opv[e.v][j][tp]));
            }
          }
        }
      }
    }
  }

  // -------- CONCURRENCY -----------------------------------------------------
  //   ∀(u,v,_,_) ∈ E, t, w, i, o ≠ v
  //     op[v,i,t] ∧ opw[v,w] ∧ blocking(u,v) ⇒
  //       ∀ i', t' ∈ [t - (cycles(o) - 1), t] :  ¬(op[o,i',t'] ∧ opw[o,w])
  //
  std::vector<bool> isBlockingConsumer(P.V, false);
  for (size_t k = 0; k < P.edges.size(); k++)
    if (k < P.blocking.size() && P.blocking[k])
      isBlockingConsumer[P.edges[k].v] = true;

  for (int v = 0; v < P.V; v++) {
    if (!isBlockingConsumer[v])
      continue;
    for (int t = 0; t <= P.L; t++) {
      if (!inOpRange(v, t))
        continue;
      for (int w = 0; w < P.W; w++) {
        for (int i = 0; i < P.Iu; i++) {
          z3::expr lhs = opv[v][i][t] && opw[v][w];
          for (int o = 0; o < P.V; o++) {
            if (o == v)
              continue;
            int win_lo = std::max(0, t - (P.cycles[o] - 1));
            for (int ip = 0; ip < P.Iu; ip++) {
              for (int tp = win_lo; tp <= t; tp++) {
                if (!inOpRange(o, tp))
                  continue;
                s.add(z3::implies(lhs, !(opv[o][ip][tp] && opw[o][w])));
              }
            }
          }
        }
      }
    }
  }

  printf("[SMT] built; assertions=%u  → check()...\n",
         (unsigned)s.assertions().size());
  z3::check_result r = s.check();
  if (r != z3::sat) {
    printf("[SMT] %s\n", r == z3::unsat ? "UNSAT" : "UNKNOWN");
    return out;
  }
  out.solved = true;
  z3::model m = s.get_model();

  for (int v = 0; v < P.V; v++) {
    for (int w = 0; w < P.W; w++) {
      if (m.eval(opw[v][w], true).is_true()) {
        out.assign[v] = w;
        break;
      }
    }
  }
  out.sched.solved = true;
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      for (int t = 0; t <= P.L; t++) {
        if (inOpRange(v, t) &&
            m.eval(opv[v][i][t], true).is_true())
          out.sched.op[{v, i, t}] = 1;
        if (m.eval(liv[v][i][t], true).is_true())
          out.sched.live[{v, i, t}] = 1;
      }
    }
  }
  return out;
}

} // namespace twill

#ifndef TWILL_NO_MAIN

namespace {

void dumpWarpAssignment(const twill::SMTProblem &P,
                        const twill::WarpAssignment &A) {
  if (!A.solved) {
    printf("UNSAT / no warp assignment\n");
    return;
  }
  printf("\n== opw[v,w] = 1 ==\n");
  for (int v = 0; v < P.V; v++) {
    printf("  op%d  →  warp %d", v, A.assign[v]);
    if (P.variable_latency[v])
      printf("   (variable_latency)");
    if (A.assign[v] == P.W_vl)
      printf("   [W_vl]");
    printf("\n");
  }

  printf("\n== per-warp timeline (rows: warps, cols: t∈[0..L]) ==\n");
  printf("w \\ t | ");
  for (int t = 0; t <= P.L; t++)
    printf("%2d ", t);
  printf("\n");
  for (int w = 0; w < P.W; w++) {
    printf("w%-4d | ", w);
    for (int t = 0; t <= P.L; t++) {
      int marker = -1;
      for (int v = 0; v < P.V; v++) {
        if (A.assign[v] != w)
          continue;
        for (int i = 0; i < P.Iu; i++) {
          auto it = A.sched.op.find({v, i, t});
          if (it != A.sched.op.end() && it->second) {
            marker = v;
            break;
          }
        }
        if (marker >= 0)
          break;
      }
      if (marker >= 0)
        printf("v%-2d", marker);
      else
        printf(" . ");
    }
    printf("\n");
  }
}

twill::SMTProblem buildSMTDemoProblem() {
  twill::SMTProblem P;

  P.II = 3;
  P.L = 6;
  P.Iu = (P.L + P.II - 1) / P.II;

  // Block-edge
  P.V = 3;
  P.edges = {
      { 0, 1, 1, 0},
      {1, 2, 1, 0},
      {2, 0, 1, 1},
  };
  P.blocking = {false, true, false};

  // Per-op
  P.cycles = {1, 1, 1};
  P.regs = {2, 4, 1};
  P.variable_latency = {true, false, false};
  P.spillcost = {0, 0, 0};

  // Hardware
  P.W = 3;
  P.W_vl = 0;
  P.reg_limit = 32;

  return P;
}

} // namespace

int main() {
  using namespace twill;
  setvbuf(stdout, nullptr, _IONBF, 0);

  printf("[Step 2] solving SMT (joint schedule + warp assignment)...\n");
  SMTProblem P = buildSMTDemoProblem();

  printf("  V=%d  W=%d  W_vl=%d  reg_limit=%d  II=%d  L=%d  Iu=%d\n", P.V, P.W,
         P.W_vl, P.reg_limit, P.II, P.L, P.Iu);
  printf("  edges (u→v,d,δ,blocking):");
  for (size_t k = 0; k < P.edges.size(); k++) {
    const auto &e = P.edges[k];
    printf(" (%d→%d,d=%d,δ=%d,blk=%d)", e.u, e.v, e.d, e.delta,
           (int)P.blocking[k]);
  }
  printf("\n  variable_latency:");
  for (bool b : P.variable_latency)
    printf(" %d", (int)b);
  printf("\n  regs:");
  for (int r : P.regs)
    printf(" %d", r);
  printf("\n  spillcost:");
  for (int c : P.spillcost)
    printf(" %d", c);
  printf("\n");

  WarpAssignment A = solveSMT(P);
  dumpWarpAssignment(P, A);
  return 0;
}

#endif // TWILL_NO_MAIN
