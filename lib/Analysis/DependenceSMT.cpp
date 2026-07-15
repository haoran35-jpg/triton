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

  // -------- STEP-2 RE-SOLVES THE SCHEDULE ----------------------------------

  std::vector<std::vector<z3::expr>> st(
      P.V, std::vector<z3::expr>(P.Iu, ctx.int_val(0)));
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      char nm[64];
      snprintf(nm, sizeof(nm), "st_%d_%d", v, i);
      st[v][i] = ctx.int_const(nm);
      // Exactly one issue slot in [0, L - cycles(v)]; op[v,i,t] ⇒ start == t.
      z3::expr_vector slots(ctx);
      for (int t = 0; t <= P.L; t++) {
        if (!inOpRange(v, t))
          continue;
        slots.push_back(opv[v][i][t]);
        s.add(z3::implies(opv[v][i][t], st[v][i] == ctx.int_val(t)));
      }
      s.add(z3::atleast(slots, 1) && z3::atmost(slots, 1));
      s.add(st[v][i] >= 0 &&
            st[v][i] + ctx.int_val(P.cycles[v]) <= ctx.int_val(P.L));
    }
  }

  // Dependences:  start(v, i+δ) ≥ start(u, i) + d.
  for (const Edge &e : P.edges) {
    for (int i = 0; i < P.Iu; i++) {
      int j = i + e.delta;
      if (j < 0 || j >= P.Iu)
        continue;
      s.add(st[e.v][j] >= st[e.u][i] + ctx.int_val(e.d));
    }
  }

  // live[v,i,t] ⇔ start[v,i] ≤ t ≤ lastUse(v,i), where
  //   lastUse = max(done(v,i)=start+cycles, max over consumers' start times).
  for (int v = 0; v < P.V; v++) {
    for (int i = 0; i < P.Iu; i++) {
      z3::expr lu = st[v][i] + ctx.int_val(P.cycles[v]); // done(v,i)
      for (const Edge &e : P.edges) {
        if (e.u != v)
          continue;
        int j = i + e.delta;
        if (j < 0 || j >= P.Iu)
          continue;
        lu = z3::ite(st[e.v][j] > lu, st[e.v][j], lu);
      }
      for (int t = 0; t <= P.L; t++)
        s.add(liv[v][i][t] ==
              ((ctx.int_val(t) >= st[v][i]) && (ctx.int_val(t) <= lu)));
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

// solveTwillCacheJoint : original Twill (Op[v,i,t], Live[v,i,t], Opw[v,w,i])
// jointly with the GPU warp-scheduler + abstract-cache model.
CacheRRIterResult solveTwillCacheJoint(const CacheRRIterProblem &P) {
  CacheRRIterResult out;
  const int V = P.V(), I = P.I, W = P.W, S = P.S, T = P.T;
  out.warp.assign(V, std::vector<int>(I, -1));
  out.start.assign(V, std::vector<int>(I, -1));
  out.lat.assign(V, std::vector<int>(I, 0));
  out.hit.assign(V, std::vector<int>(I, -1));

  z3::context c;
  // Accumulate all constraints; fed to either a feasibility solver or, when
  // P.minimizeMakespan is set, a z3::optimize that minimizes the makespan.
  z3::expr_vector C(c);
  auto ADD = [&](const z3::expr &e) { C.push_back(e); };
  auto B = [&](const std::string &n) { return c.bool_const(n.c_str()); };
  auto Iv = [&](const std::string &n) { return c.int_const(n.c_str()); };
  auto N = [&](int v) { return c.int_val(v); };
  auto nm = [&](const char *p, int a, int b) {
    return std::string(p) + std::to_string(a) + "_" + std::to_string(b);
  };
  auto nm3 = [&](const char *p, int a, int b, int d) {
    return std::string(p) + std::to_string(a) + "_" + std::to_string(b) + "_" +
           std::to_string(d);
  };
  auto ord = [&](int v, int i) { return i * V + v; }; // iteration-major order

  // ---- decision variables ----
  // Opw[v][w][i]; start/done/lat/hit[v][i]
  std::vector<std::vector<std::vector<z3::expr>>> opw(
      V, std::vector<std::vector<z3::expr>>(W));
  std::vector<std::vector<z3::expr>> start(V), done(V), lat(V), hitv(V);
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i) {
      for (int w = 0; w < W; ++w)
        opw[v][w].push_back(B(nm3("Opw_", v, w, i)));
      start[v].push_back(Iv(nm("St_", v, i)));
      done[v].push_back(Iv(nm("Dn_", v, i)));
      lat[v].push_back(Iv(nm("Lat_", v, i)));
      hitv[v].push_back(B(nm("Hit_", v, i)));
    }
  // Op[v][i][t], Live[v][i][t]  (canonical Twill time-indexed structures)
  std::vector<std::vector<std::vector<z3::expr>>> Op(
      V, std::vector<std::vector<z3::expr>>(I));
  std::vector<std::vector<std::vector<z3::expr>>> Live(
      V, std::vector<std::vector<z3::expr>>(I));
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i)
      for (int t = 0; t <= T; ++t) {
        Op[v][i].push_back(B("Op_" + std::to_string(v) + "_" +
                             std::to_string(i) + "_" + std::to_string(t)));
        Live[v][i].push_back(B("Lv_" + std::to_string(v) + "_" +
                               std::to_string(i) + "_" + std::to_string(t)));
      }
  // Set-associative abstract cache: G sets × K ways.  Falls back to a single
  // fully-associative set of S ways when numSets/numWays are unset.  The set
  // index of every load is a compile-time constant (line % G), so only the
  // per-set victim pointer, valid bits, and stored lines are SMT variables.
  const bool setAssoc = P.numWays > 0 && P.numSets > 0;
  const int G = setAssoc ? P.numSets : 1;  // sets
  const int K = setAssoc ? P.numWays : S;  // ways per set
  const int Stot = G * K;                  // total cache lines
  auto slotIx = [&](int g, int k) { return g * K + k; };
  auto setOf = [&](int line) { return setAssoc ? (line % G) : 0; };
  std::vector<z3::expr> Rr;
  std::vector<std::vector<z3::expr>> vps(T + 1); // vps[t][g] victim ptr per set
  std::vector<std::vector<z3::expr>> Iss(W), Ready(W);
  std::vector<std::vector<z3::expr>> cV(T + 1), cL(T + 1);
  for (int t = 0; t <= T; ++t) {
    Rr.push_back(Iv("Rr_" + std::to_string(t)));
    for (int g = 0; g < G; ++g)
      vps[t].push_back(Iv(nm("VictimSlot_", t, g)));
    for (int sl = 0; sl < Stot; ++sl) {
      cV[t].push_back(B(nm("CacheValid_", t, sl)));
      cL[t].push_back(Iv(nm("CacheLine_", t, sl)));
    }
  }
  for (int w = 0; w < W; ++w)
    for (int t = 0; t <= T; ++t) {
      Iss[w].push_back(B(nm("Issued_", w, t)));
      Ready[w].push_back(B(nm("Ready_", w, t)));
    }

  // ---- Opw uniqueness (exactly one warp per op-instance) ----
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i) {
      z3::expr_vector ow(c);
      for (int w = 0; w < W; ++w)
        ow.push_back(opw[v][w][i]);
      ADD(z3::atleast(ow, 1) && z3::atmost(ow, 1));
    }
  auto warpOf = [&](int v, int i) {
    z3::expr e = N(0);
    for (int w = 1; w < W; ++w)
      e = z3::ite(opw[v][w][i], N(w), e);
    return e;
  };

  // ---- latency (hit/miss), Done, dependences ----
  //  !IsLoad ⇒ Lat = StaticLat ;  Hit ⇒ FastLat ;  !Hit ⇒ SlowLat
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i) {
      ADD(start[v][i] >= 0 && start[v][i] < N(T));
      if (P.ops[v].isLoad)
        ADD(lat[v][i] ==
            z3::ite(hitv[v][i], N(P.ops[v].fastLat), N(P.ops[v].slowLat)));
      else
        ADD(lat[v][i] == N(P.ops[v].staticLat));
      ADD(done[v][i] == start[v][i] + lat[v][i]); // Done = Start + Lat
      ADD(done[v][i] <= N(T));                    // Done ≤ T
      for (auto [u, d] : P.ops[v].deps)
        if (i - d >= 0)
          ADD(start[v][i] >= done[u][i - d]);
    }

  // ---- Op[v,i,t] ⇔ (Start[v,i] == t)  (⇒ exactly one t, and Op⇒Start=t) ----
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i)
      for (int t = 0; t <= T; ++t)
        ADD(Op[v][i][t] == (start[v][i] == N(t)));

  // ---- issue width: Σ_{v,i} Op[v,i,t] ≤ IssueWidth ----
  for (int t = 0; t <= T; ++t) {
    z3::expr_vector ev(c);
    for (int v = 0; v < V; ++v)
      for (int i = 0; i < I; ++i)
        ev.push_back(Op[v][i][t]);
    ADD(z3::atmost(ev, P.issueWidth));
  }

  // ---- in-order per warp by program order (iteration-major) ----
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i)
      for (int v2 = 0; v2 < V; ++v2)
        for (int i2 = 0; i2 < I; ++i2)
          if (ord(v, i) < ord(v2, i2))
            for (int w = 0; w < W; ++w)
              ADD(z3::implies(opw[v][w][i] && opw[v2][w][i2],
                              start[v][i] < start[v2][i2]));

  // ---- CROSS-WARP SPILLS --------------------------------------------------
  //   A value read by a consumer placed on a *different* warp than its producer
  //   must be spilled/communicated, costing spillcost(u) extra cycles:
  //     Opw[u,w,i-d] ∧ Opw[v,w',i] ∧ w'≠w ⇒ Start[v,i] ≥ Done[u,i-d]+spill(u)
  for (int v = 0; v < V; ++v)
    for (auto [u, d] : P.ops[v].deps) {
      int sc = P.ops[u].spillcost;
      if (sc <= 0)
        continue;
      for (int i = 0; i < I; ++i) {
        if (i - d < 0)
          continue;
        for (int w = 0; w < W; ++w)
          for (int wp = 0; wp < W; ++wp)
            if (wp != w)
              ADD(z3::implies(opw[u][w][i - d] && opw[v][wp][i],
                              start[v][i] >= done[u][i - d] + N(sc)));
      }
    }

  // ---- BLOCKING CONCURRENCY -----------------------------------------------
  //   A blocking op holds its warp for its whole latency; no other op on the
  //   same warp may overlap its [Start, Done) interval:
  //     Opw[v,w,i] ∧ Opw[o,w,i'] ⇒ Done[o,i'] ≤ Start[v,i] ∨ Start[o,i'] ≥ Done[v,i]
  for (int v = 0; v < V; ++v) {
    if (!P.ops[v].blocking)
      continue;
    for (int i = 0; i < I; ++i)
      for (int o = 0; o < V; ++o)
        for (int i2 = 0; i2 < I; ++i2) {
          if (o == v && i2 == i) // skip only the same instance
            continue;
          for (int w = 0; w < W; ++w)
            ADD(z3::implies(opw[v][w][i] && opw[o][w][i2],
                            (done[o][i2] <= start[v][i]) ||
                                (start[o][i2] >= done[v][i])));
        }
  }

  auto depsReady = [&](int v, int i, int t) {
    z3::expr e = c.bool_val(true);
    for (auto [u, d] : P.ops[v].deps)
      if (i - d >= 0)
        e = e && (done[u][i - d] <= N(t));
    return e;
  };

  // ---- priority assigned warp scheduler ----
  ADD(Rr[0] == 0);
  auto rank = [&](int w, int t) {
    return z3::ite(N(w) >= Rr[t], N(w) - Rr[t], N(w) - Rr[t] + N(W));
  };
  for (int t = 0; t < T; ++t) {
    // Issued[w,t] ⇔ OR_{v,i}(Opw[v,w,i] ∧ Op[v,i,t])
    for (int w = 0; w < W; ++w) {
      z3::expr e = c.bool_val(false);
      for (int v = 0; v < V; ++v)
        for (int i = 0; i < I; ++i)
          e = e || (opw[v][w][i] && Op[v][i][t]);
      ADD(Iss[w][t] == e);
    }
    // Ready[w,t] ⇔ OR_{v,i}(Opw[v,w,i] ∧ ReadyOp[v,i,t]); ReadyOp = next
    // un-issued op on w (lowest ord with start≥t) with deps satisfied by t.
    for (int w = 0; w < W; ++w) {
      z3::expr rdy = c.bool_val(false);
      for (int v = 0; v < V; ++v)
        for (int i = 0; i < I; ++i) {
          z3::expr isNext = opw[v][w][i] && (start[v][i] >= N(t));
          for (int v2 = 0; v2 < V; ++v2)
            for (int i2 = 0; i2 < I; ++i2)
              if (ord(v2, i2) < ord(v, i))
                isNext = isNext && !(opw[v2][w][i2] && (start[v2][i2] >= N(t)));
          rdy = rdy || (isNext && depsReady(v, i, t));
        }
      ADD(Ready[w][t] == rdy);
      ADD(z3::implies(Iss[w][t], Ready[w][t])); // Issued ⇒ Ready
    }
    z3::expr_vector issv(c);
    z3::expr anyReady = c.bool_val(false), anyIss = c.bool_val(false);
    for (int w = 0; w < W; ++w) {
      issv.push_back(Iss[w][t]);
      anyReady = anyReady || Ready[w][t];
      anyIss = anyIss || Iss[w][t];
    }
    ADD(z3::atmost(issv, 1));
    ADD(anyReady == anyIss); // issue iff some warp ready
    for (int w = 0; w < W; ++w)
      for (int wp = 0; wp < W; ++wp)
        if (wp != w)
          ADD(z3::implies(Iss[w][t] && Ready[wp][t],
                          rank(w, t) <= rank(wp, t)));
    // Rr advance: Issued[w,t] ⇒ Rr[t+1]=(w+1) mod W ; else unchanged
    z3::expr none = c.bool_val(true);
    for (int w = 0; w < W; ++w) {
      none = none && !Iss[w][t];
      ADD(z3::implies(Iss[w][t], Rr[t + 1] == N((w + 1) % W)));
    }
    ADD(z3::implies(none, Rr[t + 1] == Rr[t]));
  }

  // ---- abstract set-associative cache, per-set round-robin VictimSlot ----
  for (int sl = 0; sl < Stot; ++sl) {
    ADD(cV[0][sl] == c.bool_val(false));
    ADD(cL[0][sl] == -1);
  }
  for (int g = 0; g < G; ++g)
    ADD(vps[0][g] == 0);
  // ExecLoad[v,i,t] := Op[v,i,t] ∧ IsLoad[v]; on exec, Hit ⇔ line present in
  // its set's ways (set index g = line % G is a compile-time constant).
  for (int v = 0; v < V; ++v) {
    if (!P.ops[v].isLoad)
      continue;
    for (int i = 0; i < I; ++i) {
      int line = P.line(v, i), g = setOf(line);
      for (int t = 0; t < T; ++t) {
        z3::expr present = c.bool_val(false);
        for (int k = 0; k < K; ++k)
          present = present ||
                    (cV[t][slotIx(g, k)] && (cL[t][slotIx(g, k)] == N(line)));
        ADD(z3::implies(Op[v][i][t], hitv[v][i] == present));
      }
    }
  }
  // cache update: a miss installs its line at the victim way of *its* set;
  // other sets are unchanged.  Issue width 1 ⇒ ≤1 install per set per cycle.
  for (int t = 0; t < T; ++t) {
    for (int g = 0; g < G; ++g) {
      z3::expr install = c.bool_val(false);
      z3::expr missLine = N(-1);
      for (int v = 0; v < V; ++v) {
        if (!P.ops[v].isLoad)
          continue;
        for (int i = 0; i < I; ++i) {
          if (setOf(P.line(v, i)) != g)
            continue;
          z3::expr ev = Op[v][i][t] && !hitv[v][i];
          install = install || ev;
          missLine = z3::ite(ev, N(P.line(v, i)), missLine);
        }
      }
      z3::expr nextp = z3::ite(vps[t][g] == N(K - 1), N(0), vps[t][g] + 1);
      ADD(vps[t + 1][g] == z3::ite(install, nextp, vps[t][g]));
      for (int k = 0; k < K; ++k) {
        int sl = slotIx(g, k);
        z3::expr here = install && (vps[t][g] == N(k));
        ADD(cV[t + 1][sl] == z3::ite(here, c.bool_val(true), cV[t][sl]));
        ADD(cL[t + 1][sl] == z3::ite(here, missLine, cL[t][sl]));
      }
    }
  }

  // ---- Live[v,i,t] ⇔ start ≤ t ≤ lastUse ; per-warp register budget ----
  std::vector<std::vector<z3::expr>> lastUse(V);
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i) {
      z3::expr lu = done[v][i];
      for (int v2 = 0; v2 < V; ++v2)
        for (auto [u, d] : P.ops[v2].deps)
          if (u == v && i + d < I)
            lu = z3::ite(start[v2][i + d] > lu, start[v2][i + d], lu);
      lastUse[v].push_back(lu);
    }
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i)
      for (int t = 0; t <= T; ++t)
        ADD(Live[v][i][t] ==
            ((N(t) >= start[v][i]) && (N(t) <= lastUse[v][i])));
  // per-warp:  Σ_{v,i} (Live[v,i,t] ∧ Opw[v,w,i]) · footprint ≤ regLimit
  if (P.regLimit < (1 << 20)) {
    for (int t = 0; t <= T; ++t)
      for (int w = 0; w < W; ++w) {
        z3::expr load = N(0);
        for (int v = 0; v < V; ++v) {
          if (P.ops[v].footprint == 0)
            continue;
          for (int i = 0; i < I; ++i)
            load = load + z3::ite(Live[v][i][t] && opw[v][w][i],
                                  N(P.ops[v].footprint), N(0));
        }
        ADD(load <= N(P.regLimit));
      }
  }
  // global buffer capacity (sum over all warps)
  if (P.bufferCapacity < (1 << 20)) {
    for (int t = 0; t <= T; ++t) {
      z3::expr load = N(0);
      for (int v = 0; v < V; ++v) {
        if (P.ops[v].footprint == 0)
          continue;
        for (int i = 0; i < I; ++i)
          load = load + z3::ite(Live[v][i][t], N(P.ops[v].footprint), N(0));
      }
      ADD(load <= N(P.bufferCapacity));
    }
  }

  // ---- makespan bound ----
  z3::expr Mk = Iv("makespan");
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i)
      ADD(done[v][i] <= Mk);
  ADD(Mk <= N(T));

  printf("[twill-cache-joint SMT] V=%d I=%d W=%d cache=%dset×%dway T=%d (%s) "
         "→ check()...\n",
         V, I, W, G, K, T,
         P.minimizeMakespan ? "minimize makespan" : "feasibility");
  z3::model m(c);
  z3::check_result res = z3::unknown;
  if (P.minimizeMakespan) {
    z3::solver s(c);
    z3::params pr(c);
    pr.set("timeout", (unsigned)P.optimizeTimeoutMs);
    s.set(pr);
    for (unsigned k = 0; k < C.size(); ++k)
      s.add(C[k]);
    res = s.check();
    if (res == z3::sat) {
      m = s.get_model();
      int best = m.eval(Mk, true).get_numeral_int();
      printf("[twill-cache-joint SMT] feasible makespan=%d; tightening...\n",
             best);
      while (best > 0) {
        s.push();
        s.add(Mk <= N(best - 1));
        z3::check_result r2 = s.check();
        if (r2 == z3::sat) {
          m = s.get_model();
          best = m.eval(Mk, true).get_numeral_int();
          s.pop();
        } else {
          s.pop();
          if (r2 == z3::unknown)
            printf("[twill-cache-joint SMT] descent stopped (timeout) at "
                   "makespan=%d (may be non-optimal)\n",
                   best);
          break;
        }
      }
      printf("[twill-cache-joint SMT] minimized makespan=%d\n", best);
    }
  } else {
    z3::solver s(c);
    for (unsigned k = 0; k < C.size(); ++k)
      s.add(C[k]);
    res = s.check();
    if (res == z3::sat)
      m = s.get_model();
  }
  if (res != z3::sat) {
    printf("[twill-cache-joint SMT] UNSAT (increase T?)\n");
    return out;
  }
  auto iv = [&](const z3::expr &e) { return m.eval(e, true).get_numeral_int(); };
  auto bv = [&](const z3::expr &e) { return m.eval(e, true).is_true(); };

  out.solved = true;
  out.makespan = 0;
  out.misses = 0;
  for (int v = 0; v < V; ++v)
    for (int i = 0; i < I; ++i) {
      out.warp[v][i] = iv(warpOf(v, i));
      out.start[v][i] = iv(start[v][i]);
      out.lat[v][i] = iv(lat[v][i]);
      out.makespan = std::max(out.makespan, out.start[v][i] + out.lat[v][i]);
      if (P.ops[v].isLoad) {
        out.hit[v][i] = bv(hitv[v][i]) ? 1 : 0;
        if (!out.hit[v][i])
          out.misses++;
      }
      for (int t = 0; t <= T; ++t) {
        if (bv(Op[v][i][t]))
          out.op[{v, i, t}] = 1;
        if (bv(Live[v][i][t]))
          out.live[{v, i, t}] = 1;
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
