//   op  [v][i][t]
//   live[v][i][t]
//   opw [v][w][i]

#include "triton/Analysis/TwillTypes.h"

#include <z3++.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace twill {

PingPongAssignment solveSMTPingPong(const PingPongSMTProblem &P) {
  PingPongAssignment out;
  out.assign.assign(P.V, std::vector<int>(P.Iu, -1));

  z3::context ctx;

  z3::optimize s(ctx);
    z3::params prm(ctx);
    prm.set("timeout", (unsigned)P.opt_timeout_ms);
    s.set(prm);
  }

  auto inOpRange = [&](int v, int t) {
    return t >= 0 && t + P.cycles[v] <= P.L;
  };
  auto inLiveRange = [&](int t) { return t >= 0 && t <= P.L; };

  // ---- op[v][i][t] : created only when t + cycles[v] ≤ L --------------------
  std::vector<std::vector<std::vector<z3::expr>>> opv(
      P.V,
      std::vector<std::vector<z3::expr>>(
          P.Iu, std::vector<z3::expr>(P.L + 1, ctx.bool_val(false))));
  for (int v = 0; v < P.V; v++)
    for (int i = 0; i < P.Iu; i++)
      for (int t = 0; t <= P.L; t++) {
        if (!inOpRange(v, t))
          continue;
        char nm[64];
        snprintf(nm, sizeof(nm), "op_%d_%d_%d", v, i, t);
        opv[v][i][t] = ctx.bool_const(nm);
      }

  // ---- live[v][i][t] : created for all t ∈ [0, L] ---------------------------
  std::vector<std::vector<std::vector<z3::expr>>> liv(
      P.V,
      std::vector<std::vector<z3::expr>>(
          P.Iu, std::vector<z3::expr>(P.L + 1, ctx.bool_val(false))));
  for (int v = 0; v < P.V; v++)
    for (int i = 0; i < P.Iu; i++)
      for (int t = 0; t <= P.L; t++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "live_%d_%d_%d", v, i, t);
        liv[v][i][t] = ctx.bool_const(nm);
      }

  // opw[v][w][i] : ITERATION-AWARE warp binding (Opw[v,w,i])

  std::vector<std::vector<std::vector<z3::expr>>> opw(
      P.V, std::vector<std::vector<z3::expr>>(P.W));
  for (int v = 0; v < P.V; v++)
    for (int w = 0; w < P.W; w++)
      for (int i = 0; i < P.Iu; i++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "opw_%d_%d_%d", v, w, i);
        opw[v][w].push_back(ctx.bool_const(nm));
      }

  // start[v][i] : issue cycle, linked to the one-hot op[v][i][*].
  std::vector<std::vector<z3::expr>> st(
      P.V, std::vector<z3::expr>(P.Iu, ctx.int_val(0)));
  for (int v = 0; v < P.V; v++)
    for (int i = 0; i < P.Iu; i++) {
      char nm[64];
      snprintf(nm, sizeof(nm), "st_%d_%d", v, i);
      st[v][i] = ctx.int_const(nm);
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

  // Dependences:  start(v, i+δ) ≥ start(u, i) + d.
  for (const Edge &e : P.edges)
    for (int i = 0; i < P.Iu; i++) {
      int j = i + e.delta;
      if (j < 0 || j >= P.Iu)
        continue;
      s.add(st[e.v][j] >= st[e.u][i] + ctx.int_val(e.d));
    }

  // live[v,i,t] ⇔ start[v,i] ≤ t ≤ lastUse(v,i).
  for (int v = 0; v < P.V; v++)
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

  // WARP UNIQUENESS ∀v,i  Σ_w opw[v,i,w] = 1
  for (int v = 0; v < P.V; v++)
    for (int i = 0; i < P.Iu; i++) {
      z3::expr_vector lits(ctx);
      std::vector<int> coeffs;
      for (int w = 0; w < P.W; w++) {
        lits.push_back(opw[v][w][i]);
        coeffs.push_back(1);
      }
      if (P.warp_uniqueness)
        s.add(z3::pbeq(lits, coeffs.data(), 1));
      else
        s.add(z3::atleast(lits, 1));
    }

  // (Dedicated variable-latency warp W_vl constraint removed: variable-latency
  //  ops are now free to land on any warp, decided by the joint schedule.)

  // -------- ORIGINAL-TWILL SINGLE-WARP BINDING (opw[v,w], no ping-pong) ------
  //   Force all iterations of an op onto the same warp: opw[v,w,i] = opw[v,w,0].
  if (P.pin_op_single_warp)
    for (int v = 0; v < P.V; v++)
      for (int w = 0; w < P.W; w++)
        for (int i = 1; i < P.Iu; i++)
          s.add(opw[v][w][i] == opw[v][w][0]);

  // -------- REGISTER LIMIT --------------------------------------------------
  //   ∀t,w  Σ_{v,i} (live[v,i,t] ∧ opw[v,w,i]) · regs[v] ≤ reg_limit
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
          sum = sum + z3::ite(liv[v][i][t] && opw[v][w][i],
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
  //     op[u,i,t] ∧ opw[u,w,i] ∧ opw[v,w',i+δ] ⇒ ¬op[v, i+δ, t+d+s]
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
          for (int w = 0; w < P.W; w++)
            for (int wp = 0; wp < P.W; wp++) {
              if (wp == w)
                continue;
              s.add(z3::implies(
                  opv[e.u][i][t] && opw[e.u][w][i] && opw[e.v][wp][j],
                  !opv[e.v][j][tp]));
            }
        }
      }
    }
  }

  // -------- BLOCKING CONCURRENCY -------------------------------------------
  //   ∀(u,v,_,_) ∈ E, t, w, i, o ≠ v
  //     op[v,i,t] ∧ opw[v,w,i] ∧ blocking(u,v) ⇒
  //       ∀ i', t' ∈ [t - (cycles(o) - 1), t] :  ¬(op[o,i',t'] ∧ opw[o,w,i'])
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
      for (int w = 0; w < P.W; w++)
        for (int i = 0; i < P.Iu; i++) {
          z3::expr lhs = opv[v][i][t] && opw[v][w][i];
          for (int o = 0; o < P.V; o++) {
            if (o == v)
              continue;
            int win_lo = std::max(0, t - (P.cycles[o] - 1));
            for (int ip = 0; ip < P.Iu; ip++)
              for (int tp = win_lo; tp <= t; tp++) {
                if (!inOpRange(o, tp))
                  continue;
                s.add(z3::implies(lhs, !(opv[o][ip][tp] && opw[o][w][ip])));
              }
          }
        }
    }
  }

  //PER-WARP IN-ORDER ISSUE

  //     opw[v,w,i] ∧ opw[o,w,i'] ⇒
  //       start[o,i'] ≥ start[v,i]+cycles(v) ∨ start[v,i] ≥ start[o,i']+cycles(o)
  for (int v = 0; v < P.V; v++)
    for (int i = 0; i < P.Iu; i++)
      for (int o = 0; o < P.V; o++)
        for (int ip = 0; ip < P.Iu; ip++) {
          if (o < v || (o == v && ip <= i))
            continue; // unordered pair, count each once; skip self
          for (int w = 0; w < P.W; w++)
            s.add(z3::implies(
                opw[v][w][i] && opw[o][w][ip],
                (st[o][ip] >= st[v][i] + ctx.int_val(P.cycles[v])) ||
                    (st[v][i] >= st[o][ip] + ctx.int_val(P.cycles[o]))));
        }

  //  PING-PONG / TENSOR-CORE SINGLE-ISSUE
  //   TCUse[w,t] ⇔ ∨_{v,i} (IsMMA(v) ∧ op[v,i,t] ∧ opw[v,w,i])
  //   ∀t.  Σ_w TCUse[w,t] ≤ TensorCoreIssueWidth

  std::vector<std::vector<z3::expr>> tcUse(P.W);
  bool anyMMA = false;
  for (int v = 0; v < P.V; v++)
    if (v < (int)P.is_mma.size() && P.is_mma[v])
      anyMMA = true;

  if (P.tc_single_issue && anyMMA) {
    for (int w = 0; w < P.W; w++)
      for (int t = 0; t <= P.L; t++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "tcuse_%d_%d", w, t);
        z3::expr tu = ctx.bool_const(nm);
        tcUse[w].push_back(tu);

        z3::expr disj = ctx.bool_val(false);
        for (int v = 0; v < P.V; v++) {
          if (!(v < (int)P.is_mma.size() && P.is_mma[v]))
            continue;
          int lo = std::max(0, t - (P.cycles[v] - 1));
          for (int i = 0; i < P.Iu; i++)
            for (int tp = lo; tp <= t; tp++) {
              if (!inOpRange(v, tp))
                continue;
              disj = disj || (opv[v][i][tp] && opw[v][w][i]);
            }
        }
        s.add(tu == disj);
      }

    for (int t = 0; t <= P.L; t++) {
      z3::expr_vector lits(ctx);
      for (int w = 0; w < P.W; w++)
        lits.push_back(tcUse[w][t]);
      s.add(z3::atmost(lits, P.tc_issue_width));
    }
  }


  //   consumerOp[v] = is_mma[v] ∨ is_softmax[v]
  //   consumerWarp[w] ⇔ ∨_{v∈consumer,i} opw[v,w,i]           (definition)
  //   ¬consumerOp[v] ∧ opw[v,w,i] ⇒ ¬consumerWarp[w]          (dedicate)
  if (P.enforce_consumer_warp) {
    std::vector<bool> consumerOp(P.V, false);
    for (int v = 0; v < P.V; v++) {
      bool mma = v < (int)P.is_mma.size() && P.is_mma[v];
      bool sm = v < (int)P.is_softmax.size() && P.is_softmax[v];
      consumerOp[v] = mma || sm;
    }

    std::vector<z3::expr> consumerWarp;
    for (int w = 0; w < P.W; w++) {
      char nm[64];
      snprintf(nm, sizeof(nm), "consumerWarp_%d", w);
      consumerWarp.push_back(ctx.bool_const(nm));

      // consumerWarp[w] ⇔ some consumer op-instance is bound to w.
      z3::expr hosts = ctx.bool_val(false);
      for (int v = 0; v < P.V; v++) {
        if (!consumerOp[v])
          continue;
        for (int i = 0; i < P.Iu; i++)
          hosts = hosts || opw[v][w][i];
      }
      s.add(consumerWarp[w] == hosts);
    }

    for (int v = 0; v < P.V; v++) {
      if (consumerOp[v])
        continue;
      for (int w = 0; w < P.W; w++)
        for (int i = 0; i < P.Iu; i++)
          s.add(z3::implies(opw[v][w][i], !consumerWarp[w]));
    }
  }

  //   IsMMA(v)         : opw[v,i,w] ⇒ ¬opw[v,i+1,w]
  //   IsMMA(v),SM(v')  : opw[v,i,w] ⇒ ¬opw[v',i,w]

  if (P.pingpong_alternate) {
    for (int v = 0; v < P.V; v++) {
      bool mma = v < (int)P.is_mma.size() && P.is_mma[v];
      if (!mma)
        continue;
      // (1) same MMA op, consecutive iterations, must change warp.
      for (int i = 0; i + 1 < P.Iu; i++)
        for (int w = 0; w < P.W; w++)
          s.add(z3::implies(opw[v][w][i], !opw[v][w][i + 1]));
      // (2) this iteration's MMA and softmax ops on different warps.
      for (int vs = 0; vs < P.V; vs++) {
        bool sm = vs < (int)P.is_softmax.size() && P.is_softmax[vs];
        if (!sm)
          continue;
        for (int i = 0; i < P.Iu; i++)
          for (int w = 0; w < P.W; w++)
            s.add(z3::implies(opw[v][w][i], !opw[vs][w][i]));
      }
    }
  }

  //  CIRCULAR / MULTI-STAGE SHARED-MEMORY BUFFER
  //   slot(i) = i mod D.  Tile i is filled by Copy ops and drained by MMA ops.
  //     Start[Copy,i]=min_c st[c,i]   Done[Copy,i]=max_c (st[c,i]+lat[c])
  //     Start[MMA ,i]=min_m st[m,i]   Done[MMA ,i]=max_m (st[m,i]+lat[m])
  //     BufLive[i,t] ⇔ Start[Copy,i] ≤ t < Done[MMA,i]
  //   (capacity)  ∀t Σ_i BufLive[i,t]·bytes ≤ smem_capacity
  //   (reuse)     ∀i<j slot(i)=slot(j) ⇒ Done[MMA,i] ≤ Start[Copy,j]
  //   (order)     ∀i Done[Copy,i] ≤ Start[MMA,i]
  if (P.circular_buffer) {
    std::vector<int> copyOps, mmaOps;
    for (int v = 0; v < P.V; v++) {
      if (v < (int)P.is_copy.size() && P.is_copy[v])
        copyOps.push_back(v);
      if (v < (int)P.is_mma.size() && P.is_mma[v])
        mmaOps.push_back(v);
    }
    const int D = P.buf_depth > 0 ? P.buf_depth : 1;

    if (!copyOps.empty() && !mmaOps.empty()) {
      auto mkMin = [&](const char *tag, int i, const std::vector<int> &ops,
                       bool addLat) {
        char nm[64];
        snprintf(nm, sizeof(nm), "%s_%d", tag, i);
        z3::expr r = ctx.int_const(nm);
        z3::expr_vector eqs(ctx);
        for (int v : ops) {
          z3::expr e = addLat ? (st[v][i] + ctx.int_val(P.cycles[v])) : st[v][i];
          s.add(r <= e);
          eqs.push_back(r == e);
        }
        s.add(z3::mk_or(eqs));
        return r;
      };
      auto mkMax = [&](const char *tag, int i, const std::vector<int> &ops,
                       bool addLat) {
        char nm[64];
        snprintf(nm, sizeof(nm), "%s_%d", tag, i);
        z3::expr r = ctx.int_const(nm);
        z3::expr_vector eqs(ctx);
        for (int v : ops) {
          z3::expr e = addLat ? (st[v][i] + ctx.int_val(P.cycles[v])) : st[v][i];
          s.add(r >= e);
          eqs.push_back(r == e);
        }
        s.add(z3::mk_or(eqs));
        return r;
      };

      std::vector<z3::expr> cStart, cDone, mStart, mDone;
      for (int i = 0; i < P.Iu; i++) {
        cStart.push_back(mkMin("cStart", i, copyOps, /*addLat=*/false));
        cDone.push_back(mkMax("cDone", i, copyOps, /*addLat=*/true));
        mStart.push_back(mkMin("mStart", i, mmaOps, /*addLat=*/false));
        mDone.push_back(mkMax("mDone", i, mmaOps, /*addLat=*/true));
      }

      // (order) consume-after-produce: Done[Copy,i] ≤ Start[MMA,i].
      for (int i = 0; i < P.Iu; i++)
        s.add(cDone[i] <= mStart[i]);

      // (reuse) same slot -> next producer waits for prior consumer.
      for (int i = 0; i < P.Iu; i++)
        for (int j = i + 1; j < P.Iu; j++)
          if (((j - i) % D) == 0)
            s.add(mDone[i] <= cStart[j]);

      // (capacity) BufLive[i,t] ⇔ Start[Copy,i] ≤ t < Done[MMA,i];
      std::vector<std::vector<z3::expr>> bufLive(P.Iu);
      for (int i = 0; i < P.Iu; i++)
        for (int t = 0; t <= P.L; t++) {
          char nm[64];
          snprintf(nm, sizeof(nm), "buflive_%d_%d", i, t);
          z3::expr b = ctx.bool_const(nm);
          s.add(b == ((cStart[i] <= ctx.int_val(t)) &&
                      (ctx.int_val(t) < mDone[i])));
          bufLive[i].push_back(b);
        }
      std::vector<int> coeffs(P.Iu, P.bytes_per_tile);
      for (int t = 0; t <= P.L; t++) {
        z3::expr_vector lits(ctx);
        for (int i = 0; i < P.Iu; i++)
          lits.push_back(bufLive[i][t]);
        s.add(z3::pble(lits, coeffs.data(), P.smem_capacity));
      }
    }
  }

  // MINIMIZE MAKESPAN
  //   Mk ≥ start[v,i] + cycles(v)  ∀v,i ;  minimize Mk
  if (P.minimize_makespan) {
    z3::expr Mk = ctx.int_const("makespan");
    for (int v = 0; v < P.V; v++)
      for (int i = 0; i < P.Iu; i++)
        s.add(Mk >= st[v][i] + ctx.int_val(P.cycles[v]));
    s.minimize(Mk);
  }

  printf("[ping-pong SMT] built; %s → check()...\n",
         P.minimize_makespan ? "minimizing makespan;" : "feasibility;");
  z3::check_result r = s.check();

  z3::model m(ctx);
  if (r == z3::sat) {
    m = s.get_model();
  } else if (r == z3::unknown) {
    try {
      m = s.get_model();
    } catch (...) {
    }
    if (m.size() == 0) {
      printf("[ping-pong SMT] UNKNOWN (no model within time budget)\n");
      return out;
    }
    printf("[ping-pong SMT] timed out; using best schedule found so far\n");
  } else {
    printf("[ping-pong SMT] UNSAT\n");
    return out;
  }
  out.solved = true;

  out.warps.assign(P.V, std::vector<std::vector<int>>(P.Iu));
  for (int v = 0; v < P.V; v++)
    for (int i = 0; i < P.Iu; i++)
      for (int w = 0; w < P.W; w++)
        if (m.eval(opw[v][w][i], true).is_true()) {
          if (out.assign[v][i] < 0)
            out.assign[v][i] = w;
          out.warps[v][i].push_back(w);
        }

  out.sched.solved = true;
  for (int v = 0; v < P.V; v++)
    for (int i = 0; i < P.Iu; i++)
      for (int t = 0; t <= P.L; t++) {
        if (inOpRange(v, t) && m.eval(opv[v][i][t], true).is_true())
          out.sched.op[{v, i, t}] = 1;
        if (m.eval(liv[v][i][t], true).is_true())
          out.sched.live[{v, i, t}] = 1;
      }
  return out;
}

} // namespace twill
