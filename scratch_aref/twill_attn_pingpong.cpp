#include "triton/Analysis/TwillTypes.h"

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

using namespace twill;
// Cost normalization
static int norm(int rawCycles) {
  int c = (rawCycles + 500) / 1000;
  return c < 1 ? 1 : c;
}

static void dump(const PingPongSMTProblem &P, const PingPongAssignment &A,
                 const std::vector<std::string> &names) {
  if (!A.solved) {
    printf("UNSAT / no schedule\n");
    return;
  }
  int makespan = 0;
  for (const auto &kv : A.sched.op) {
    int v = std::get<0>(kv.first), t = std::get<2>(kv.first);
    int done = t + P.cycles[v];
    if (done > makespan)
      makespan = done;
  }
  printf("\n== makespan = %d ==\n", makespan);
  printf("\n== (op, iter) -> warp assignment ==\n");
  for (int v = 0; v < P.V; v++)
    for (int i = 0; i < P.Iu; i++) {
      const auto &ws = A.warps[v][i];
      printf("  v%d/i%d  ", v, i);
      for (size_t k = 0; k < ws.size(); k++)
        printf("%sw%d", k ? "," : "", ws[k]);
      printf("   op %s, iteration i%d is assigned to warp", names[v].c_str(), i);
      for (size_t k = 0; k < ws.size(); k++)
        printf("%s w%d", k ? "," : "", ws[k]);
      if (v < (int)P.is_mma.size() && P.is_mma[v])
        printf("   [MMA]");
      else if (v < (int)P.is_softmax.size() && P.is_softmax[v])
        printf("   [softmax]");
      printf("\n");
    }
  printf("\n== per-warp timeline (cols t=0..%d, cell=<op>.<iter>) ==\n", P.L);
  printf("        ");
  for (int t = 0; t <= P.L; t++)
    printf("%-3d", t);
  printf("\n");
  for (int w = 0; w < P.W; w++) {
    printf("  w%-2d | ", w);
    for (int t = 0; t <= P.L; t++) {
      int mv = -1, mi = -1;
      for (int v = 0; v < P.V && mv < 0; v++)
        for (int i = 0; i < P.Iu; i++) {
          const auto &ws = A.warps[v][i];
          bool here = false;
          for (int w2 : ws)
            if (w2 == w)
              here = true;
          if (!here)
            continue;
          if (A.sched.op.count({v, i, t})) {
            mv = v;
            mi = i;
            break;
          }
        }
      if (mv >= 0) {
        char buf[8];
        snprintf(buf, sizeof(buf), "%d.%d", mv, mi);
        printf("%-3s", buf);
      } else
        printf(" . ");
    }
    printf("\n");
  }
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);

  const int V = 9;
  //             K    V    ldK  ldV  QK    max  exp  l_i  PV
  std::vector<int> cyc = {1, 1, 1,  1,  3, 1,  3,  1,  3};

  std::vector<Edge> edges = {
      {0, 2, 1, 2},        // cp.async K -> local_load K (2-stage smem buffer)
      {1, 3, 1, 2},        // cp.async V -> local_load V
      {2, 4, cyc[2], 0},   // local_load K -> tt.dot QK
      {3, 8, cyc[3], 0},   // local_load V -> tt.dot PV
      {4, 5, cyc[4], 0},   // QK -> reduce max
      {5, 6, cyc[5], 0},   // reduce max -> exp
      {6, 7, cyc[6], 0},   // exp -> reduce l_i
      {6, 8, cyc[6], 0},   // exp -> P -> tt.dot PV
      {8, 8, cyc[8], 1},   // PV -> PV (fp32 accumulator, loop-carried) -> sets II
      {5, 5, 1, 1},        // m_i recurrence (maxnumf)
      {7, 7, 1, 1},        // l_i recurrence (addf)
  };
  std::vector<bool> blk = {true,  true,  false, false, false, false,
                           false, false, false, false, false};
  std::vector<std::string> names = {"cp.async K",  "cp.async V", "local_load K",
                                    "local_load V", "tt.dot QK",  "reduce max",
                                    "math.exp",     "reduce l_i", "tt.dot PV"};


  int foundII = -1, foundL = -1;
  Schedule M;
  int iiLB = 1;
  for (const Edge &e : edges)
    if (e.delta >= 1) {
      int b = (e.d + e.delta - 1) / e.delta;
      if (b > iiLB)
        iiLB = b;
    }

  for (int II = iiLB; II <= iiLB + 40 && foundII < 0; ++II) {
    ILPProblem ilp;
    ilp.V = V;
    ilp.edges = edges;
    ilp.cycles = cyc;
    ilp.F = 0;
    ilp.M = 0;
    ilp.II = II;
    ilp.L = II;
    ilp.deriveIu();
    Schedule s = solveILP(ilp);
    if (!s.solved)
      continue;
    bool allPlaced = true;
    for (int v = 0; v < V && allPlaced; ++v) {
      bool placed = false;
      for (int t = 0; t <= II; ++t)
        if (s.op.count({v, 0, t})) {
          placed = true;
          break;
        }
      allPlaced = placed;
    }
    if (allPlaced) {
      foundII = II;
      foundL = II;
      M = s;
    }
  }
  printf("== STEP-2 ILP (Cbc) ==\n");
  if (foundII < 0) {
    printf("ILP: no feasible modulo schedule found\n");
    return 1;
  }
  int ilpIu = (foundL + foundII - 1) / foundII;
  printf("MII = %d  (PV-bound: tt.dot PV -> tt.dot PV dist=1 lat=%d)\n", foundII,
         cyc[8]);
  printf("modulo schedule: II=%d L=%d Iu=%d ; issue time M(v):\n", foundII,
         foundL, ilpIu);
  for (int v = 0; v < V; ++v) {
    int at = -1;
    for (int t = 0; t <= foundL; ++t)
      if (M.op.count({v, 0, t})) {
        at = t;
        break;
      }
    printf("  M(%-12s) = %d\n", names[v].c_str(), at);
  }

  PingPongSMTProblem P;
  P.V = V;
  P.cycles = cyc;
  P.edges = edges;
  P.blocking = blk;
  P.II = foundII;
  P.Iu = 2;
  P.L = 2 * (cyc[4] + cyc[5] + cyc[6] + cyc[8]) + 4;

  P.regs = {0, 0, 2, 2, 16, 4, 4, 4, 24};
  P.spillcost = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  P.is_mma = {false, false, false, false, true, false, false, false, true};
  P.is_softmax = {false, false, false, false, false, true, true, true, false};
  // Copy ops: cp.async K/V (global->smem) and local_load K/V (smem->reg).
  P.is_copy = {true, true, true, true, false, false, false, false, false};

  P.W = 3;
  P.reg_limit = 128;


  P.tc_single_issue = false;
  P.pingpong_alternate = false;
  P.pin_op_single_warp = true; // original Twill opw[v,w]: one warp per op, all iters
  P.enforce_consumer_warp = true;
  P.warp_uniqueness = true;
  P.minimize_makespan = true;   // optimal: minimize makespan
  P.opt_timeout_ms = 120000;
  // Circular buffer, depth D=2
  P.circular_buffer = true;
  P.buf_depth = 2;
  P.bytes_per_tile = 1;
  P.smem_capacity = 2;

  printf("\n== STEP-3 SMT ping-pong + circular buffer (triton_tem_fused_no_exp2.ttgir) ==\n");
  printf("V=%d W=%d II=%d L=%d Iu=%d  pingpong_alternate=%d circular_buffer=%d(D=%d)\n",
         P.V, P.W, P.II, P.L, P.Iu, P.pingpong_alternate, P.circular_buffer,
         P.buf_depth);
  printf("normalized cycles: QK=%d max=%d exp=%d l_i=%d PV=%d\n", cyc[4], cyc[5],
         cyc[6], cyc[7], cyc[8]);

  PingPongAssignment A = solveSMTPingPong(P);
  dump(P, A, names);
  return 0;
}
