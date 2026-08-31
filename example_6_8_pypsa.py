"""
Example 6.8 - 4-Bus Power System (Power System Engineering)
============================================================
4-bus system solved using PyPSA power flow.

System Data (from Table 6.5 & Fig. 6.7):
------------------------------------------
Bus 1: Slack bus,  V = 1.04∠0° pu
Bus 2: PQ bus,     P =  0.5 pu,  Q = -0.2 pu  (net injection)
Bus 3: PQ bus,     P = -1.0 pu,  Q =  0.5 pu  (net injection)
Bus 4: PQ bus,     P =  0.3 pu,  Q = -0.1 pu  (net injection)

Y-BUS (from the book):
-----------------------
Y11 =  3 - j9      Y12 = -2 + j6      Y13 = -1 + j3     Y14 =  0
Y21 = -2 + j6      Y22 =  3.666-j11   Y23 = -0.666+j2   Y24 = -1 + j3
Y31 = -1 + j3      Y32 = -0.666+j2    Y33 =  3.666-j11  Y34 = -2 + j6
Y41 =  0           Y42 = -1 + j3      Y43 = -2 + j6     Y44 =  3 - j9

Line admittances (negatives of off-diagonal Y-BUS entries):
------------------------------------------------------------
y12 = 2 - j6    =>  z12 = 0.05  + j0.15
y13 = 1 - j3    =>  z13 = 0.10  + j0.30
y23 = 0.666-j2  =>  z23 = 0.150 + j0.45
y24 = 1 - j3    =>  z24 = 0.10  + j0.30
y34 = 2 - j6    =>  z34 = 0.05  + j0.15

No line between bus 1 and bus 4 (Y14 = Y41 = 0).
"""

import numpy as np
import pandas as pd
import pypsa

# ── Base quantities ─────────────────────────────────────────────────────────
S_base = 100e6   # 100 MVA
V_base = 1.0     # 1.0 pu (already in pu system)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  CREATE PYPSA NETWORK
# ═══════════════════════════════════════════════════════════════════════════
net = pypsa.Network()
net.set_snapshots(["now"])   # single time-step (steady-state)

# ── Add buses ────────────────────────────────────────────────────────────────
bus_list = ["Bus1", "Bus2", "Bus3", "Bus4"]
for name in bus_list:
    net.add("Bus", name, v_nom=1.0)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  TRANSMISSION LINES
#     z = r + jx  (per-unit, already on 100 MVA base)
#     Derived from Y-BUS off-diagonal elements: z_ij = 1 / y_ij
#
#     y12 = 2 - j6   => z12 = 1/(2-j6) = (2+j6)/40   = 0.05 + j0.15
#     y13 = 1 - j3   => z13 = 1/(1-j3) = (1+j3)/10   = 0.10 + j0.30
#     y23 = 0.666-j2 => z23 = (0.666+j2)/4.443        = 0.15 + j0.45
#     y24 = 1 - j3   => z24 = 0.10 + j0.30
#     y34 = 2 - j6   => z34 = 0.05 + j0.15
# ═══════════════════════════════════════════════════════════════════════════
lines = [
    #  name     bus0      bus1      r       x
    ("L12", "Bus1", "Bus2",  0.05,   0.15),
    ("L13", "Bus1", "Bus3",  0.10,   0.30),
    ("L23", "Bus2", "Bus3",  0.150,  0.45),
    ("L24", "Bus2", "Bus4",  0.10,   0.30),
    ("L34", "Bus3", "Bus4",  0.05,   0.15),
]

for name, bus0, bus1, r, x in lines:
    net.add(
        "Line",
        name,
        bus0=bus0,
        bus1=bus1,
        r=r,
        x=x,
        s_nom=9999,   # no thermal limit for power-flow study
    )

# ═══════════════════════════════════════════════════════════════════════════
# 3.  SLACK BUS  (Bus 1 -> Generator acting as slack)
# ═══════════════════════════════════════════════════════════════════════════
net.add(
    "Generator",
    "G1_slack",
    bus="Bus1",
    control="Slack",
    v_set_pu=1.04,
    p_nom=9999,
    p_set=0.0,
)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  PQ BUSES (net power injections via Load component)
#
#     PyPSA Load uses CONSUMER sign convention:
#       p_set > 0  =>  absorbs real power
#       q_set > 0  =>  absorbs reactive power
#
#     Net injection (book convention):
#       Bus2: P_net = +0.5, Q_net = -0.2
#       Bus3: P_net = -1.0, Q_net = +0.5
#       Bus4: P_net = +0.3, Q_net = -0.1
#
#     => Load p_set = -P_net, q_set = -Q_net
# ═══════════════════════════════════════════════════════════════════════════
pq_data = {
    "Bus2": ( 0.5,  -0.2),
    "Bus3": (-1.0,   0.5),
    "Bus4": ( 0.3,  -0.1),
}

for bus, (p_net, q_net) in pq_data.items():
    net.add(
        "Load",
        f"Load_{bus}",
        bus=bus,
        p_set=-p_net,
        q_set=-q_net,
    )

# ═══════════════════════════════════════════════════════════════════════════
# 5.  GAUSS-SEIDEL ITERATION (manual, 1 iteration - matches book)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("   MANUAL GAUSS-SEIDEL POWER FLOW (1 iteration, flat start)")
print("=" * 65)

# Full 4x4 Y-BUS matrix
Y = np.array([
    [ 3-9j,       -2+6j,      -1+3j,      0+0j      ],
    [-2+6j,      3.666-11j,  -0.666+2j,  -1+3j      ],
    [-1+3j,     -0.666+2j,   3.666-11j,  -2+6j      ],
    [ 0+0j,      -1+3j,      -2+6j,       3-9j      ],
], dtype=complex)

# Specified net complex power injections for PQ buses (indices 1,2,3)
S_spec = np.array([0.5-0.2j, -1.0+0.5j, 0.3-0.1j])

# Flat start: all voltages = 1.0 + j0
V = np.array([1.04+0j, 1.0+0j, 1.0+0j, 1.0+0j])

print("\nInitial voltages (flat start):")
for i, v in enumerate(V):
    print(f"  Bus {i+1}: |V| = {abs(v):.4f} pu,  delta = {np.angle(v, deg=True):.4f} deg")

# One Gauss-Seidel sweep (update buses 2, 3, 4 in order)
print("\n--- Gauss-Seidel Iteration 1 ---")
for k in range(1, 4):       # bus index 1,2,3 (0-based)
    # GS formula: V_k = (1/Y_kk) * [ conj(S_k)/conj(V_k) - sum_{m!=k} Y_km*V_m ]
    sigma = sum(Y[k, m] * V[m] for m in range(4) if m != k)
    V[k] = (1.0 / Y[k, k]) * (np.conj(S_spec[k - 1]) / np.conj(V[k]) - sigma)
    print(f"  Bus {k+1}: |V| = {abs(V[k]):.6f} pu,  delta = {np.angle(V[k], deg=True):.6f} deg")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  PYPSA NEWTON-RAPHSON POWER FLOW (converged)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("   PYPSA NEWTON-RAPHSON POWER FLOW (converged solution)")
print("=" * 65)

pf_result = net.pf()

# ── Bus voltage results ──────────────────────────────────────────────────
print("\nBus Voltage Results (converged):")
print("-" * 55)
print(f"  {'Bus':<10} {'|V| (pu)':<15} {'Angle (deg)':<15}")
print("-" * 55)

for b in bus_list:
    v_mag   = net.buses_t.v_mag_pu.loc["now", b]
    v_angle = net.buses_t.v_ang.loc["now", b] * (180.0 / np.pi)
    print(f"  {b:<10} {v_mag:<15.6f} {v_angle:<15.6f}")

# ── Line power flows ─────────────────────────────────────────────────────
print("\nLine Power Flows:")
print("-" * 65)
print(f"  {'Line':<8} {'From':<8} {'To':<8} {'P (pu)':<14} {'Q (pu)':<14}")
print("-" * 65)
for ln in net.lines.index:
    p0 = net.lines_t.p0.loc["now", ln]
    q0 = net.lines_t.q0.loc["now", ln]
    b0 = net.lines.loc[ln, "bus0"]
    b1 = net.lines.loc[ln, "bus1"]
    print(f"  {ln:<8} {b0:<8} {b1:<8} {p0:<14.6f} {q0:<14.6f}")

# ── Slack bus generation ─────────────────────────────────────────────────
p_slack = net.generators_t.p.loc["now", "G1_slack"]
q_slack = net.generators_t.q.loc["now", "G1_slack"]
print(f"\nSlack Bus (Bus 1) Generation:")
print(f"  P_slack = {p_slack:.6f} pu  ({p_slack * S_base / 1e6:.4f} MW)")
print(f"  Q_slack = {q_slack:.6f} pu  ({q_slack * S_base / 1e6:.4f} MVAR)")

# ── Summary comparison table ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("   SUMMARY: GS 1st Iteration  vs  Converged NR (PyPSA)")
print("=" * 80)
print(f"\n  {'Bus':<8} {'Type':<8} {'|V| NR':<13} {'d NR':<13} {'|V| GS-1':<13} {'d GS-1'}")
print("-" * 72)

bus_types = ["Slack", "PQ", "PQ", "PQ"]
for i, b in enumerate(bus_list):
    v_nr = net.buses_t.v_mag_pu.loc["now", b]
    d_nr = net.buses_t.v_ang.loc["now", b] * (180.0 / np.pi)
    v_gs = abs(V[i])
    d_gs = np.angle(V[i], deg=True)
    print(f"  {b:<8} {bus_types[i]:<8} {v_nr:<13.6f} {d_nr:<13.6f} {v_gs:<13.6f} {d_gs:.6f}")

print("\nNote: GS-1 = 1 Gauss-Seidel iteration from flat start (book Eq. 6.41)")
print("      NR    = fully converged Newton-Raphson (PyPSA pf())\n")

# ── Verify Y-BUS from PyPSA (optional sanity check) ─────────────────────
print("=" * 65)
print("   Y-BUS VERIFICATION  (from line impedances)")
print("=" * 65)
Y_check = np.zeros((4, 4), dtype=complex)
for name, bus0, bus1, r, x in lines:
    i = int(bus0[-1]) - 1
    j = int(bus1[-1]) - 1
    y_line = 1.0 / complex(r, x)
    Y_check[i, i] += y_line
    Y_check[j, j] += y_line
    Y_check[i, j] -= y_line
    Y_check[j, i] -= y_line

print("\nY-BUS (recomputed from line data):")
for row in Y_check:
    vals = "  ".join(f"{v.real:+.3f}{v.imag:+.3f}j" for v in row)
    print(f"  [ {vals} ]")
