"""
IEEE / WSCC 9-bus, 3-machine system - AC load flow (Newton-Raphson) in PyPSA.

Data taken from the Simulink diagram (MathWorks "IEEE 9-Bus System"):
  * Bus base voltages : Bus1 16.5 kV, Bus2 18 kV, Bus3 13.8 kV, Bus4-9 230 kV
  * Gen1 @ Bus1 : Swing, V = 1.040 pu
    Gen2 @ Bus2 : PV,    V = 1.025 pu, P = 163 MW
    Gen3 @ Bus3 : PV,    V = 1.025 pu, P =  85 MW
  * Load A 125 MW + j50 MVAr @ Bus5
    Load B  90 MW + j30 MVAr @ Bus6
    Load C 100 MW + j35 MVAr @ Bus8
    (the 50 MW switched load at Bus6 is open in the load flow -> 0.0 MW in the diagram)
  * Transformers TF 4-1, TF 7-2, TF 9-3 (generator bus <-> 230 kV bus)
  * Line lengths: 4-5 50 km, 4-6 50 km, 5-7 100 km, 6-9 100 km, 7-8 50 km, 8-9 100 km

NOT shown in the diagram (standard WSCC values used, 100 MVA base):
  * line R, X, B (pu) and transformer X (pu), transformer MVA ratings
Frequency (60 Hz) does not enter the power flow when impedances are in ohm/siemens.
"""
import logging
import warnings
import numpy as np
import pandas as pd
import pypsa

logging.getLogger("pypsa").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 140)

# ------------------------------------------------------------------ settings
LOAD_MODEL = "constant_z"   # "constant_z" reproduces the Simulink diagram
                            # "constant_p" = textbook constant-power loads
PHASE_SHIFT_DEG = -30.0     # transformer shift: 230 kV side leads gen side by 30 deg
S_BASE = 100.0              # MVA
V_HV = 230.0                # kV
Z_BASE = V_HV**2 / S_BASE   # ohm

# ------------------------------------------------------------------ data
BUS_KV   = {1: 16.5, 2: 18.0, 3: 13.8}                   # others 230 kV
V_SET    = {1: 1.040, 2: 1.025, 3: 1.025}                # gen buses (pu)
LOADS    = [("Load A", 5, 125.0, 50.0),
            ("Load B", 6,  90.0, 30.0),
            ("Load C", 8, 100.0, 35.0)]                  # name, bus, MW, MVAr
GENS     = [("G1", 1, "Slack", 0.0),                     # P of slack is computed
            ("G2", 2, "PV", 163.0),
            ("G3", 3, "PV", 85.0)]
# name, from (gen bus), to (230 kV bus), X pu on 100 MVA, MVA rating (standard)
TRAFOS   = [("TF 4-1", 1, 4, 0.0576, 247.5),
            ("TF 7-2", 2, 7, 0.0625, 192.0),
            ("TF 9-3", 3, 9, 0.0586, 128.0)]
# name, bus0, bus1, R pu, X pu, B pu (total charging), length km
LINES    = [("L4-5", 4, 5, 0.0100, 0.0850, 0.176,  50),
            ("L4-6", 4, 6, 0.0170, 0.0920, 0.158,  50),
            ("L5-7", 5, 7, 0.0320, 0.1610, 0.306, 100),
            ("L6-9", 6, 9, 0.0390, 0.1700, 0.358, 100),
            ("L7-8", 7, 8, 0.0085, 0.0720, 0.149,  50),
            ("L8-9", 8, 9, 0.0119, 0.1008, 0.209, 100)]

# Simulink load-flow results read from the diagram (validation targets)
REF_V   = [1.040, 1.025, 1.025, 1.026, 0.996, 1.012, 1.025, 1.015, 1.032]
REF_ANG = [0.00, 8.89, 4.24, 27.64, 25.81, 26.01, 33.33, 30.25, 31.55]
REF_GP  = [76.4, 163.0, 85.0]
REF_GQ  = [27.5, 7.3, -9.8]


def build_network():
    n = pypsa.Network()
    for i in range(1, 10):
        n.add("Bus", f"Bus {i}", v_nom=BUS_KV.get(i, V_HV),
              v_mag_pu_set=V_SET.get(i, 1.0))

    for name, bus, p, q in LOADS:
        if LOAD_MODEL == "constant_p":
            n.add("Load", name, bus=f"Bus {bus}", p_set=p, q_set=q)
        else:  # constant impedance, rated P,Q at 1.0 pu (as in the Simulink load blocks)
            n.add("ShuntImpedance", name, bus=f"Bus {bus}",
                  g=p / V_HV**2, b=-q / V_HV**2)

    for name, bus, ctrl, p in GENS:
        n.add("Generator", name, bus=f"Bus {bus}", control=ctrl, p_set=p)

    for name, b0, b1, x, s in TRAFOS:
        # PyPSA takes transformer x on the transformer's own s_nom
        n.add("Transformer", name, bus0=f"Bus {b0}", bus1=f"Bus {b1}",
              x=x * s / S_BASE, r=0.0, s_nom=s, phase_shift=PHASE_SHIFT_DEG)

    for name, b0, b1, r, x, b, km in LINES:
        n.add("Line", name, bus0=f"Bus {b0}", bus1=f"Bus {b1}",
              r=r * Z_BASE, x=x * Z_BASE, b=b / Z_BASE, length=km)
    return n


def run_pf(n, label):
    info = n.pf()
    ok = bool(info.converged.all().all())
    it = int(info.n_iter.max().max())
    print(f"[{label}] converged = {ok}, Newton-Raphson iterations = {it}")
    return ok


def load_pq(n, snap):
    if LOAD_MODEL == "constant_p":
        return n.loads_t.p.loc[snap], n.loads_t.q.loc[snap]
    # PyPSA reports an inductive shunt with negative q -> flip so that Q > 0 = absorbed
    return n.shunt_impedances_t.p.loc[snap], -n.shunt_impedances_t.q.loc[snap]


def summarize(n):
    snap = n.snapshots[0]
    r = {}
    r["V"]   = n.buses_t.v_mag_pu.loc[snap]
    r["ang"] = np.degrees(n.buses_t.v_ang.loc[snap])
    r["GP"], r["GQ"] = n.generators_t.p.loc[snap], n.generators_t.q.loc[snap]
    r["LP"], r["LQ"] = load_pq(n, snap)

    def branch(comp):
        t = getattr(n, comp + "_t")
        df = pd.DataFrame({"P0_MW": t.p0.loc[snap], "Q0_MVAr": t.q0.loc[snap],
                           "P1_MW": t.p1.loc[snap], "Q1_MVAr": t.q1.loc[snap]})
        df["S0_MVA"] = np.hypot(df.P0_MW, df.Q0_MVAr)
        df["Ploss_MW"] = df.P0_MW + df.P1_MW
        return df

    r["lines"], r["trafos"] = branch("lines"), branch("transformers")
    r["Ploss"] = r["lines"].Ploss_MW.sum() + r["trafos"].Ploss_MW.sum()
    return r


def report(r, title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)
    print("\n--- Bus voltages ---")
    print(pd.DataFrame({"V_pu": r["V"], "V_kV": r["V"] * n_kv(r["V"].index),
                        "Angle_deg": r["ang"]}))
    print("\n--- Generators ---")
    print(pd.DataFrame({"P_MW": r["GP"], "Q_MVAr": r["GQ"]}))
    print("\n--- Loads (actual consumption at solved voltage) ---")
    print(pd.DataFrame({"P_MW": r["LP"], "Q_MVAr": r["LQ"]}))
    print("\n--- Lines (P0/Q0 sending end, P1/Q1 receiving end) ---")
    print(r["lines"])
    print("\n--- Transformers (bus0 = generator side) ---")
    print(r["trafos"])
    PG, PL = r["GP"].sum(), r["LP"].sum()
    print("\n--- Power balance ---")
    print(f"Total generation  P_G    = {PG:.4f} MW")
    print(f"Total load        P_L    = {PL:.4f} MW")
    print(f"Branch losses     P_loss = {r['Ploss']:.4f} MW  (lines + transformers)")
    print(f"Check P_G - P_L - P_loss = {PG - PL - r['Ploss']:.2e} MW")
    QG, QL = r["GQ"].sum(), r["LQ"].sum()
    Qbr = (r["lines"].Q0_MVAr + r["lines"].Q1_MVAr).sum() \
        + (r["trafos"].Q0_MVAr + r["trafos"].Q1_MVAr).sum()   # net absorbed (<0: line charging dominates)
    print(f"Reactive: Q_G = {QG:.4f}, Q_L = {QL:.4f}, branch net = {Qbr:.4f} MVAr, "
          f"check Q_G - Q_L - Q_branch = {QG - QL - Qbr:.2e}")


def n_kv(idx):
    return pd.Series({f"Bus {i}": BUS_KV.get(i, V_HV) for i in range(1, 10)}).reindex(idx)


# =================================================================== BASE CASE
base = build_network()
ok = run_pf(base, "Base case")
rb = summarize(base)
report(rb, f"BASE CASE  (loads: {LOAD_MODEL})")

print("\n" + "=" * 64 + "\nCOMPARISON WITH SIMULINK DIAGRAM\n" + "=" * 64)
print(pd.DataFrame({"V_pypsa": rb["V"].values, "V_sim": REF_V,
                    "Ang_pypsa": rb["ang"].values, "Ang_sim": REF_ANG},
                   index=rb["V"].index))
print(pd.DataFrame({"P_pypsa": rb["GP"].values, "P_sim": REF_GP,
                    "Q_pypsa": rb["GQ"].values, "Q_sim": REF_GQ},
                   index=rb["GP"].index))
print("\nSmall differences (<~0.001 pu, <~0.05 deg, <~1 MW at the slack) are expected:"
      " line/transformer data are not printed in the diagram.")

# ============================================================ N-1: LINE 4-5 OUT
cont = build_network()
cont.remove("Line", "L4-5")
ok_c = run_pf(cont, "N-1, line 4-5 removed")

if ok_c:
    rc = summarize(cont)
    report(rc, "CONTINGENCY: LINE 4-5 REMOVED")
    print("\n" + "=" * 64 + "\nBASE vs N-1\n" + "=" * 64)
    print(pd.DataFrame({"V_base": rb["V"], "V_N-1": rc["V"],
                        "dV": rc["V"] - rb["V"]}))
    print(pd.DataFrame({"P_base": rb["GP"], "P_N-1": rc["GP"],
                        "Q_base": rb["GQ"], "Q_N-1": rc["GQ"]}))
    print(pd.DataFrame({"S_base_MVA": rb["lines"].S0_MVA,
                        "S_N-1_MVA": rc["lines"].S0_MVA}))
    print(f"\nLosses: base = {rb['Ploss']:.4f} MW, N-1 = {rc['Ploss']:.4f} MW, "
          f"change = {rc['Ploss'] - rb['Ploss']:+.4f} MW")
else:
    print("N-1 load flow did NOT converge - report this as a finding.")