# -*- coding: utf-8 -*-
"""
Wind Energy Integration Study – 4-Bus System in PyPSA
======================================================

A self-contained demonstration that PyPSA can co-dispatch a wind farm
alongside a conventional synchronous generator over a 24-hour horizon
using Optimal Power Flow (linear OPF).

System topology
~~~~~~~~~~~~~~~
         Bus 1 ──── Bus 2
         (Conv)      (Wind Farm + Load)
          │              │
         Bus 4 ──── Bus 3
         (Load)      (junction)

 • Bus 1  – Conventional generator (coal/gas), 200 MW, slack bus
 • Bus 2  – 100 MW wind farm  +  60 MW / 20 MVAr load
 • Bus 3  – Pure junction (no gen / no load)
 • Bus 4  – 100 MW / 35 MVAr load (city)
 • Lines  – 230 kV, four transmission lines forming a ring

Key difference from solar:
  Wind generation is NOT correlated with the sun — it blows day AND
  night, with higher speeds typically during evening / early morning.
  The profile used here mimics a realistic onshore wind pattern with
  variable output across all 24 hours.

The script:
  1.  Builds the network and adds a realistic 24-hour wind profile.
  2.  Runs a 24-hour Optimal Power Flow (linear OPF) with HiGHS solver.
  3.  Prints a clear hour-by-hour dispatch table with wind penetration %.
  4.  Saves four publication-quality plots:
        • Generation dispatch stack  (MW vs hour)
        • Wind available vs dispatched (+ curtailment if any)
        • Line loading (%)
        • Energy share pie chart

Author:  PSModel – Phase 2
Date:    October 2026
"""

import logging
import warnings
import sys
import io
import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from   matplotlib.ticker import MaxNLocator

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.getLogger("pypsa").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.width", 120)

# ====================================================================
#                        SYSTEM  PARAMETERS
# ====================================================================

S_BASE   = 100.0       # MVA base
V_NOM    = 230.0       # kV  (all buses at same voltage level)
HOURS    = 24          # simulation horizon

# Generator data
CONV_PNOM     = 200.0   # MW   – conventional plant capacity
CONV_COST     = 3400.0  # ₹/MWh (approx ₹3.40/kWh – coal thermal fuel + O&M)
WIND_PNOM     = 100.0   # MW   – wind farm rated capacity
WIND_COST     = 255.0   # ₹/MWh (approx ₹0.25/kWh – variable O&M: turbine maintenance, lubrication)

# Load data  (base / peak values – actual profiles are time-varying)
LOAD_BUS2_P   = 60.0   # MW  – industrial base
LOAD_BUS2_Q   = 20.0   # MVAr
LOAD_BUS4_P   = 100.0  # MW  – city base
LOAD_BUS4_Q   = 35.0   # MVAr

# Line data  (typical 230 kV overhead line per-km values)
LINE_R_PER_KM = 0.032  # ohm/km
LINE_X_PER_KM = 0.32   # ohm/km
LINE_B_PER_KM = 3.5e-6 # S/km
LINE_S_NOM    = 200.0  # MVA thermal rating

LINES = [
    # name,    from,    to,      length_km
    ("L1-2",  "Bus 1", "Bus 2",  80),
    ("L2-3",  "Bus 2", "Bus 3",  60),
    ("L3-4",  "Bus 3", "Bus 4",  70),
    ("L4-1",  "Bus 4", "Bus 1",  50),
]


def load_profile(hours: int) -> dict:
    """
    Generate realistic 24-hour load profiles for both load buses.

    Returns a dict with keys 'bus2' and 'bus4', each an ndarray of
    MW values.  Daily mean equals the base MW constant.

    Bus 4 – City:  dual-peak (morning + evening), overnight trough.
    Bus 2 – Industrial:  flat weekday, ramp up/down at shift changes.
    """
    t = np.arange(hours, dtype=float)

    # City / residential (Bus 4)
    city = (0.85
            + 0.10 * np.sin(np.pi * (t - 5) / 12)
            + 0.12 * np.exp(-0.5 * ((t - 19) / 1.8)**2)
            - 0.15 * np.exp(-0.5 * ((t - 3) / 2.5)**2))
    city = city / city.mean()
    city_mw = LOAD_BUS4_P * city

    # Industrial (Bus 2)
    ind = (0.90
           + 0.12 * np.tanh((t - 6.5) / 1.0)
           - 0.12 * np.tanh((t - 20) / 1.5)
           - 0.08 * np.exp(-0.5 * ((t - 3) / 2.0)**2))
    ind = ind / ind.mean()
    ind_mw = LOAD_BUS2_P * ind

    return {"bus2": ind_mw, "bus4": city_mw}


def wind_profile(hours: int) -> np.ndarray:
    """
    Generate a realistic 24-hour onshore wind capacity-factor profile.

    Wind characteristics (unlike solar):
      • Generates day AND night
      • Typically stronger in late evening and early morning
      • Variable / gusty with no smooth bell shape
      • Never reaches full rated output for long periods

    Returns an array of length `hours` with values in [0, 1].
    """
    # Base diurnal pattern: stronger at night, weaker midday
    t = np.arange(hours, dtype=float)
    diurnal = 0.35 + 0.20 * np.cos(2 * np.pi * (t - 3) / 24)  # peak ~3am

    # Add realistic variability (gusts and lulls)
    # Using a fixed seed for reproducibility in demonstrations
    rng = np.random.default_rng(seed=42)
    variability = rng.normal(0, 0.08, hours)

    # Weather event: a wind ramp-up in the evening (hours 17-22)
    ramp = np.zeros(hours)
    ramp[17:23] = np.array([0.10, 0.18, 0.25, 0.22, 0.15, 0.08])

    cf = diurnal + variability + ramp
    cf = np.clip(cf, 0.05, 0.85)  # min 5% (never fully calm), max 85%
    return cf


# ====================================================================
#                     BUILD  THE  NETWORK
# ====================================================================

def build_network():
    n = pypsa.Network()

    # --- time axis: 24 hourly snapshots ---
    snapshots = pd.date_range("2026-10-04", periods=HOURS, freq="h")
    n.set_snapshots(snapshots)

    # --- buses ---
    for i in range(1, 5):
        n.add("Bus", f"Bus {i}", v_nom=V_NOM)

    # --- conventional generator at Bus 1 (slack equivalent) ---
    n.add("Generator", "Conv Gen (Bus 1)",
          bus="Bus 1",
          p_nom=CONV_PNOM,
          marginal_cost=CONV_COST,       # ₹/MWh
          carrier="conventional")

    # --- wind farm at Bus 2 ---
    cf = wind_profile(HOURS)
    n.add("Generator", "Wind Farm (Bus 2)",
          bus="Bus 2",
          p_nom=WIND_PNOM,
          p_max_pu=cf,                    # time-varying capacity factor
          marginal_cost=WIND_COST,        # free fuel
          carrier="wind")

    # --- loads (time-varying) ---
    lp = load_profile(HOURS)
    n.add("Load", "City Load (Bus 4)",
          bus="Bus 4", p_set=lp["bus4"], q_set=LOAD_BUS4_Q)
    n.add("Load", "Industrial Load (Bus 2)",
          bus="Bus 2", p_set=lp["bus2"], q_set=LOAD_BUS2_Q)

    # --- transmission lines ---
    for name, b0, b1, km in LINES:
        n.add("Line", name,
              bus0=b0, bus1=b1,
              r=LINE_R_PER_KM * km,
              x=LINE_X_PER_KM * km,
              b=LINE_B_PER_KM * km,
              s_nom=LINE_S_NOM,
              length=km)

    n.sanitize()
    return n


# ====================================================================
#                  RUN  OPTIMAL  POWER  FLOW
# ====================================================================

def run_opf(n):
    """Run linear OPF and report numerical method & solver convergence details."""
    status = n.optimize(solver_name="highs")
    if isinstance(status, tuple):
        stat_str, cond_str = status[0], status[1]
    else:
        stat_str, cond_str = str(status), "optimal" if status == "ok" else "failed"

    ok = stat_str == "ok"

    print("\n" + "=" * 72)
    print("  NUMERICAL METHOD & OPTIMIZATION SOLVER DETAILS")
    print("=" * 72)
    print("  Formulation            : 24-Hour Multi-Period Linear Optimal Power Flow (DC-OPF)")
    print("  Optimization Problem   : Continuous Linear Program (LP)")
    print("  PyPSA Optimization API : Linopy frontend -> HiGHS solver backend")
    print("  Numerical Method       : Revised Dual Simplex Method / Presolve Reductions")

    sm = getattr(n.model, "solver_model", None)
    if sm is not None:
        info = sm.getInfo()
        print(f"  Simplex Iterations     : {info.simplex_iteration_count} iterations (solution obtained)")
        if hasattr(info, "ipm_iteration_count") and info.ipm_iteration_count > 0:
            print(f"  Interior Point Iters   : {info.ipm_iteration_count}")
        print(f"  Presolve Status        : {sm.getModelPresolveStatus()}")
        print(f"  Solver Model Status    : {sm.modelStatusToString(sm.getModelStatus())}")
    print(f"  Optimization Status    : {stat_str}, solver condition: {cond_str}")
    print("=" * 72 + "\n")
    return ok


# ====================================================================
#                     RESULTS  &  REPORTING
# ====================================================================

def print_dispatch_table(n):
    """Hour-by-hour dispatch table for the terminal."""
    gen_p = n.generators_t.p
    total_load = n.loads_t.p.sum(axis=1)

    tbl = pd.DataFrame({
        "Hour":          gen_p.index.hour,
        "Conv_MW":       gen_p["Conv Gen (Bus 1)"].values,
        "Wind_MW":       gen_p["Wind Farm (Bus 2)"].values,
        "Total_Gen_MW":  gen_p.sum(axis=1).values,
        "Total_Load_MW": total_load.values,
    })
    tbl["Wind_%"] = 100.0 * tbl.Wind_MW / tbl.Total_Gen_MW
    tbl.loc[tbl.Total_Gen_MW == 0, "Wind_%"] = 0.0
    tbl = tbl.set_index("Hour")

    print("\n" + "=" * 72)
    print("  24-HOUR DISPATCH TABLE  -  4-Bus Wind Integration Study")
    print("=" * 72)
    print(tbl.to_string())

    # --- energy totals ---
    conv_energy  = tbl.Conv_MW.sum()       # MWh (1-h steps)
    wind_energy  = tbl.Wind_MW.sum()
    total_energy = conv_energy + wind_energy
    print("\n--- Energy Summary (24 h) ---")
    print(f"  Conventional : {conv_energy:8.1f} MWh  ({100*conv_energy/total_energy:.1f}%)")
    print(f"  Wind         : {wind_energy:8.1f} MWh  ({100*wind_energy/total_energy:.1f}%)")
    print(f"  Total        : {total_energy:8.1f} MWh")

    # --- capacity factor ---
    wind_cf = wind_energy / (WIND_PNOM * HOURS) * 100
    print(f"\n  Wind capacity factor : {wind_cf:.1f}%")

    # --- cost ---
    conv_cost     = conv_energy * CONV_COST
    wind_cost     = wind_energy * WIND_COST
    actual_cost   = conv_cost + wind_cost
    all_conv_cost = total_energy * CONV_COST
    savings       = all_conv_cost - actual_cost

    print("\n--- Economic Summary (24 h) ---")
    print(f"  Conventional cost (₹{CONV_COST:.1f}/MWh) : ₹{conv_cost:,.0f}  ({100*conv_cost/actual_cost:.1f}%)")
    print(f"  Wind farm O&M cost (₹{WIND_COST:.1f}/MWh) : ₹{wind_cost:,.0f}  ({100*wind_cost/actual_cost:.1f}%)")
    print(f"  Total system operating cost   : ₹{actual_cost:,.0f}")
    print(f"  Without wind (all conv)       : ₹{all_conv_cost:,.0f}")
    print(f"  Net savings from wind farm    : ₹{savings:,.0f}  ({100*savings/all_conv_cost:.1f}% reduction)")

    return tbl


def n_lines_peak(n):
    """Peak power flow and loading for each line."""
    p0_peak = n.lines_t.p0.abs().max()
    loading = 100.0 * p0_peak / n.lines.s_nom
    return pd.DataFrame({
        "Peak_MW": p0_peak,
        "S_nom_MVA": n.lines.s_nom,
        "Loading_%": loading,
        "Length_km": n.lines.length,
    })


def plot_results(n, tbl):
    """Generate four publication-quality plots and save to disk."""
    hours = np.arange(HOURS)
    gen_p = n.generators_t.p
    conv  = gen_p["Conv Gen (Bus 1)"].values
    wind  = gen_p["Wind Farm (Bus 2)"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Wind Energy Integration - 4-Bus System (24-Hour OPF)",
                 fontsize=15, fontweight="bold", y=0.98)

    # -- 1. Stacked dispatch --------------------------------------------------
    ax = axes[0, 0]
    ax.fill_between(hours, 0, conv, color="#4a6fa5", alpha=0.85, label="Conventional")
    ax.fill_between(hours, conv, conv + wind, color="#2ecc71", alpha=0.85, label="Wind")
    total_load = n.loads_t.p.sum(axis=1).values
    ax.plot(hours, total_load, "k--", lw=1.5, label="Total Load")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Generation Dispatch")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 23)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # -- 2. Wind profile: available vs dispatched ------------------------------
    ax = axes[0, 1]
    available = WIND_PNOM * wind_profile(HOURS)
    ax.fill_between(hours, 0, available, color="#2ecc71", alpha=0.3, label="Available")
    ax.fill_between(hours, 0, wind, color="#2ecc71", alpha=0.85, label="Dispatched")
    curtailed = available - wind
    if curtailed.max() > 0.5:
        ax.fill_between(hours, wind, available, color="#e74c3c", alpha=0.5,
                        label="Curtailed")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Wind Power (MW)")
    ax.set_title("Wind Farm: Available vs Dispatched")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 23)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # -- 3. Line loading -------------------------------------------------------
    ax = axes[1, 0]
    line_loading = n.lines_t.p0.abs()
    for col in line_loading.columns:
        s_nom = n.lines.loc[col, "s_nom"]
        pct = 100.0 * line_loading[col].values / s_nom
        ax.plot(hours, pct, marker="o", ms=3, lw=1.4, label=col)
    ax.axhline(100, color="red", ls="--", lw=1, alpha=0.7, label="Thermal limit")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Line Loading (%)")
    ax.set_title("Transmission Line Loading")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, 23)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # -- 4. Energy mix pie chart -----------------------------------------------
    ax = axes[1, 1]
    conv_e  = conv.sum()
    wind_e  = wind.sum()
    sizes   = [conv_e, wind_e]
    labels  = [f"Conventional\n{conv_e:.0f} MWh", f"Wind\n{wind_e:.0f} MWh"]
    colors  = ["#4a6fa5", "#2ecc71"]
    explode = (0, 0.06)
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       explode=explode, autopct="%1.1f%%",
                                       startangle=140, textprops={"fontsize": 9})
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("24-Hour Energy Mix")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = r"d:\psmodel\phase2\wind_integration_results.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\n-> Plots saved: {out}")
    plt.show(block=False)
    plt.pause(2)


# ====================================================================
#                          MAIN
# ====================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("   Wind Energy Integration Study - 4-Bus System (PyPSA OPF)")
    print("=" * 62)

    # Build
    net = build_network()
    print(f"\nNetwork: {len(net.buses)} buses, {len(net.generators)} generators, "
          f"{len(net.loads)} loads, {len(net.lines)} lines")
    print(f"Snapshots: {HOURS} hours")
    print(f"Wind farm rated: {WIND_PNOM} MW (₹{WIND_COST:.1f}/MWh)  |  Conv rated: {CONV_PNOM} MW (₹{CONV_COST:.1f}/MWh)")
    lp = load_profile(HOURS)
    print(f"Total load: {LOAD_BUS2_P + LOAD_BUS4_P} MW avg  "
          f"(varies {lp['bus2'].min() + lp['bus4'].min():.1f} – "
          f"{lp['bus2'].max() + lp['bus4'].max():.1f} MW)")

    # Wind profile preview
    cf = wind_profile(HOURS)
    print(f"\nWind capacity factor range: {cf.min():.2f} - {cf.max():.2f}")
    print(f"Wind daily average CF: {cf.mean():.2f}")

    # Solve
    print("\n--- Running 24-hour Linear OPF ---")
    ok = run_opf(net)

    if ok:
        tbl = print_dispatch_table(net)

        # Line flows summary
        print("\n--- Peak Line Flows ---")
        peak = n_lines_peak(net)
        print(peak.to_string())

        # Plot
        plot_results(net, tbl)
    else:
        print("OPF did NOT converge. Check data / solver.")
