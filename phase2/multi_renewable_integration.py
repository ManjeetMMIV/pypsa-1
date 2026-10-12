# -*- coding: utf-8 -*-
r"""
Multi-Renewable Integration Study - 5-Bus System in PyPSA
==========================================================

A comprehensive demonstration of integrating MULTIPLE renewable energy
sources alongside a conventional generator.
Uses 24-hour Optimal Power Flow (linear OPF).

System topology (5-bus ring with cross-link)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            Bus 1 ---------- Bus 2
          (Conv 200MW)     (Solar 80MWp + Load 60MW)
             |    /             |
             |   /              |
          Bus 5  /           Bus 3
     (Hydro 20MW    \     (Wind 100MW)
      + Load 25MW)   \       |
             |         \     |
          Bus 4 ---------- Bus 3
     (City Load 100MW)

Renewable sources modelled:
  1. Solar PV (80 MWp)     - bell-shaped daytime profile
  2. Wind Farm (100 MW)    - variable day+night, stronger at night
  3. Small Hydro (20 MW)   - steady run-of-river, seasonal variation

The script:
  1.  Builds the 5-bus network with all sources.
  2.  Runs 24-hour linear OPF to find least-cost dispatch.
  3.  Prints hour-by-hour dispatch with renewable penetration %.
  4.  Saves six publication-quality plots.

Author:  PSModel - Phase 2
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
pd.set_option("display.width", 140)

# ====================================================================
#                        SYSTEM  PARAMETERS
# ====================================================================

S_BASE   = 100.0       # MVA base
V_NOM    = 230.0       # kV
HOURS    = 24

# ---- Generators ----
CONV_PNOM     = 200.0   # MW  – conventional plant
CONV_COST     = 45.0    # $/MWh – fuel + O&M + carbon

WIND_PNOM     = 100.0   # MW  – onshore wind farm
WIND_COST     = 3.0     # $/MWh – variable O&M (turbine upkeep, lubrication)

SOLAR_PNOM    = 80.0    # MWp – solar PV farm
SOLAR_COST    = 5.0     # $/MWh – variable O&M (panel cleaning, inverter servicing)

HYDRO_PNOM    = 20.0    # MW  – small run-of-river hydro
HYDRO_COST    = 8.0     # $/MWh – variable O&M (mechanical servicing, water rights)


# ---- Loads (base / peak values – actual profiles are time-varying) ----
LOAD_BUS2_P   = 60.0    # MW – industrial base
LOAD_BUS2_Q   = 20.0    # MVAr
LOAD_BUS4_P   = 100.0   # MW – city base
LOAD_BUS4_Q   = 35.0    # MVAr
LOAD_BUS5_P   = 25.0    # MW – rural base
LOAD_BUS5_Q   = 8.0     # MVAr
TOTAL_LOAD    = LOAD_BUS2_P + LOAD_BUS4_P + LOAD_BUS5_P  # 185 MW avg

# ---- Lines ----
LINE_R_PER_KM = 0.032   # ohm/km
LINE_X_PER_KM = 0.32    # ohm/km
LINE_B_PER_KM = 3.5e-6  # S/km
LINE_S_NOM    = 200.0   # MVA

LINES = [
    # name,    from,    to,      length_km
    ("L1-2",  "Bus 1", "Bus 2",  80),
    ("L2-3",  "Bus 2", "Bus 3",  60),
    ("L3-4",  "Bus 3", "Bus 4",  70),
    ("L4-5",  "Bus 4", "Bus 5",  55),
    ("L5-1",  "Bus 5", "Bus 1",  40),
    ("L1-4",  "Bus 1", "Bus 4",  90),   # cross-link for reliability
]


# ====================================================================
#                     LOAD  PROFILES
# ====================================================================

def load_profile(hours: int) -> dict:
    """
    Generate realistic 24-hour load profiles for all three load buses.

    Returns a dict with keys 'bus2', 'bus4', 'bus5', each an ndarray
    of MW values.  Daily mean equals the base MW constant.

    Bus 4 – City:       dual-peak (morning + evening), overnight trough.
    Bus 2 – Industrial: flat weekday, ramp up/down at shift changes.
    Bus 5 – Rural:      mild dual-peak, lower overnight.
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

    # Rural (Bus 5)
    rural = (0.90
             + 0.08 * np.sin(np.pi * (t - 5) / 12)
             + 0.08 * np.exp(-0.5 * ((t - 19) / 2.0)**2)
             - 0.10 * np.exp(-0.5 * ((t - 3) / 3.0)**2))
    rural = rural / rural.mean()
    rural_mw = LOAD_BUS5_P * rural

    return {"bus2": ind_mw, "bus4": city_mw, "bus5": rural_mw}


# ====================================================================
#                   RENEWABLE  PROFILES
# ====================================================================

def solar_profile(hours: int) -> np.ndarray:
    """Bell-shaped solar: sunrise 6am, peak 85% at noon, sunset 6pm."""
    t = np.arange(hours, dtype=float)
    sunrise, sunset = 6.0, 18.0
    cf = np.zeros(hours)
    day = (t >= sunrise) & (t <= sunset)
    mid = (sunrise + sunset) / 2.0
    sigma = (sunset - sunrise) / 4.5
    cf[day] = 0.85 * np.exp(-0.5 * ((t[day] - mid) / sigma) ** 2)
    return cf


def wind_profile(hours: int) -> np.ndarray:
    """Variable wind: stronger at night, evening ramp, always > 5%."""
    t = np.arange(hours, dtype=float)
    diurnal = 0.35 + 0.20 * np.cos(2 * np.pi * (t - 3) / 24)
    rng = np.random.default_rng(seed=42)
    variability = rng.normal(0, 0.08, hours)
    ramp = np.zeros(hours)
    ramp[17:23] = np.array([0.10, 0.18, 0.25, 0.22, 0.15, 0.08])
    cf = diurnal + variability + ramp
    return np.clip(cf, 0.05, 0.85)


def hydro_profile(hours: int) -> np.ndarray:
    """Run-of-river hydro: fairly steady, slight dip in afternoon."""
    t = np.arange(hours, dtype=float)
    # River flow is largely steady with mild diurnal variation
    cf = 0.70 - 0.05 * np.cos(2 * np.pi * (t - 6) / 24)
    return np.clip(cf, 0.55, 0.80)


# ====================================================================
#                     BUILD  THE  NETWORK
# ====================================================================

def build_network():
    n = pypsa.Network()
    snapshots = pd.date_range("2026-10-04", periods=HOURS, freq="h")
    n.set_snapshots(snapshots)

    # --- buses ---
    for i in range(1, 6):
        n.add("Bus", f"Bus {i}", v_nom=V_NOM)

    # --- conventional generator at Bus 1 ---
    n.add("Generator", "Conventional (Bus 1)",
          bus="Bus 1", p_nom=CONV_PNOM,
          marginal_cost=CONV_COST, carrier="conventional")

    # --- solar PV at Bus 2 ---
    n.add("Generator", "Solar PV (Bus 2)",
          bus="Bus 2", p_nom=SOLAR_PNOM,
          p_max_pu=solar_profile(HOURS),
          marginal_cost=SOLAR_COST, carrier="solar")

    # --- wind farm at Bus 3 ---
    n.add("Generator", "Wind Farm (Bus 3)",
          bus="Bus 3", p_nom=WIND_PNOM,
          p_max_pu=wind_profile(HOURS),
          marginal_cost=WIND_COST, carrier="wind")

    # --- small hydro at Bus 5 ---
    n.add("Generator", "Small Hydro (Bus 5)",
          bus="Bus 5", p_nom=HYDRO_PNOM,
          p_max_pu=hydro_profile(HOURS),
          marginal_cost=HYDRO_COST, carrier="hydro")


    # --- loads (time-varying) ---
    lp = load_profile(HOURS)
    n.add("Load", "Industrial (Bus 2)",
          bus="Bus 2", p_set=lp["bus2"], q_set=LOAD_BUS2_Q)
    n.add("Load", "City (Bus 4)",
          bus="Bus 4", p_set=lp["bus4"], q_set=LOAD_BUS4_Q)
    n.add("Load", "Rural (Bus 5)",
          bus="Bus 5", p_set=lp["bus5"], q_set=LOAD_BUS5_Q)

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
    gen_p = n.generators_t.p
    total_load = n.loads_t.p.sum(axis=1)

    tbl = pd.DataFrame({
        "Hour":     gen_p.index.hour,
        "Conv_MW":  gen_p["Conventional (Bus 1)"].values,
        "Solar_MW": gen_p["Solar PV (Bus 2)"].values,
        "Wind_MW":  gen_p["Wind Farm (Bus 3)"].values,
        "Hydro_MW": gen_p["Small Hydro (Bus 5)"].values,
        "Load_MW":  total_load.values,
    })
    tbl["RE_MW"]  = tbl.Solar_MW + tbl.Wind_MW + tbl.Hydro_MW
    tbl["RE_%"]   = 100.0 * tbl.RE_MW / tbl.Load_MW
    tbl = tbl.set_index("Hour")

    print("\n" + "=" * 100)
    print("  24-HOUR DISPATCH TABLE  -  Multi-Renewable Integration (Solar + Wind + Hydro)")
    print("=" * 100)
    print(tbl.to_string())

    # --- energy totals ---
    conv_e  = tbl.Conv_MW.sum()
    solar_e = tbl.Solar_MW.sum()
    wind_e  = tbl.Wind_MW.sum()
    hydro_e = tbl.Hydro_MW.sum()
    re_e    = solar_e + wind_e + hydro_e
    total_e = conv_e + re_e

    print("\n" + "-" * 50)
    print("  ENERGY SUMMARY (24 hours)")
    print("-" * 50)
    print(f"  Conventional : {conv_e:8.1f} MWh  ({100*conv_e/total_e:.1f}%)")
    print(f"  Solar PV     : {solar_e:8.1f} MWh  ({100*solar_e/total_e:.1f}%)")
    print(f"  Wind         : {wind_e:8.1f} MWh  ({100*wind_e/total_e:.1f}%)")
    print(f"  Small Hydro  : {hydro_e:8.1f} MWh  ({100*hydro_e/total_e:.1f}%)")
    print(f"  ----------------------------------------")
    print(f"  Total RE     : {re_e:8.1f} MWh  ({100*re_e/total_e:.1f}%)")
    print(f"  Total Gen    : {total_e:8.1f} MWh")

    # --- capacity factors ---
    print(f"\n  Capacity Factors:")
    print(f"    Solar : {100*solar_e/(SOLAR_PNOM*HOURS):.1f}%")
    print(f"    Wind  : {100*wind_e/(WIND_PNOM*HOURS):.1f}%")
    print(f"    Hydro : {100*hydro_e/(HYDRO_PNOM*HOURS):.1f}%")

    # --- cost ---
    conv_cost     = conv_e * CONV_COST
    wind_cost     = wind_e * WIND_COST
    solar_cost    = solar_e * SOLAR_COST
    hydro_cost    = hydro_e * HYDRO_COST
    actual_cost   = conv_cost + wind_cost + solar_cost + hydro_cost
    all_conv_cost = total_e * CONV_COST
    savings       = all_conv_cost - actual_cost

    print("\n--- Economic Summary (24 h) ---")
    print(f"  Conventional cost (${CONV_COST:.1f}/MWh) : ${conv_cost:,.0f}  ({100*conv_cost/actual_cost:.1f}%)")
    print(f"  Wind farm O&M cost (${WIND_COST:.1f}/MWh) : ${wind_cost:,.0f}  ({100*wind_cost/actual_cost:.1f}%)")
    print(f"  Solar PV O&M cost (${SOLAR_COST:.1f}/MWh) : ${solar_cost:,.0f}  ({100*solar_cost/actual_cost:.1f}%)")
    print(f"  Small Hydro cost (${HYDRO_COST:.1f}/MWh)  : ${hydro_cost:,.0f}  ({100*hydro_cost/actual_cost:.1f}%)")
    print(f"  Total system operating cost   : ${actual_cost:,.0f}")
    print(f"  Without renewables (all conv) : ${all_conv_cost:,.0f}")
    print(f"  Net savings from renewables   : ${savings:,.0f}  ({100*savings/all_conv_cost:.1f}% reduction)")

    return tbl


def n_lines_peak(n):
    p0_peak = n.lines_t.p0.abs().max()
    loading = 100.0 * p0_peak / n.lines.s_nom
    return pd.DataFrame({
        "Peak_MW": p0_peak, "S_nom_MVA": n.lines.s_nom,
        "Loading_%": loading, "Length_km": n.lines.length,
    })


def plot_results(n, tbl):
    hours = np.arange(HOURS)
    gen_p = n.generators_t.p
    conv  = gen_p["Conventional (Bus 1)"].values
    solar = gen_p["Solar PV (Bus 2)"].values
    wind  = gen_p["Wind Farm (Bus 3)"].values
    hydro = gen_p["Small Hydro (Bus 5)"].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Multi-Renewable Integration - 5-Bus System (24-Hour OPF)",
                 fontsize=16, fontweight="bold", y=0.99)

    # -- 1. Stacked generation dispatch --------------------------------
    ax = axes[0, 0]
    y1 = hydro
    y2 = y1 + solar
    y3 = y2 + wind
    y4 = y3 + conv
    ax.fill_between(hours, 0,  y1, color="#3498db", alpha=0.85, label="Hydro")
    ax.fill_between(hours, y1, y2, color="#f5a623", alpha=0.85, label="Solar")
    ax.fill_between(hours, y2, y3, color="#2ecc71", alpha=0.85, label="Wind")
    ax.fill_between(hours, y3, y4, color="#7f8c8d", alpha=0.85, label="Conventional")
    total_load = n.loads_t.p.sum(axis=1).values
    ax.plot(hours, total_load, "k--", lw=2, label="Total Load")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Generation Dispatch Stack")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlim(0, 23)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # -- 2. Individual RE profiles -------------------------------------
    ax = axes[0, 1]
    ax.plot(hours, solar, "o-", color="#f5a623", ms=4, lw=1.5, label=f"Solar ({solar.sum():.0f} MWh)")
    ax.plot(hours, wind,  "s-", color="#2ecc71", ms=4, lw=1.5, label=f"Wind ({wind.sum():.0f} MWh)")
    ax.plot(hours, hydro, "^-", color="#3498db", ms=4, lw=1.5, label=f"Hydro ({hydro.sum():.0f} MWh)")
    ax.fill_between(hours, 0, solar, color="#f5a623", alpha=0.15)
    ax.fill_between(hours, 0, wind,  color="#2ecc71", alpha=0.15)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Renewable Generation Profiles")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 23)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # -- 3. Cost breakdown (replaces battery plot) ---------------------
    ax = axes[0, 2]
    conv_e  = conv.sum()
    solar_e = solar.sum()
    wind_e  = wind.sum()
    hydro_e = hydro.sum()
    total_e = conv_e + solar_e + wind_e + hydro_e
    conv_cost   = conv_e * CONV_COST
    wind_cost   = wind_e * WIND_COST
    solar_cost  = solar_e * SOLAR_COST
    hydro_cost  = hydro_e * HYDRO_COST
    actual_cost = conv_cost + wind_cost + solar_cost + hydro_cost
    all_conv_cost = total_e * CONV_COST
    savings     = all_conv_cost - actual_cost
    bars = ax.bar(["System\nCost", "All-Conv\nCost", "Savings"],
                  [actual_cost, all_conv_cost, savings],
                  color=["#2ecc71", "#e74c3c", "#3498db"], alpha=0.85, width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + all_conv_cost * 0.01,
                f"${h:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Cost ($)")
    ax.set_title("24-Hour Cost Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # -- 4. Renewable penetration % ------------------------------------
    ax = axes[1, 0]
    re_pct = tbl["RE_%"].values
    ax.fill_between(hours, 0, re_pct, color="#27ae60", alpha=0.4)
    ax.plot(hours, re_pct, "o-", color="#27ae60", lw=2, ms=4)
    ax.axhline(re_pct.mean(), color="#e74c3c", ls="--", lw=1.5,
               label=f"Average: {re_pct.mean():.1f}%")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("RE Penetration (%)")
    ax.set_title("Renewable Energy Penetration")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # -- 5. Line loading -----------------------------------------------
    ax = axes[1, 1]
    line_loading = n.lines_t.p0.abs()
    for col in line_loading.columns:
        s_nom = n.lines.loc[col, "s_nom"]
        pct = 100.0 * line_loading[col].values / s_nom
        ax.plot(hours, pct, marker="o", ms=3, lw=1.3, label=col)
    ax.axhline(100, color="red", ls="--", lw=1, alpha=0.7, label="Thermal limit")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Line Loading (%)")
    ax.set_title("Transmission Line Loading")
    ax.legend(loc="upper right", fontsize=6)
    ax.set_xlim(0, 23)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # -- 6. Energy mix pie chart ---------------------------------------
    ax = axes[1, 2]
    energies = [conv.sum(), solar.sum(), wind.sum(), hydro.sum()]
    labels   = [f"Conv.\n{energies[0]:.0f} MWh",
                f"Solar\n{energies[1]:.0f} MWh",
                f"Wind\n{energies[2]:.0f} MWh",
                f"Hydro\n{energies[3]:.0f} MWh"]
    colors   = ["#7f8c8d", "#f5a623", "#2ecc71", "#3498db"]
    explode  = (0, 0.04, 0.04, 0.04)
    wedges, texts, autotexts = ax.pie(
        energies, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140, textprops={"fontsize": 8})
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("24-Hour Energy Mix")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_renewable_results.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\n-> Plots saved: {out}")
    plt.show(block=False)
    plt.pause(2)


# ====================================================================
#                          MAIN
# ====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Multi-Renewable Integration Study - 5-Bus System (PyPSA OPF)")
    print("  Sources: Solar PV + Wind + Small Hydro")
    print("=" * 70)

    # Build
    net = build_network()
    print(f"\nNetwork: {len(net.buses)} buses, {len(net.generators)} generators, "
          f"{len(net.loads)} loads, {len(net.lines)} lines")
    print(f"Snapshots: {HOURS} hours")
    print(f"\nGeneration capacity:")
    print(f"  Conventional : {CONV_PNOM:6.0f} MW  (${CONV_COST:.1f}/MWh)")
    print(f"  Wind Farm    : {WIND_PNOM:6.0f} MW  (${WIND_COST:.1f}/MWh)")
    print(f"  Solar PV     : {SOLAR_PNOM:6.0f} MWp (${SOLAR_COST:.1f}/MWh)")
    print(f"  Small Hydro  : {HYDRO_PNOM:6.0f} MW  (${HYDRO_COST:.1f}/MWh)")
    print(f"  Total RE cap : {SOLAR_PNOM + WIND_PNOM + HYDRO_PNOM:.0f} MW")
    lp = load_profile(HOURS)
    total_min = lp['bus2'].min() + lp['bus4'].min() + lp['bus5'].min()
    total_max = lp['bus2'].max() + lp['bus4'].max() + lp['bus5'].max()
    print(f"\nTotal load: {TOTAL_LOAD} MW avg  (varies {total_min:.1f} – {total_max:.1f} MW)")

    # Solve
    print("\n--- Running 24-hour Linear OPF ---")
    ok = run_opf(net)

    if ok:
        tbl = print_dispatch_table(net)

        print("\n--- Peak Line Flows ---")
        peak = n_lines_peak(net)
        print(peak.to_string())

        plot_results(net, tbl)
    else:
        print("OPF did NOT converge. Check data / solver.")