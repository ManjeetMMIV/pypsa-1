import streamlit as st
import numpy as np
import pandas as pd
import pypsa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Power System Solver", layout="wide", page_icon="⚡")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Power System Solver")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Problem",
    ["🔌 Problem 1 — 4-Bus Power Flow", "☀️ Problem 2 — Solar Economic Dispatch"]
)

# =============================================================================
# PROBLEM 1: 4-BUS POWER FLOW (Example 6.8)
# =============================================================================
if page == "🔌 Problem 1 — 4-Bus Power Flow":

    st.title("🔌 Problem 1 — 4-Bus Power Flow (Example 6.8)")

    # ── Problem Statement ────────────────────────────────────────────────────
    with st.expander("📖 Problem Statement", expanded=True):
        st.markdown("""
        ### Problem 6.8 — Power System Engineering

        For the **4-bus sample system** shown in Fig 6.7:
        - **Bus 1** is the **Slack Bus** with voltage fixed at **V = 1.04∠0° pu**
        - **Buses 2, 3, 4** are **PQ buses** (load buses) with specified real and reactive power injections
        - The system is solved from a **flat voltage start** (all non-slack buses initialised at 1.0∠0°)

        **Objective:** Find the bus voltages and angles using:
        1. The **1st iteration of the Gauss-Seidel (GS)** method
        2. The **fully converged Newton-Raphson (NR)** solution via PyPSA
        """)

        col_img, col_data = st.columns([1.2, 1])

        with col_img:
            st.subheader("System Diagram (Fig 6.7)")
            try:
                st.image("problem 1.jpeg", caption="4-Bus Sample System — Fig 6.7", use_container_width=True)
            except Exception:
                st.warning("Image `problem 1.jpeg` not found in the working directory.")

        with col_data:
            st.subheader("Table 6.5 — Bus Input Data")
            bus_input = pd.DataFrame({
                "Bus": ["1", "2", "3", "4"],
                "Type": ["Slack", "PQ", "PQ", "PQ"],
                "P_net (pu)": ["—", "+0.5", "−1.0", "+0.3"],
                "Q_net (pu)": ["—", "−0.2", "+0.5", "−0.1"],
                "V₀ (pu)": ["1.04∠0°", "—", "—", "—"],
            })
            st.table(bus_input)

        st.subheader("Y-BUS Matrix (from book)")
        st.latex(r"""
        Y_{BUS} = \begin{bmatrix}
        3-j9 & -2+j6 & -1+j3 & 0 \\
        -2+j6 & 3.666-j11 & -0.666+j2 & -1+j3 \\
        -1+j3 & -0.666+j2 & 3.666-j11 & -2+j6 \\
        0 & -1+j3 & -2+j6 & 3-j9
        \end{bmatrix}
        """)

        st.subheader("Line Impedances (derived from Y-BUS)")
        st.markdown("Off-diagonal element $Y_{ij} = -y_{ij}$, so $z_{ij} = 1/y_{ij}$:")
        line_imp = pd.DataFrame({
            "Line": ["1–2", "1–3", "2–3", "2–4", "3–4"],
            "Admittance y_ij": ["2 − j6", "1 − j3", "0.666 − j2", "1 − j3", "2 − j6"],
            "Impedance z_ij = R + jX": ["0.05 + j0.15", "0.10 + j0.30", "0.15 + j0.45", "0.10 + j0.30", "0.05 + j0.15"],
        })
        st.table(line_imp)

    # ── How it Solves ────────────────────────────────────────────────────────
    with st.expander("⚙️ How the Solution is Obtained", expanded=False):
        st.markdown("""
        ### Step 1 — Gauss-Seidel (1 Iteration, Flat Start)

        The **Gauss-Seidel (GS)** method is an iterative technique for solving the nonlinear power flow equations.
        Starting from a **flat start** (all voltages = 1.0∠0°), each bus voltage is updated in sequence using:

        """)
        st.latex(r"""
        V_k^{(new)} = \frac{1}{Y_{kk}} \left[ \frac{P_k - jQ_k}{(V_k^{(old)})^*} - \sum_{m \neq k} Y_{km} V_m \right]
        """)
        st.markdown("""
        This is applied to **Bus 2 → Bus 3 → Bus 4** in order (Bus 1 is the fixed slack reference).
        Only **1 iteration** is performed here, as the textbook asks for the voltages at the end of the first GS iteration.

        ### Step 2 — Newton-Raphson via PyPSA (`n.pf()`)

        PyPSA then performs a **fully converged Newton-Raphson** AC power flow.
        NR solves the mismatch equations iteratively:

        """)
        st.latex(r"""
        \begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \mathbf{J} \begin{bmatrix} \Delta \delta \\ \Delta |V| \end{bmatrix}
        """)
        st.markdown("""
        where **J** is the Jacobian matrix. The solver iterates until the mismatch is below the convergence tolerance (~1e-6 pu).
        The NR method typically converges in **3–5 iterations** for well-conditioned systems.
        """)

    # ── Solver ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("🚀 Run Solver")

    @st.cache_data
    def solve_problem1():
        net = pypsa.Network()
        net.set_snapshots(["now"])
        for name in ["Bus1", "Bus2", "Bus3", "Bus4"]:
            net.add("Bus", name, v_nom=1.0)
        lines = [
            ("L12", "Bus1", "Bus2", 0.05, 0.15),
            ("L13", "Bus1", "Bus3", 0.10, 0.30),
            ("L23", "Bus2", "Bus3", 0.150, 0.45),
            ("L24", "Bus2", "Bus4", 0.10, 0.30),
            ("L34", "Bus3", "Bus4", 0.05, 0.15),
        ]
        for name, bus0, bus1, r, x in lines:
            net.add("Line", name, bus0=bus0, bus1=bus1, r=r, x=x, s_nom=9999)
        net.add("Generator", "G1_slack", bus="Bus1", control="Slack", v_set_pu=1.04, p_nom=9999, p_set=0.0)
        for bus, (p_net, q_net) in {"Bus2": (0.5, -0.2), "Bus3": (-1.0, 0.5), "Bus4": (0.3, -0.1)}.items():
            net.add("Load", f"Load_{bus}", bus=bus, p_set=-p_net, q_set=-q_net)
        net.pf()

        # GS 1 iteration
        Y = np.array([
            [ 3-9j, -2+6j, -1+3j, 0+0j],
            [-2+6j, 3.666-11j, -0.666+2j, -1+3j],
            [-1+3j, -0.666+2j, 3.666-11j, -2+6j],
            [ 0+0j, -1+3j, -2+6j, 3-9j],
        ], dtype=complex)
        S_spec = np.array([0.5-0.2j, -1.0+0.5j, 0.3-0.1j])
        V = np.array([1.04+0j, 1.0+0j, 1.0+0j, 1.0+0j])
        for k in range(1, 4):
            sigma = sum(Y[k, m] * V[m] for m in range(4) if m != k)
            V[k] = (1.0 / Y[k, k]) * (np.conj(S_spec[k-1]) / np.conj(V[k]) - sigma)

        bus_list = ["Bus1", "Bus2", "Bus3", "Bus4"]
        bus_results = pd.DataFrame({
            "Bus": bus_list,
            "Type": ["Slack", "PQ", "PQ", "PQ"],
            "|V| NR (pu)": [round(net.buses_t.v_mag_pu.loc["now", b], 6) for b in bus_list],
            "Angle NR (deg)": [round(net.buses_t.v_ang.loc["now", b] * 180/np.pi, 6) for b in bus_list],
            "|V| GS-1 (pu)": [round(abs(V[i]), 6) for i in range(4)],
            "Angle GS-1 (deg)": [round(np.angle(V[i], deg=True), 6) for i in range(4)],
        })

        line_flows = pd.DataFrame({
            "Line": net.lines.index,
            "From": net.lines.bus0.values,
            "To": net.lines.bus1.values,
            "P_flow (pu)": [round(net.lines_t.p0.loc["now", ln], 6) for ln in net.lines.index],
            "Q_flow (pu)": [round(net.lines_t.q0.loc["now", ln], 6) for ln in net.lines.index],
        })

        p_slack = net.generators_t.p.loc["now", "G1_slack"]
        q_slack = net.generators_t.q.loc["now", "G1_slack"]
        return bus_results, line_flows, p_slack, q_slack

    if st.button("▶ Solve Power Flow", type="primary"):
        with st.spinner("Running PyPSA Newton-Raphson..."):
            br, lf, ps, qs = solve_problem1()
        st.success("✅ Power flow converged!")

        st.subheader("1. Bus Voltage Results — NR (converged) vs GS (1st iteration)")
        st.dataframe(br, use_container_width=True, hide_index=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("2. Line Power Flows (converged)")
            st.dataframe(lf, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("3. Slack Bus (Bus 1)")
            st.metric("P Generation", f"{ps:.5f} pu", delta=f"{ps*100:.2f} MW")
            st.metric("Q Generation", f"{qs:.5f} pu", delta=f"{qs*100:.2f} MVAR")
    else:
        st.info("👆 Click the button above to run the solver.")


# =============================================================================
# PROBLEM 2: SOLAR ECONOMIC DISPATCH
# =============================================================================
elif page == "☀️ Problem 2 — Solar Economic Dispatch":

    st.title("☀️ Problem 2 — Solar Integration & Economic Dispatch")

    # ── Problem Statement ────────────────────────────────────────────────────
    with st.expander("📖 Problem Statement", expanded=True):
        st.markdown("""
        ### Problem — Optimal Generator Dispatch with Solar (Weather-Dependent)

        A single-bus power system supplies a **time-varying electrical load** over a **24-hour period**.
        Three generators are available to meet the demand:

        | Generator | Capacity | Marginal Cost | Notes |
        |-----------|----------|---------------|-------|
        | **Solar** | 50 MW    | ₹0/MWh        | Output limited by sunlight (weather profile) |
        | **Coal**  | 50 MW    | ₹30/MWh       | Conventional baseload |
        | **Gas**   | 100 MW   | ₹70/MWh       | Expensive peaker, used last |

        **Objective:** Schedule the generators over 24 hours to **minimise total generation cost**
        while ensuring supply always meets demand at every hour.

        **Key Effects to Observe:**
        - The **Duck Curve** — how solar reduces the net load during the day but causes a steep evening ramp
        - **Merit-order dispatch** — cheapest generator always dispatched first
        - **Solar share** — what percentage of total daily energy is supplied by solar
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("24-Hour Load Profile (MW)")
            load_df = pd.DataFrame({
                "Hour": list(range(24)),
                "Load (MW)": [50,48,47,46,45,48,55,65,75,80,85,90,95,92,88,85,87,92,100,98,90,80,70,60]
            })
            st.dataframe(load_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Solar Availability Profile (pu)")
            solar_df = pd.DataFrame({
                "Hour": list(range(24)),
                "Availability (pu)": [0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.15,0.30,0.50,0.70,0.85,
                                      1.00,0.95,0.85,0.70,0.45,0.20,0.02,0.00,0.00,0.00,0.00,0.00]
            })
            st.dataframe(solar_df, use_container_width=True, hide_index=True)

    # ── How it Solves ────────────────────────────────────────────────────────
    with st.expander("⚙️ How the Optimization Works", expanded=False):
        st.markdown("""
        ### Linear Optimal Power Flow (LOPF) via PyPSA `n.optimize()`

        This is **not** a traditional power flow (`n.pf()`). It is an **Economic Dispatch** —
        a Linear Program (LP) that PyPSA formulates and hands to the **HiGHS solver**.

        **Objective Function — Minimise total cost:**
        """)
        st.latex(r"""
        \min \sum_{t=0}^{23} \left[ c_{Solar} \cdot p_{Solar,t} + c_{Coal} \cdot p_{Coal,t} + c_{Gas} \cdot p_{Gas,t} \right]
        """)
        st.markdown("where $c_{Solar}=0$, $c_{Coal}=30$, $c_{Gas}=70$ (₹/MWh).")

        st.markdown("**Constraints at every hour *t*:**")
        st.latex(r"""
        p_{Solar,t} + p_{Coal,t} + p_{Gas,t} = \text{Load}_t \quad \text{(supply = demand)}
        """)
        st.latex(r"""
        0 \leq p_{Solar,t} \leq 50 \times \text{availability}_t, \quad
        0 \leq p_{Coal,t} \leq 50, \quad
        0 \leq p_{Gas,t} \leq 100
        """)
        st.markdown("""
        **Merit Order Result:** Because Solar costs ₹0, it is always **dispatched first** up to its weather limit.
        Coal fills the remaining gap (up to 50 MW), and Gas is used only when Solar + Coal < Load.

        **The Duck Curve** appears because solar pushes the *net load* (Load − Solar) very low during midday.
        When the sun sets, conventional generators must ramp up sharply — this steep evening ramp is the
        "neck" of the duck, and is a real grid stability challenge.
        """)

    # ── Solver ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("🚀 Run Optimization")

    @st.cache_data
    def solve_problem2():
        hours = list(range(24))
        load = [50,48,47,46,45,48,55,65,75,80,85,90,95,92,88,85,87,92,100,98,90,80,70,60]
        solar_avail = [0.00,0.00,0.00,0.00,0.00,0.00,0.05,0.15,0.30,0.50,0.70,0.85,
                       1.00,0.95,0.85,0.70,0.45,0.20,0.02,0.00,0.00,0.00,0.00,0.00]

        n = pypsa.Network()
        n.set_snapshots(hours)
        n.add("Bus", "Grid")
        n.add("Load", "Load", bus="Grid", p_set=pd.Series(load, index=hours))
        n.add("Generator", "Solar", bus="Grid", p_nom=50, p_max_pu=pd.Series(solar_avail, index=hours), marginal_cost=0, carrier="solar")
        n.add("Generator", "Coal", bus="Grid", p_nom=50, marginal_cost=30, carrier="coal")
        n.add("Generator", "Gas", bus="Grid", p_nom=100, marginal_cost=70, carrier="gas")
        n.optimize(solver_name="highs")

        gen = n.generators_t.p
        actual_load = pd.Series(load, index=hours)
        net_load = actual_load - gen["Solar"]
        solar_avail_s = pd.Series(solar_avail, index=hours)

        summary = pd.DataFrame({
            "Hour": hours,
            "Load (MW)": actual_load.values,
            "Solar (MW)": gen["Solar"].round(2).values,
            "Coal (MW)": gen["Coal"].round(2).values,
            "Gas (MW)": gen["Gas"].round(2).values,
            "Net Load (MW)": net_load.round(2).values,
        })

        solar_energy = gen["Solar"].sum()
        total_energy = actual_load.sum()
        solar_share = solar_energy / total_energy * 100
        stats = {
            "Max Load (MW)": actual_load.max(),
            "Max Net Load (MW)": net_load.max(),
            "Min Net Load (MW)": net_load.min(),
            "Solar Energy (MWh)": round(solar_energy, 2),
            "Coal Energy (MWh)": round(gen["Coal"].sum(), 2),
            "Gas Energy (MWh)": round(gen["Gas"].sum(), 2),
            "Solar Share (%)": round(solar_share, 2),
        }

        # Build figures
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(hours, actual_load.values, label="Load", linewidth=2)
        ax1.plot(hours, gen["Solar"].values, label="Solar", linewidth=2)
        ax1.plot(hours, gen["Coal"].values, label="Coal", linewidth=2)
        ax1.plot(hours, gen["Gas"].values, label="Gas", linewidth=2)
        ax1.set_xlabel("Hour of Day (0–23)")
        ax1.set_ylabel("Power (MW)")
        ax1.set_title("Generator Dispatch Schedule")
        ax1.legend(); ax1.grid(True); ax1.set_xticks(hours)
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(hours, actual_load.values, label="Gross Load", linewidth=2)
        ax2.plot(hours, net_load.values, label="Net Load (Load − Solar)", linewidth=2, linestyle="--")
        ax2.fill_between(hours, net_load.values, actual_load.values, alpha=0.2, label="Solar Contribution")
        ax2.set_xlabel("Hour of Day (0–23)")
        ax2.set_ylabel("Power (MW)")
        ax2.set_title("Duck Curve: Gross Load vs Net Load")
        ax2.legend(); ax2.grid(True); ax2.set_xticks(hours)
        fig2.tight_layout()

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(hours, solar_avail_s.values, label="Solar Availability", linewidth=2, color="orange")
        ax3.set_xlabel("Hour of Day (0–23)")
        ax3.set_ylabel("Per Unit Availability")
        ax3.set_title("Solar Resource Availability (Weather Profile)")
        ax3.legend(); ax3.grid(True); ax3.set_xticks(hours)
        fig3.tight_layout()

        return summary, stats, fig1, fig2, fig3

    if st.button("▶ Run Economic Dispatch", type="primary"):
        with st.spinner("Running HiGHS optimizer via PyPSA..."):
            summary, stats, fig1, fig2, fig3 = solve_problem2()
        st.success("✅ Optimization converged!")

        # Key metrics
        st.subheader("📊 Key Statistics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Load", f"{stats['Max Load (MW)']} MW")
        m2.metric("Solar Energy", f"{stats['Solar Energy (MWh)']} MWh")
        m3.metric("Solar Share", f"{stats['Solar Share (%)']} %")
        m4.metric("Gas Energy", f"{stats['Gas Energy (MWh)']} MWh")

        st.subheader("1. Hourly Dispatch Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("2. Generator Dispatch Schedule")
        st.pyplot(fig1)

        st.subheader("3. Duck Curve")
        st.pyplot(fig2)

        st.subheader("4. Solar Weather Profile")
        st.pyplot(fig3)

    else:
        st.info("👆 Click the button above to run the economic dispatch optimizer.")
