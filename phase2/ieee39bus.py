# ================================================================
# 39-BUS / 10-GENERATOR POWER SYSTEM
#
# Table B.3 Load-Flow Case
#
# Objective:
# Reproduce the textbook load-flow results of Table B.3.
#
# Assumption:
# Transmission and transformer active-power losses are neglected.
#
# Therefore:
#       R_line = 0
#       R_transformer = 0
#
# X and line charging Bc are retained.
# ================================================================

# ------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------

import logging
import warnings
import numpy as np
import pandas as pd
import pypsa


# ------------------------------------------------
# 2. DISPLAY / WARNING SETTINGS
# ------------------------------------------------

logging.getLogger("pypsa").setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

pd.set_option(
    "display.float_format",
    "{:.5f}".format
)

pd.set_option(
    "display.width",
    180
)

pd.set_option(
    "display.max_rows",
    100
)

# ------------------------------------------------
# 3. SYSTEM BASE
# ------------------------------------------------

S_BASE = 100.0       # MVA

# Transmission voltage level
V_HV = 345.0         # kV

# Base impedance
#
# Z_base = V_base^2 / S_base
#
# Since V is in kV and S is in MVA,
# the result is in ohms.

Z_BASE = V_HV**2 / S_BASE


print("=" * 70)
print("39-BUS / 10-GENERATOR POWER SYSTEM")
print("=" * 70)

print(f"System base       : {S_BASE} MVA")
print(f"HV voltage base   : {V_HV} kV")
print(f"Z base            : {Z_BASE:.4f} ohm")
print()


# ------------------------------------------------
# 4. BUS NOMINAL VOLTAGES
# ------------------------------------------------
#
# Generator buses:
#
# 1, 3, 4, 5, 6, 7, 8, 9, 10
#
# are generator-side 22-kV buses.
#
# Bus 2 is directly connected to the
# 345-kV transmission network.
#
# Buses 11-39 are 345-kV network buses.
# ------------------------------------------------

BUS_KV = {
    i: 345.0
    for i in range(1, 40)
}

for i in [1, 3, 4, 5, 6, 7, 8, 9, 10]:

    BUS_KV[i] = 22.0


# ------------------------------------------------
# 5. VOLTAGE SETPOINTS FROM TABLE B.3
# ------------------------------------------------
#
# These are the generator-bus voltage
# magnitudes.
#
# Bus 1 = Slack
# Buses 2-10 = PV
# ------------------------------------------------

V_SET = {

    1: 0.98200,
    2: 1.03000,
    3: 0.98310,
    4: 1.01230,
    5: 0.99720,
    6: 1.04930,
    7: 1.06350,
    8: 1.02780,
    9: 1.02650,
    10: 1.04750,

}


# ------------------------------------------------
# 6. GENERATOR ACTIVE POWER
# ------------------------------------------------
#
# Table B.3
#
# Bus 1:
#     Slack generator
#     P is NOT specified.
#
# Buses 2-10:
#     PV generators
#     P is specified.
#
# Values below are in per-unit.
# PyPSA uses MW.
#
# Therefore:
#
#       P_MW = P_pu * 100
# ------------------------------------------------

GEN_P_PU = {

    2: 10.00000,
    3: 6.50000,
    4: 5.08000,
    5: 6.32000,
    6: 6.50000,
    7: 5.60000,
    8: 5.40000,
    9: 8.30000,
    10: 2.50000,

}


# ------------------------------------------------
# 7. LOAD DATA FROM TABLE B.3
# ------------------------------------------------
#
# IMPORTANT:
#
# Bus 30 DOES HAVE a load:
#
#       P_L = 6.280 pu
#       Q_L = 1.030 pu
#
# The corrected bus numbering is used below.
#
# All values are in per-unit.
# ------------------------------------------------

LOADS_PU = {

    # Bus : (P_L, Q_L)

    1:  (0.09200, 0.04600),
    2:  (11.04000, 2.50000),

    13: (3.22000, 0.02400),
    14: (5.00000, 1.84000),

    17: (2.33800, 0.84000),
    18: (5.22000, 1.76000),

    21: (2.74000, 1.15000),

    23: (2.74500, 0.84660),
    24: (3.08600, 0.92200),
    25: (2.24000, 0.47200),
    26: (1.39000, 0.17000),
    27: (2.81000, 0.75500),
    28: (2.06000, 0.27600),
    29: (2.83500, 0.26900),

    # IMPORTANT: Bus 30 load
    30: (6.28000, 1.03000),

    32: (0.07500, 0.88000),

    35: (3.20000, 1.53000),
    36: (3.29400, 0.32300),
    38: (1.58000, 0.30000),

}


# ------------------------------------------------
# 8. CHECK TOTAL LOAD
# ------------------------------------------------

total_load_P_pu = sum(
    p for p, q in LOADS_PU.values()
)

total_load_Q_pu = sum(
    q for p, q in LOADS_PU.values()
)

print("LOAD CHECK")
print("-" * 70)

print(
    f"Total active load  = "
    f"{total_load_P_pu:.5f} pu "
    f"= {total_load_P_pu * S_BASE:.2f} MW"
)

print(
    f"Total reactive load = "
    f"{total_load_Q_pu:.5f} pu "
    f"= {total_load_Q_pu * S_BASE:.2f} MVAr"
)

print()


# ------------------------------------------------
# 9. TRANSMISSION LINE DATA
# ------------------------------------------------
#
# Table B.2
#
# Format:
#
# (name, from, to, R_pu, X_pu, Bc_pu)
#
# For the Table B.3 loss-neglected case:
#
#       R = 0
#
# X and Bc are retained.
# ------------------------------------------------

LINES = [

    ("L01", 37, 27, 0.0013, 0.0173, 0.3216),
    ("L02", 37, 38, 0.0007, 0.0082, 0.1319),
    ("L03", 36, 24, 0.0003, 0.0059, 0.0680),
    ("L04", 36, 21, 0.0008, 0.0135, 0.2548),
    ("L05", 36, 39, 0.0016, 0.0195, 0.3040),
    ("L06", 36, 37, 0.0007, 0.0089, 0.1342),
    ("L07", 35, 36, 0.0009, 0.0094, 0.1710),
    ("L08", 34, 35, 0.0018, 0.0217, 0.3660),
    ("L09", 33, 34, 0.0009, 0.0101, 0.1723),

    ("L10", 28, 29, 0.0014, 0.0151, 0.2490),
    ("L11", 26, 29, 0.0057, 0.0625, 1.0290),
    ("L12", 26, 28, 0.0043, 0.0474, 0.7802),
    ("L13", 26, 27, 0.0014, 0.0147, 0.2396),
    ("L14", 25, 26, 0.0032, 0.0323, 0.5130),

    ("L15", 23, 24, 0.0022, 0.0350, 0.3610),
    ("L16", 22, 23, 0.0006, 0.0096, 0.1846),
    ("L17", 21, 22, 0.0008, 0.0135, 0.2548),

    ("L18", 20, 33, 0.0004, 0.0043, 0.0729),
    ("L19", 20, 31, 0.0004, 0.0043, 0.0729),

    ("L20", 19, 2, 0.0010, 0.0250, 1.2000),
    ("L21", 18, 19, 0.0023, 0.0363, 0.3804),
    ("L22", 17, 18, 0.0004, 0.0046, 0.0780),

    ("L23", 16, 31, 0.0007, 0.0082, 0.1389),
    ("L24", 16, 17, 0.0006, 0.0092, 0.1130),

    ("L25", 15, 18, 0.0008, 0.0112, 0.1476),
    ("L26", 15, 16, 0.0002, 0.0026, 0.0434),

    ("L27", 14, 34, 0.0008, 0.0129, 0.1382),
    ("L28", 14, 15, 0.0008, 0.0128, 0.1342),

    ("L29", 13, 38, 0.0011, 0.0133, 0.2138),
    ("L30", 13, 14, 0.0013, 0.0213, 0.2214),

    ("L31", 12, 25, 0.0070, 0.0086, 0.1460),
    ("L32", 12, 13, 0.0013, 0.0151, 0.2572),

    ("L33", 11, 12, 0.0035, 0.0411, 0.6987),
    ("L34", 11, 2, 0.0010, 0.0250, 0.7500),

]


# ------------------------------------------------
# 10. TRANSFORMER DATA
# ------------------------------------------------
#
# Table B.4
#
# Format:
#
# (name, from, to, R_pu, X_pu, tap)
#
# For the lossless B.3 case:
#
#       R = 0
#
# Transformer X is already in per-unit.
# ------------------------------------------------

TRANSFORMERS = [

    ("T01", 39, 30, 0.0007, 0.0138, 1.0),
    ("T02", 39, 5,  0.0007, 0.0142, 1.0),

    ("T03", 32, 33, 0.0016, 0.0435, 1.0),
    ("T04", 32, 31, 0.0016, 0.0435, 1.0),

    ("T05", 30, 4,  0.0009, 0.0180, 1.0),
    ("T06", 29, 9,  0.0008, 0.0156, 1.0),

    ("T07", 25, 8,  0.0006, 0.0232, 1.0),
    ("T08", 23, 7, 0.0005, 0.0272, 1.0),

    ("T09", 22, 6, 0.0000, 0.0143, 1.0),
    ("T10", 20, 3, 0.0000, 0.0200, 1.0),

    ("T11", 16, 1, 0.0000, 0.0250, 1.0),
    ("T12", 12, 10, 0.0000, 0.0181, 1.0),

]


# ------------------------------------------------
# 11. TABLE B.3 REFERENCE BUS VOLTAGES
# ------------------------------------------------
#
# These are used only to compare our PyPSA result
# against the textbook.
# ------------------------------------------------

REF_V = [

    0.98200,
    1.03000,
    0.98310,
    1.01230,
    0.99720,
    1.04930,
    1.06350,
    1.02780,
    1.02650,
    1.04750,

    1.03829,
    1.02310,
    0.99576,
    0.95894,
    0.95660,
    0.95688,
    0.95140,
    0.95276,
    1.01028,
    0.95988,

    0.99046,
    1.01550,
    1.01344,
    0.98179,
    1.02088,
    1.01822,
    1.00150,
    1.02204,
    1.02143,
    0.98832,

    0.95760,
    0.93795,
    0.95912,
    0.96168,
    0.96683,
    0.98196,
    0.99086,
    0.99197,
    0.98770,

]


# ------------------------------------------------
# 12. TABLE B.3 REFERENCE ANGLES
# ------------------------------------------------

REF_ANGLE = [

     0.00000,
    -9.55016,
     3.20174,
     4.61664,
     5.57217,
     6.62654,
     9.46958,
     3.16537,
     9.04654,
    -2.47597,

    -7.79710,
    -4.89487,
    -8.07759,
    -9.35310,
    -8.29471,
    -7.56925,
    -9.97400,
    -10.50170,
    -9.92054,
    -4.71314,

    -2.98024,
     1.62430,
     1.34841,
    -5.45955,
    -3.68918,
    -4.76321,
    -6.92554,
    -0.95906,
     1.95588,
    -0.62515,

    -5.69316,
    -5.68713,
    -5.47342,
    -7.20767,
    -7.32475,
    -5.55956,
    -6.73437,
    -7.71437,
     0.34648,

]


# ------------------------------------------------
# 13. REFERENCE GENERATOR RESULTS
# ------------------------------------------------

REF_GP_PU = {

    1: 5.04509,
    2: 10.00000,
    3: 6.50000,
    4: 5.08000,
    5: 6.32000,
    6: 6.50000,
    7: 5.60000,
    8: 5.40000,
    9: 8.30000,
    10: 2.50000,

}


REF_GQ_PU = {

    1: 1.36036,
    2: 1.95746,
    3: 1.59104,
    4: 1.58151,
    5: 0.95582,
    6: 2.76414,
    7: 2.35485,
    8: 0.63019,
    9: 0.84790,
    10: 1.46483,

}


# ================================================================
# 14. BUILD PYPSA NETWORK
# ================================================================

def build_network():

    n = pypsa.Network()


    # ------------------------------------------------
    # 14.1 ADD BUSES
    # ------------------------------------------------

    for i in range(1, 40):

        n.add(
            "Bus",
            f"Bus {i}",
            v_nom=BUS_KV[i],
            v_mag_pu_set=V_SET.get(i, 1.0)
        )


    # ------------------------------------------------
    # 14.2 ADD LOADS
    # ------------------------------------------------

    for bus, (p_pu, q_pu) in LOADS_PU.items():

        n.add(
            "Load",
            f"Load {bus}",
            bus=f"Bus {bus}",
            p_set=p_pu * S_BASE,
            q_set=q_pu * S_BASE
        )


    # ------------------------------------------------
    # 14.3 ADD SLACK GENERATOR
    # ------------------------------------------------
    #
    # Bus 1:
    #
    # V = 0.982 pu
    # angle = reference
    #
    # P and Q are solved.
    # ------------------------------------------------

    n.add(
        "Generator",
        "G1",
        bus="Bus 1",
        control="Slack",
        p_set=0.0
    )


    # ------------------------------------------------
    # 14.4 ADD PV GENERATORS
    # ------------------------------------------------
    #
    # Buses 2-10:
    #
    # P specified
    # V specified
    #
    # Q and angle are solved.
    # ------------------------------------------------

    for bus, p_pu in GEN_P_PU.items():

        n.add(
            "Generator",
            f"G{bus}",
            bus=f"Bus {bus}",
            control="PV",
            p_set=p_pu * S_BASE
        )


    # ------------------------------------------------
    # 14.5 ADD TRANSMISSION LINES
    # ------------------------------------------------

    for name, b0, b1, r_pu, x_pu, bc_pu in LINES:

        # LOSSLESS CASE:
        #
        # R = 0
        #
        # Convert X from pu to ohms:
        #
        # X_ohm = X_pu * Z_base
        #
        # Convert Bc from pu to Siemens:
        #
        # B_S = B_pu / Z_base

        x_ohm = x_pu * Z_BASE

        b_siemens = bc_pu / Z_BASE


        n.add(
            "Line",
            name,
            bus0=f"Bus {b0}",
            bus1=f"Bus {b1}",

            # Resistance neglected
            r=0.0,

            # Reactance retained
            x=x_ohm,

            # Line charging retained
            b=b_siemens
        )


    # ------------------------------------------------
    # 14.6 ADD TRANSFORMERS
    # ------------------------------------------------

    for name, b0, b1, r_pu, x_pu, tap in TRANSFORMERS:

        n.add(
            "Transformer",
            name,

            bus0=f"Bus {b0}",
            bus1=f"Bus {b1}",

            # Resistance neglected
            r=0.0,

            # Reactance retained
            x=x_pu,

            # 100 MVA system base
            s_nom=S_BASE,

            # All taps are 1.0
            tap_ratio=tap
        )


    return n


# ================================================================
# 15. CREATE NETWORK
# ================================================================

n = build_network()


print("=" * 70)
print("NETWORK CREATED")
print("=" * 70)

print(f"Buses         : {len(n.buses)}")
print(f"Generators    : {len(n.generators)}")
print(f"Loads         : {len(n.loads)}")
print(f"Lines         : {len(n.lines)}")
print(f"Transformers  : {len(n.transformers)}")
print()


# ================================================================
# 16. RUN AC POWER FLOW
# ================================================================

print("=" * 70)
print("RUNNING AC NEWTON-RAPHSON POWER FLOW")
print("=" * 70)

pf = n.pf()


# ================================================================
# 17. CHECK CONVERGENCE
# ================================================================

converged = bool(
    pf["converged"].to_numpy().all()
)

iterations = int(
    pf["n_iter"].to_numpy().max()
)


print(f"Converged      : {converged}")
print(f"NR iterations  : {iterations}")
print()


if not converged:

    raise RuntimeError(
        "Power flow did not converge."
    )


# ================================================================
# 18. EXTRACT RESULTS
# ================================================================

snap = n.snapshots[0]


# Bus voltage magnitude
V = n.buses_t.v_mag_pu.loc[snap]


# Bus voltage angle
#
# PyPSA stores angle in radians.
# Convert to degrees.

angle_deg = np.degrees(
    n.buses_t.v_ang.loc[snap]
)


# Generator active power
gen_P_pu = (
    n.generators_t.p.loc[snap]
    / S_BASE
)


# Generator reactive power
gen_Q_pu = (
    n.generators_t.q.loc[snap]
    / S_BASE
)


# ================================================================
# 19. BUS VOLTAGE RESULTS
# ================================================================

bus_names = [
    f"Bus {i}"
    for i in range(1, 40)
]


bus_result = pd.DataFrame(

    {
        "V_pu":
            V.reindex(bus_names).values,

        "Angle_deg":
            angle_deg.reindex(bus_names).values,
    },

    index=range(1, 40)
)


bus_result.index.name = "Bus"


print("=" * 70)
print("BUS VOLTAGE RESULTS")
print("=" * 70)

print(bus_result)
print()


# ================================================================
# 20. COMPARE WITH TABLE B.3
# ================================================================

comparison = pd.DataFrame(

    {

        "V_PyPSA":
            V.reindex(bus_names).values,

        "V_Table_B3":
            REF_V,

        "Delta_V":
            V.reindex(bus_names).values
            - np.array(REF_V),


        "Angle_PyPSA":
            angle_deg.reindex(bus_names).values,

        "Angle_Table_B3":
            REF_ANGLE,

        "Delta_Angle":
            angle_deg.reindex(bus_names).values
            - np.array(REF_ANGLE),

    },

    index=range(1, 40)
)


comparison.index.name = "Bus"


print("=" * 70)
print("COMPARISON WITH TABLE B.3")
print("=" * 70)

print(comparison)
print()


# ================================================================
# 21. MAXIMUM VOLTAGE / ANGLE ERROR
# ================================================================

max_voltage_error = np.max(
    np.abs(
        comparison["Delta_V"]
    )
)


max_angle_error = np.max(
    np.abs(
        comparison["Delta_Angle"]
    )
)


print("=" * 70)
print("MAXIMUM DIFFERENCE FROM TABLE B.3")
print("=" * 70)

print(
    f"Maximum |Delta V|     = "
    f"{max_voltage_error:.6f} pu"
)

print(
    f"Maximum |Delta angle| = "
    f"{max_angle_error:.6f} degree"
)

print()


# ================================================================
# 22. GENERATOR RESULTS
# ================================================================

generator_buses = range(1, 11)


generator_result = pd.DataFrame(

    {

        "P_PyPSA_pu": [
            gen_P_pu[f"G{i}"]
            for i in generator_buses
        ],

        "P_Table_B3_pu": [
            REF_GP_PU[i]
            for i in generator_buses
        ],

        "Q_PyPSA_pu": [
            gen_Q_pu[f"G{i}"]
            for i in generator_buses
        ],

        "Q_Table_B3_pu": [
            REF_GQ_PU[i]
            for i in generator_buses
        ],

    },

    index=generator_buses
)


generator_result["Delta_P"] = (
    generator_result["P_PyPSA_pu"]
    - generator_result["P_Table_B3_pu"]
)


generator_result["Delta_Q"] = (
    generator_result["Q_PyPSA_pu"]
    - generator_result["Q_Table_B3_pu"]
)


generator_result.index.name = "Generator Bus"


print("=" * 70)
print("GENERATOR RESULTS")
print("=" * 70)

print(generator_result)
print()


# ================================================================
# 23. TRANSMISSION LINE FLOWS
# ================================================================

line_flows = pd.DataFrame(

    {

        "P0_MW":
            n.lines_t.p0.loc[snap],

        "Q0_MVAr":
            n.lines_t.q0.loc[snap],

        "P1_MW":
            n.lines_t.p1.loc[snap],

        "Q1_MVAr":
            n.lines_t.q1.loc[snap],

    }
)


line_flows["S0_MVA"] = np.hypot(
    line_flows["P0_MW"],
    line_flows["Q0_MVAr"]
)


line_flows["S1_MVA"] = np.hypot(
    line_flows["P1_MW"],
    line_flows["Q1_MVAr"]
)


# Because R = 0, active power loss
# should be approximately zero.

line_flows["P_loss_MW"] = (
    line_flows["P0_MW"]
    + line_flows["P1_MW"]
)


print("=" * 70)
print("TRANSMISSION LINE FLOWS")
print("=" * 70)

print(line_flows)
print()


# ================================================================
# 24. TRANSFORMER FLOWS
# ================================================================

transformer_flows = pd.DataFrame(

    {

        "P0_MW":
            n.transformers_t.p0.loc[snap],

        "Q0_MVAr":
            n.transformers_t.q0.loc[snap],

        "P1_MW":
            n.transformers_t.p1.loc[snap],

        "Q1_MVAr":
            n.transformers_t.q1.loc[snap],

    }
)


transformer_flows["S0_MVA"] = np.hypot(
    transformer_flows["P0_MW"],
    transformer_flows["Q0_MVAr"]
)


transformer_flows["S1_MVA"] = np.hypot(
    transformer_flows["P1_MW"],
    transformer_flows["Q1_MVAr"]
)


transformer_flows["P_loss_MW"] = (
    transformer_flows["P0_MW"]
    + transformer_flows["P1_MW"]
)


print("=" * 70)
print("TRANSFORMER FLOWS")
print("=" * 70)

print(transformer_flows)
print()


# ================================================================
# 25. POWER BALANCE
# ================================================================

total_generation_MW = (
    n.generators_t.p.loc[snap].sum()
)


total_load_MW = (
    n.loads_t.p.loc[snap].sum()
)


line_losses_MW = (
    line_flows["P_loss_MW"].sum()
)


transformer_losses_MW = (
    transformer_flows["P_loss_MW"].sum()
)


total_losses_MW = (
    line_losses_MW
    + transformer_losses_MW
)


balance_error_MW = (
    total_generation_MW
    - total_load_MW
    - total_losses_MW
)


print("=" * 70)
print("ACTIVE POWER BALANCE")
print("=" * 70)

print(
    f"Generation       = "
    f"{total_generation_MW:.6f} MW"
)

print(
    f"Load             = "
    f"{total_load_MW:.6f} MW"
)

print(
    f"Line losses      = "
    f"{line_losses_MW:.6f} MW"
)

print(
    f"Transformer loss = "
    f"{transformer_losses_MW:.6f} MW"
)

print(
    f"Total losses     = "
    f"{total_losses_MW:.6f} MW"
)

print(
    f"Balance error    = "
    f"{balance_error_MW:.6e} MW"
)

print()


# ================================================================
# 26. SLACK GENERATOR RESULT
# ================================================================

slack_P_pu = gen_P_pu["G1"]
slack_Q_pu = gen_Q_pu["G1"]


print("=" * 70)
print("SLACK GENERATOR")
print("=" * 70)

print(
    f"Bus 1 P_G = "
    f"{slack_P_pu:.6f} pu "
    f"= {slack_P_pu * S_BASE:.3f} MW"
)

print(
    f"Bus 1 Q_G = "
    f"{slack_Q_pu:.6f} pu "
    f"= {slack_Q_pu * S_BASE:.3f} MVAr"
)

print()


# ================================================================
# 27. FINAL SUMMARY
# ================================================================

print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Power flow converged : {converged}"
)

print(
    f"NR iterations        : {iterations}"
)

print(
    f"Total load           : "
    f"{total_load_P_pu:.5f} pu"
)

print(
    f"Slack P_G            : "
    f"{slack_P_pu:.5f} pu"
)

print(
    f"Slack Q_G            : "
    f"{slack_Q_pu:.5f} pu"
)

print(
    f"Maximum voltage error: "
    f"{max_voltage_error:.6f} pu"
)

print(
    f"Maximum angle error  : "
    f"{max_angle_error:.6f} degree"
)

print()

print("Lossless model:")
print("  Line R             = 0")
print("  Transformer R      = 0")
print("  Line X             = Table B.2")
print("  Line Bc            = Table B.2")
print("  Transformer X      = Table B.4")
print("  Generator data     = Table B.3")
print("  Load data          = Table B.3")

print("=" * 70)