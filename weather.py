import pypsa
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Create network
# ==========================================

n = pypsa.Network()


# ==========================================
# 2. Create 24 hourly snapshots (0 to 23)
# ==========================================

hours = list(range(24))

n.set_snapshots(hours)


# ==========================================
# 3. Add one electrical bus
# ==========================================

n.add(
    "Bus",
    "Grid"
)


# ==========================================
# 4. Add electrical load
# ==========================================

load = [
    50, 48, 47, 46, 45, 48,
    55, 65, 75, 80, 85, 90,
    95, 92, 88, 85, 87, 92,
    100, 98, 90, 80, 70, 60
]

n.add(
    "Load",
    "Load",
    bus="Grid",
    p_set=pd.Series(load, index=hours)
)


# ==========================================
# 5. Solar availability
# ==========================================

solar_availability = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.05, 0.15, 0.30, 0.50, 0.70, 0.85,
    1.00, 0.95, 0.85, 0.70, 0.45, 0.20,
    0.02, 0.00, 0.00, 0.00, 0.00, 0.00
]

solar_availability = pd.Series(
    solar_availability,
    index=hours
)


# ==========================================
# 6. Add solar generator
# ==========================================

n.add(
    "Generator",
    "Solar",
    bus="Grid",
    p_nom=50,
    p_max_pu=solar_availability,
    marginal_cost=0,
    carrier="solar"
)


# ==========================================
# 7. Add coal generator
# ==========================================

n.add(
    "Generator",
    "Coal",
    bus="Grid",
    p_nom=50,
    marginal_cost=30,
    carrier="coal"
)


# ==========================================
# 8. Add gas generator
# ==========================================

n.add(
    "Generator",
    "Gas",
    bus="Grid",
    p_nom=100,
    marginal_cost=70,
    carrier="gas"
)


# ==========================================
# 9. Run optimization
# ==========================================

print("\nRunning PyPSA optimization...")

n.optimize(
    solver_name="highs"
)

print("\nOptimization completed successfully.")


# ==========================================
# 10. Get generator outputs
# ==========================================

generation = n.generators_t.p

print("\n==========================================")
print("Generator Output (MW)")
print("==========================================")

print(generation.round(2))


# ==========================================
# 11. Calculate net load
# ==========================================

solar_output = generation["Solar"]

actual_load = n.loads_t.p_set["Load"]

net_load = actual_load - solar_output


# ==========================================
# 12. Check power balance
# ==========================================

power_balance = generation.sum(axis=1) - actual_load

print("\n==========================================")
print("Power Balance Error (MW)")
print("==========================================")

print(power_balance.round(6))


# ==========================================
# 13. Calculate total generation
# ==========================================

total_generation = generation.sum(axis=1)

print("\n==========================================")
print("Maximum Load")
print("==========================================")

print(f"{actual_load.max():.2f} MW")


print("\n==========================================")
print("Maximum Net Load")
print("==========================================")

print(f"{net_load.max():.2f} MW")


print("\n==========================================")
print("Minimum Net Load")
print("==========================================")

print(f"{net_load.min():.2f} MW")


# ==========================================
# 14. Calculate total energy generated
# ==========================================

energy_generated = generation.sum()

print("\n==========================================")
print("Total Energy Generated (MWh)")
print("==========================================")

print(energy_generated.round(2))


# ==========================================
# 15. Calculate solar energy
# ==========================================

solar_energy = generation["Solar"].sum()

print("\n==========================================")
print("Total Solar Energy")
print("==========================================")

print(f"{solar_energy:.2f} MWh")


# ==========================================
# 16. Calculate solar share
# ==========================================

total_load_energy = actual_load.sum()

solar_share = (
    solar_energy / total_load_energy
) * 100

print("\n==========================================")
print("Solar Energy Share")
print("==========================================")

print(f"{solar_share:.2f} %")


# ==========================================
# 17. Plot generator dispatch
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    hours,
    actual_load,
    label="Load",
    linewidth=2
)

plt.plot(
    hours,
    generation["Solar"],
    label="Solar",
    linewidth=2
)

plt.plot(
    hours,
    generation["Coal"],
    label="Coal",
    linewidth=2
)

plt.plot(
    hours,
    generation["Gas"],
    label="Gas",
    linewidth=2
)

plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Power (MW)")
plt.title("Solar Integration and Generator Dispatch")

plt.legend()
plt.grid(True)

plt.xticks(hours)

plt.tight_layout()

plt.show()


# ==========================================
# 18. Plot duck curve
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    hours,
    actual_load,
    label="Load",
    linewidth=2
)

plt.plot(
    hours,
    net_load,
    label="Net Load",
    linewidth=2
)

plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Power (MW)")
plt.title("Duck Curve: Load vs Net Load")

plt.legend()
plt.grid(True)

plt.xticks(hours)

plt.tight_layout()

plt.show()


# ==========================================
# 19. Plot solar availability
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    hours,
    solar_availability,
    label="Solar Availability",
    linewidth=2
)

plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Per Unit Availability")
plt.title("Solar Resource Availability")

plt.legend()
plt.grid(True)

plt.xticks(hours)

plt.tight_layout()

plt.show()


# ==========================================
# 20. Print hourly summary
# ==========================================

summary = pd.DataFrame({
    "Load (MW)": actual_load,
    "Solar (MW)": generation["Solar"],
    "Coal (MW)": generation["Coal"],
    "Gas (MW)": generation["Gas"],
    "Net Load (MW)": net_load
})

print("\n==========================================")
print("Hourly Dispatch Summary")
print("==========================================")

print(summary.round(2))
