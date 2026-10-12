import warnings
import logging
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import io

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pypsa").setLevel(logging.WARNING)

def run_economic_dispatch():
    # Initialize network
    network = pypsa.Network()
    
    # 24 hour snapshots
    network.set_snapshots(range(24))

    # Add a single bus
    network.add("Bus", "Main Bus")

    # Add generators with different marginal costs in Indian Rupees (₹/MWh)
    # Cheap base load (Coal @ ₹3.00/kWh = ₹3000/MWh)
    network.add("Generator",
                "Base Load (Coal)",
                bus="Main Bus",
                p_nom=500,
                marginal_cost=3000)
                
    # More expensive but flexible (Gas @ ₹6.00/kWh = ₹6000/MWh)
    network.add("Generator",
                "Mid Merit (Gas)",
                bus="Main Bus",
                p_nom=300,
                marginal_cost=6000)
                
    # Peaking plant (Open-cycle / Diesel @ ₹10.00/kWh = ₹10000/MWh)
    network.add("Generator",
                "Peaking Plant",
                bus="Main Bus",
                p_nom=200,
                marginal_cost=10000)

    # Add a load that varies over the day
    load_profile = [400, 350, 320, 300, 320, 380, 450, 550, 650, 700, 750, 780, 
                    800, 790, 760, 730, 700, 720, 780, 850, 820, 700, 550, 450]
    
    network.add("Load",
                "Demand",
                bus="Main Bus",
                p_set=load_profile)
                
    network.sanitize()

    # Run optimization (Linear Optimal Power Flow)
    try:
        # PyPSA >= 0.20
        status, condition = network.optimize(solver_name='highs')
    except AttributeError:
        # Fallback for older PyPSA versions
        status, condition = network.lopf(solver_name='highs')

    print("\n" + "=" * 70)
    print("NUMERICAL METHOD & OPTIMIZATION SOLVER DETAILS")
    print("=" * 70)
    print("Formulation            : Linear Optimal Power Flow (LOPF / Economic Dispatch)")
    print("Optimization Problem   : Continuous Linear Program (LP)")
    print("PyPSA Optimization API : Linopy frontend -> HiGHS solver backend")
    print("Numerical Method       : Revised Dual Simplex Method")
    sm = getattr(network.model, "solver_model", None)
    if sm is not None:
        info = sm.getInfo()
        print(f"Simplex Iterations     : {info.simplex_iteration_count} iterations (solution obtained)")
        print(f"Presolve Status        : {sm.getModelPresolveStatus()}")
        print(f"Solver Model Status    : {sm.modelStatusToString(sm.getModelStatus())}")
    print(f"Optimization Status    : {status}, condition: {condition}")
    print(f"Objective (Total Cost) : \u20b9{network.objective:,.2f}")
    print("=" * 70 + "\n")
    
    # Show dispatch results
    dispatch_results = network.generators_t.p
    print("\nDispatch Results (MW):")
    print(dispatch_results.head())
    
    # Plotting the dispatch
    fig, ax = plt.subplots(figsize=(10, 6))
    dispatch_results.plot(kind="area", stacked=True, ax=ax, colormap="tab10", alpha=0.8)
    ax.plot(network.loads_t.p_set, color='black', label="Total Demand", linewidth=2)
    
    ax.set_title("Economic Dispatch (Indian Grid Benchmark)")
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Power Output (MW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("economic_dispatch_results.png")
    print("\nPlot saved to economic_dispatch_results.png")

    # Show market clearing prices (Marginal prices at the bus)
    prices = network.buses_t.marginal_price
    print("\nMarket Clearing Prices (\u20b9/MWh):")
    print(prices.head())

    return network

if __name__ == "__main__":
    run_economic_dispatch()
