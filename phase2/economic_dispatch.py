import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_economic_dispatch():
    # Initialize network
    network = pypsa.Network()
    
    # 24 hour snapshots
    network.set_snapshots(range(24))

    # Add a single bus
    network.add("Bus", "Main Bus")

    # Add generators with different marginal costs
    # Cheap base load
    network.add("Generator",
                "Base Load (Coal)",
                bus="Main Bus",
                p_nom=500,
                marginal_cost=30)
                
    # More expensive but flexible
    network.add("Generator",
                "Mid Merit (Gas)",
                bus="Main Bus",
                p_nom=300,
                marginal_cost=60)
                
    # Peaking plant
    network.add("Generator",
                "Peaking Plant",
                bus="Main Bus",
                p_nom=200,
                marginal_cost=100)

    # Add a load that varies over the day
    load_profile = [400, 350, 320, 300, 320, 380, 450, 550, 650, 700, 750, 780, 
                    800, 790, 760, 730, 700, 720, 780, 850, 820, 700, 550, 450]
    
    network.add("Load",
                "Demand",
                bus="Main Bus",
                p_set=load_profile)
                
    # Run optimization (Linear Optimal Power Flow)
    try:
        # PyPSA >= 0.20
        status, condition = network.optimize(solver_name='highs')
    except AttributeError:
        # Fallback for older PyPSA versions
        status, condition = network.lopf(solver_name='highs')

    print(f"Optimization Status: {status}, {condition}")
    print(f"Objective value (Total Cost): {network.objective:.2f} \u20ac")
    
    # Show dispatch results
    dispatch_results = network.generators_t.p
    print("\nDispatch Results (MW):")
    print(dispatch_results.head())
    
    # Plotting the dispatch
    fig, ax = plt.subplots(figsize=(10, 6))
    dispatch_results.plot(kind="area", stacked=True, ax=ax, colormap="tab10", alpha=0.8)
    ax.plot(network.loads_t.p_set, color='black', label="Total Demand", linewidth=2)
    
    ax.set_title("Economic Dispatch")
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Power Output (MW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("economic_dispatch_results.png")
    print("\nPlot saved to economic_dispatch_results.png")

    # Show market clearing prices (Marginal prices at the bus)
    prices = network.buses_t.marginal_price
    print("\nMarket Clearing Prices (\u20ac/MWh):")
    print(prices.head())

    return network

if __name__ == "__main__":
    run_economic_dispatch()
