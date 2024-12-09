from algorithms.prioritized_sweeping import *
from domains.extra_large_grid_world import *
from algorithms.value_iteration import *
from domains.cat_vs_monsters import *
from domains.gridworld import *
from fine_tuning.evolution_strategies_for_prioritized_sweeping import *
from matplotlib import pyplot as plt
from utils.utils import *
import pandas as pd

def run_domain(domain, domain_name, prioritized_sweeping_args, hyperparameter_bounds):
    print(f"-- {domain_name} Domain --\n")
    instance = ValueIteration(domain)
    cvm_v_map, cvm_policy_map, iterations = instance.run_standard_value_iteration(init_func)
    print("Actual Optimal Value Function of States\n")
    df = pd.DataFrame(cvm_v_map)
    print(df)
    print("\nActual Optimal Policy Map of States\n")
    df = pd.DataFrame(cvm_policy_map)
    print(df)
    print("\nRunning Prioritized Sweeping Algorithm...\n")

    instance = PrioritizedSweeping(domain, cvm_v_map)
    q_map, max_norm_val, iterations, elapsed_time = instance.prioritized_sweeping(*prioritized_sweeping_args)

    print(f"The prioritized sweeping algorithm terminated after {iterations} iterations")
    print(f"It achieved a max norm value of {max_norm_val}, taking {elapsed_time:.4f} seconds\n")

    ps_vmap, ps_policy_map = get_vmap_policy_map(q_map, instance)
    print("Value Function of States output by Prioritized Sweeping\n")
    df = pd.DataFrame(ps_vmap)
    print(df)
    print("\n")

    df = pd.DataFrame(ps_policy_map)
    print(df)
    print("\n")

    es_ps = EvolutionStrategyForPrioritizedSweeping(prioritized_sweeping=PrioritizedSweeping, mdp=domain, optimal_v_map=cvm_v_map)
    best_params, min_loss, generation_min_loss_list = es_ps.run_es(hyperparameter_bounds, pop_size=50, generations=20, top_parents=10, mutation_strength=0.2)

    print("Best hyperparameters found:", best_params)
    print("Best Max Norm Value:", min_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(generation_min_loss_list)
    plt.xlabel("Generation")
    plt.ylabel("Best Max Norm Value")
    plt.grid(True)
    plt.show()

theta, alpha, epsilon = 0.1, 0.1, 0.1
prioritized_sweeping_args = [theta, alpha, epsilon]
hyperparameter_bounds = {'theta': (0.01, 1.0),'alpha': (0.01, 1.0),'epsilon': (0.01, 1.0),'n': (1, 100),'niter': (50, 400),'episode_length': (50, 400)}
run_domain(Cat_vs_Monsters(), "Cat vs Monsters", prioritized_sweeping_args, hyperparameter_bounds)
run_domain(Gridworld(), "687-Gridworld", prioritized_sweeping_args, hyperparameter_bounds)
run_domain(Extra_Large_Grid_World(), "Extra Large Grid World", prioritized_sweeping_args, hyperparameter_bounds)