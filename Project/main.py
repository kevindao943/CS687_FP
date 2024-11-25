from algorithms.prioritized_sweeping import *
from algorithms.value_iteration import *
from domains.cat_vs_monsters import *
from domains.gridworld import *
from utils.utils import *
import pandas as pd

def run_domain(domain, domain_name, prioritized_sweeping_args):
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

    instance = PrioritizedSweeping(Cat_vs_Monsters(), cvm_v_map)
    q_map, max_norm_val, iterations, elapsed_time = instance.prioritized_sweeping(*prioritized_sweeping_args)

    print(f"The prioritized sweeping algorithm terminated after {iterations} iterations")
    print(f"It achieved a max norm value of {max_norm_val}, taking {elapsed_time:.4f} seconds\n")

    ps_vmap, _ = get_vmap_policy_map(q_map, instance)
    print("Value Function of States output by Prioritized Sweeping\n")
    df = pd.DataFrame(ps_vmap)
    print(df)
    print("\n")

if __name__ == "__main__":
    theta, alpha, epsilon, episode_length, n, niter = 0.005, 0.075, 0.025, 200, 5, 1500
    prioritized_sweeping_args = (theta, alpha, epsilon, episode_length, n, niter)
    run_domain(Cat_vs_Monsters(), "Cat vs Monsters", prioritized_sweeping_args)
    run_domain(Gridworld(), "687-Gridworld", prioritized_sweeping_args)