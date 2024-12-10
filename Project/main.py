from algorithms.prioritized_sweeping import *
from algorithms.MCTS_new import *
from domains.extra_large_grid_world import *
from algorithms.value_iteration import *
from domains.cat_vs_monsters import *
from domains.gridworld import *
from fine_tuning.evolution_strategies_for_prioritized_sweeping import *
from matplotlib import pyplot as plt
from utils.utils import *
import pandas as pd

def run_domain(domain, domain_name, prioritized_sweeping_args, hyperparameter_bounds, es_pop_size, es_num_generations, es_top_parents, es_mutation_strength, es_num_avg, es_keep_parents, is_decaying_mutation):
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

    print(f"\nRunning Evolution Strategy for Prioritized Sweeping Algorithm, on {domain_name} domain...\n")

    es_ps = EvolutionStrategyForPrioritizedSweeping(prioritized_sweeping=PrioritizedSweeping, mdp=domain, optimal_v_map=cvm_v_map)
    best_params, min_loss, generation_min_loss_list = es_ps.run_es(hyperparameter_bounds, pop_size=es_pop_size, generations=es_num_generations, top_parents=es_top_parents, mutation_strength=es_mutation_strength, num_avg=es_num_avg, keep_parents=es_keep_parents, decaying_mutation=is_decaying_mutation)

    print("Best hyperparameters found:", best_params)
    print("Best Max Norm Value:", min_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(generation_min_loss_list)
    plt.xlabel("Generation")
    plt.ylabel("Best Max Norm Value")
    plt.grid(True)
    plt.show()

    print("\nRunning Prioritized Sweeping Algorithm on Fine-tuned Hyperparameters...\n")

    instance = PrioritizedSweeping(domain, cvm_v_map)
    q_map, max_norm_val, iterations, elapsed_time = instance.prioritized_sweeping(*best_params)

    print(f"The prioritized sweeping algorithm terminated after {iterations} iterations")
    print(f"It achieved a max norm value of {max_norm_val}, taking {elapsed_time:.4f} seconds\n")

    ps_vmap, ps_policy_map = get_vmap_policy_map(q_map, instance)
    print("Value Function of States output by Prioritized Sweeping\n")
    df = pd.DataFrame(ps_vmap)
    print(df)
    print("\n")

    df = pd.DataFrame(ps_policy_map)
    print(df)

    print(f"\nRunning Evolution Strategy for Prioritized Sweeping Algorithm, on {domain_name} domain...\n")

theta, alpha, epsilon = 0.1, 0.1, 0.1
prioritized_sweeping_args = [theta, alpha, epsilon]
hyperparameter_bounds = {'theta': (0.01, 0.75),'alpha': (0.01, 0.75),'epsilon': (0.01, 0.75),'n': (1, 100),'niter': (50, 500),'episode_length': (50, 500)}
es_params = [20, 20, 5, 0.05, 10, 2, True]
run_domain(Cat_vs_Monsters(), "Cat vs Monsters", prioritized_sweeping_args, hyperparameter_bounds, *es_params)
run_domain(Gridworld(), "687-Gridworld", prioritized_sweeping_args, hyperparameter_bounds, *es_params)
run_domain(Extra_Large_Grid_World(), "Extra Large Grid World", prioritized_sweeping_args, hyperparameter_bounds, 10, 10, 3, 0.05, 1, 1, False)

"""
MCTS
"""
# # Parameters for domain
def graph_MCTS(game, c_p_opt, c_p_min, c_p_max, policy_opt, policy_list, rollouts_opt, rollout_min, rollout_max):
    if game == "GridWorld":
        game = Gridworld()
        node = GridWorldNode
    elif game == "Cats vs Monsters":
        game = Cat_vs_Monsters()
        node = Cats_vs_Monsters_Node
    elif game == "Extra Large":
        game = Extra_Large_Grid_World()
        node = Really_Big_Node
    # C_p vs Total Reward
    c_p_list = []
    score_list = []
    for i in range(50):
        mcts = MCTS(game, node, np.random.uniform(c_p_min, c_p_max), rollouts_opt, policy_opt)
        iterations, C_p, score = mcts.mcts()
        print(f"MCTS with C_p = {C_p}")
        c_p_list.append(C_p)
        score_list.append(score)
    fig1, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title("Reward vs C_p")
    ax[0].set_ylabel('Reward')
    ax[0].set_xlabel("Exploration Factor C_p")
    # ax[0].set_ylim(9.5,10.5)
    ax[0].scatter(c_p_list, score_list)
    m, b = np.polyfit(c_p_list, score_list, 1) 
    ax[0].plot(c_p_list, m*np.array(c_p_list) + b, color='red')
    x_1, x_2, b = np.polyfit(c_p_list, score_list, 2) 
    x = np.linspace(0.5, 5, 1000)
    best_fit_square = x_1 * x**2 + x_2 * x + b
    ax[0].plot(x, best_fit_square, color='blue')
    print(score_list)
    # Policy Comparison
    score_list_list = []
    for policy in policy_list:
        score_list = []
        for i in range(15):
            mcts = MCTS(game, node, c_p_opt, rollouts_opt, policy)
            iterations, count, score = mcts.mcts()
            score_list.append(score)
        score_list_list.append(score_list)
    ax[1].set_title("Default Policy vs. Reward")
    ax[1].set_ylabel('Reward')
    bplot = ax[1].boxplot(score_list_list, labels=policy_list)  
    # # Learning Curve
    iterations_list = []
    score_list2 = []
    for i in range(50):
        mcts = MCTS(game, node, c_p_opt, np.random.randint(rollout_min,rollout_max), policy_opt)
        iterations, c_p, score = mcts.mcts()
        iterations_list.append(iterations)
        score_list2.append(score)
    print(score_list2)
    ax[2].scatter(iterations_list, score_list2)
    m, b = np.polyfit(iterations_list, score_list2, 1) 
    ax[2].plot(iterations_list, m*np.array(iterations_list) + b, color='red')
    x_1, x_2, b = np.polyfit(iterations_list, score_list2, 2) 
    x = np.linspace(25, 105, 1000)
    best_fit_square = x_1 * x**2 + x_2 * x + b
    ax[2].plot(x, best_fit_square, color='blue')
    ax[2].set_title("Reward vs. Number of Rollouts")
    ax[2].set_ylabel('Reward')
    ax[2].set_xlabel("# Rollouts")
    
    plt.tight_layout()
    plt.show()

"""
Recreate Graphs for MCTS
"""
#graph_MCTS("GridWorld", math.sqrt(2), 1, 5, "state_score", ["Random", "Down Left", "state_score"], 50, 0, 100)
#graph_MCTS("Cats vs Monsters", 3, 1, 5, "state_score", ["Down Left", "state_score"], 50, 0, 100)
#graph_MCTS("Extra Large", math.sqrt(2), 1, 5, "state_score", ["state_score", "state_score2"], 75, 25, 105)
"""
Run each once
"""
game = Gridworld()
node = GridWorldNode
mcts = MCTS(game, node, math.sqrt(2), 50, "state_score")
mcts.mcts()
game = Cat_vs_Monsters()
node = Cats_vs_Monsters_Node
mcts = MCTS(game, node, 3, 50, "state_score")
mcts.mcts()
game = Extra_Large_Grid_World()
node = Really_Big_Node
mcts = MCTS(game, node, math.sqrt(2), 75, "state_score")
mcts.mcts()
