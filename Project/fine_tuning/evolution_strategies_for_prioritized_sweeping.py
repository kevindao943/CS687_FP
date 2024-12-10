import random
import copy
import numpy as np

class EvolutionStrategyForPrioritizedSweeping:
    def __init__(self, prioritized_sweeping, mdp, optimal_v_map):
        self.prioritized_sweeping = prioritized_sweeping
        self.mdp = mdp
        self.optimal_v_map = optimal_v_map
        self.continuous_hyperparameters = ['theta', 'alpha', 'epsilon']
        self.discrete_hyperparameters = ['n', 'niter', 'episode_length']
        self.all_hyperparameters = self.continuous_hyperparameters + self.discrete_hyperparameters

    def evaluate_params(self, params, num_avg): # fitness function
        theta, alpha, epsilon, n, niter, episode_length = params
        prioritized_sweeping = self.prioritized_sweeping(self.mdp, self.optimal_v_map)
        avg_max_norm_val = 0
        for i in range(num_avg):
            _, max_norm_val, _, _ = prioritized_sweeping.prioritized_sweeping(theta=theta, alpha=alpha, epsilon=epsilon, n=int(n), niter=int(niter), episode_length=int(episode_length))
            avg_max_norm_val += max_norm_val / num_avg
        return avg_max_norm_val

    def initialize_population(self, pop_size, param_bounds):
        population = []
        for _ in range(pop_size):
            params = [random.uniform(*param_bounds[hyperparameter]) if hyperparameter in self.continuous_hyperparameters 
                         else random.randint(*param_bounds[hyperparameter])
                         for hyperparameter in self.all_hyperparameters]
            population.append(params)
        return population

    def mutate(self, params, hyperparameter_bounds, mutation_strength):
        c = copy.deepcopy(params)

        def mutate_continuous(val, bounds, strength):
            val += np.random.normal(0, strength * (bounds[1]-bounds[0]))
            return float(np.clip(val, bounds[0], bounds[1]))

        def mutate_discrete(val, bounds, strength):
            val += int(round(np.random.normal(0, strength * (bounds[1]-bounds[0]) / 10.0)))
            return int(np.clip(val, bounds[0], bounds[1]))

        for i in range(len(self.all_hyperparameters)):
            hyperparameter = self.all_hyperparameters[i]
            if hyperparameter in self.continuous_hyperparameters:
                c[i] = mutate_continuous(c[i], hyperparameter_bounds[hyperparameter], mutation_strength)
            else:
                mutate_discrete(c[i], hyperparameter_bounds[hyperparameter], mutation_strength)
        return c

    def run_es(self, hyperparameter_bounds, pop_size=20, generations=25, top_parents=5, mutation_strength=0.1, num_avg=3, keep_parents=2, decaying_mutation=True):
        population = self.initialize_population(pop_size, hyperparameter_bounds)
        best_params, min_loss, generation_min_loss_list = None, float('inf'), []
        for generation in range(generations):
            losses = []
            for params in population:
                loss = self.evaluate_params(params, num_avg)
                losses.append((loss, params))
            losses.sort(key=lambda x: x[0])
            generation_min_loss = losses[0][0]
            if generation_min_loss < min_loss:
                min_loss = losses[0][0]
                best_params = losses[0][1]
            parents = [x[1] for x in losses[:top_parents]]
            new_population = [x[1] for x in losses[:keep_parents]]
            current_mutation_strength = mutation_strength
            if decaying_mutation:
                current_mutation_strength = mutation_strength * (1 - generation / generations)
            for _ in range(pop_size):
                parent = random.choice(parents)
                new_params = self.mutate(parent, hyperparameter_bounds, current_mutation_strength)
                new_population.append(new_params)
            population = new_population
            generation_min_loss_list.append(generation_min_loss)
            print(f"Generation {generation+1}/{generations} - Generation Best Max Norm Value {generation_min_loss} - Best Max Norm Value: {min_loss}")
        return best_params, min_loss, generation_min_loss_list