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

    def evaluate_params(self, params): # fitness function
        theta, alpha, epsilon, n, niter, episode_length = params

        prioritized_sweeping = self.prioritized_sweeping(self.mdp, self.optimal_v_map)
        _, max_norm_val, _, _ = prioritized_sweeping.prioritized_sweeping(
            theta=theta,
            alpha=alpha,
            epsilon=epsilon,
            n=int(n),  # Ensure integer
            niter=int(niter),
            episode_length=int(episode_length)
        )
        return max_norm_val


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

    def run_es(self, hyperparameter_bounds, pop_size=100, generations=50, top_parents=3, mutation_strength=0.1):
        population = self.initialize_population(pop_size, hyperparameter_bounds)
        best_params = None
        min_loss = float('inf')

        generation_min_loss_list = []

        for generation in range(generations):
            losses = []
            for params in population:
                loss = self.evaluate_params(params)
                losses.append((loss, params))
            losses.sort(key=lambda x: x[0])

            generation_min_loss = losses[0][0]

            if generation_min_loss < min_loss:
                min_loss = losses[0][0]
                best_params = losses[0][1]
            parents = [x[1] for x in losses[:top_parents]]
            new_population = []
            for _ in range(pop_size):
                parent = random.choice(parents)
                offspring = self.mutate(parent, hyperparameter_bounds, mutation_strength)
                new_population.append(offspring)
            population = new_population

            generation_min_loss_list.append(generation_min_loss)

            print(f"Generation {generation+1}/{generations} - Best Max Norm Value: {min_loss} - Generation Best Max Norm Value {generation_min_loss}")

        return best_params, min_loss, generation_min_loss_list