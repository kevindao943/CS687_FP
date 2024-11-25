import numpy as np
from copy import deepcopy

class ValueIteration:
    # Get the value function, given state and action
    def __init__(self, mdp):
        self.mdp = mdp
    
    def _get_value(self, v_map, state, action):
        # Get possible next states and their probabilities
        next_state_probs = self.mdp.game.get_next_state_probs(state, action)
        value = 0
        # Iterating through all possible next states
        for key in next_state_probs.keys():
            # Get p - transition probability of next state, given state + action
            transition_prob = next_state_probs[key]
            next_state = list(key)
            # Get r - reward function given state, action, and next state
            reward = self.mdp.game.get_reward(state, action, next_state)
            # Get v(s + 1) - discounted value of next state
            discounted_value_next_state = self.mdp.discount * v_map[next_state[0]][next_state[1]]
            # Using the bellman equation to get the updated value function
            value += transition_prob * (reward + discounted_value_next_state)
        return value
    
    # Running value iteration
    def _run_value_iteration(self, init_func, type):
        # Get the domain for internal logic...
        nrow, ncol = self.mdp.game.nrow, self.mdp.game.ncol
        wall_states = [self.mdp.states_dict[wall_type] for wall_type in self.mdp.wall_states]
        wall_states = [elem for row in wall_states for elem in row]
        # Initialize value map and policy map of states
        v_map = [[init_func(row, col, wall_states, 'values') for col in range(ncol)] for row in range(nrow)]
        policy_map = [[init_func(row, col, wall_states, 'policies') for col in range(ncol)] for row in range(nrow)]
        # Counting number of iterations...
        iterations = 0
        while True:
            # Initialize delta
            delta = 0
            if type == "standard":
                next_v_map, next_policy_map = deepcopy(v_map), deepcopy(policy_map)
            # Iterating through all rows of the domain
            for row in range(nrow):
                # Iterating through all columns of the domain
                for col in range(ncol):
                    # Checking if a state is a wall state
                    if self.mdp.game.get_state(row, col) not in self.mdp.wall_states:
                        # If not, initialize current state and current value
                        state = (row, col)
                        v = v_map[row][col]
                        # Get the value of the state across the action space
                        valid_actions = [self._get_value(v_map, state, action) for action in self.mdp.action_space]
                        # Finding the value maximizing action
                        max_action = np.argmax(valid_actions)
                        if type == "standard":
                            # Update the value and policy map
                            next_v_map[row][col] = float("{:.4f}".format(valid_actions[max_action]))
                            next_policy_map[row][col] = self.mdp._get_policy_art(row, col, max_action)
                            # Calculate delta
                            delta = float(max(delta, abs(v - next_v_map[row][col])))
                        elif type == "in-place":
                            v_map[row][col] = float("{:.4f}".format(valid_actions[max_action]))
                            policy_map[row][col] = self.mdp._get_policy_art(row, col, max_action)
                            delta = float(max(delta, abs(v - v_map[row][col])))
            iterations += 1
            if type == "standard":
                v_map, policy_map = deepcopy(next_v_map), deepcopy(next_policy_map)
            # Check delta for stopping
            if delta < self.mdp.theta:
                break
        return v_map, policy_map, iterations

    def run_standard_value_iteration(self, init_func): #init_func: function that initializes the state value. function signature: row: int, col: int, wall_states: list(int) -> value: int
        return self._run_value_iteration(init_func, "standard")

    def run_in_place_value_iteration(self, init_func):
        return self._run_value_iteration(init_func, "in-place")