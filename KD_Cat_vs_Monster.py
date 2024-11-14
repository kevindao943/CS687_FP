import numpy as np
import pandas as pd
from copy import deepcopy

class Cat_vs_Monster:
    def __init__(self, world_size, states_dict, reward_dict, terminal_states, wall_states):
        # Initialize domain parameters
        self.nrow, self.ncol = world_size
        self.states = [['Empty' for _ in range(self.ncol)] for _ in range(self.nrow)]
        self.terminal_states = terminal_states
        self.wall_states = wall_states
        # Update self.states corresponding to domain parameters
        self._init_world(states_dict)
        self.rewards_dict = reward_dict
        self.reward = 0

    def _edit_state(self, row: int, col: int, name: str):
        self.states[row][col] = name

    # Get state given row and col
    def get_state(self, row, col):
        return self.states[row][col]

    def _init_world(self, states_dict):
        for key in states_dict.keys():
            coords = states_dict[key]
            for coord in coords:
                row, col = coord[0], coord[1]
                self._edit_state(row, col, key)

    # Given a prospective state (row, col) to move into, is that state valid? If row, col is out of range or (row, col) is a wall state -> return the previous state, else return (row, col)
    def _check_and_update_state(self, row, col, state):
        valid_row_range, valid_col_range = range(self.nrow), range(self.ncol)
        if (row not in valid_row_range) or (col not in valid_col_range) or (self.get_state(row, col) in self.wall_states):
            return state
        return (row, col)
            
    # Get the list of next states and their probabilities, given a state and action
    def get_next_state_probs(self, state, action):
        next_state_probs = {}
        # Check if current state is a terminal state
        if self.get_state(*state) in self.terminal_states:
            next_state_probs[state] = 1
            return next_state_probs
        row, col = state
        dynamics = [(1,0),(0,1),(0,-1),(0,0)]
        probabilities = [0.7, 0.12, 0.12, 0.06]
        for item in zip(dynamics, probabilities):
            dynamic, prob = item[0], item[1]
            if action == "AU":
                next_state = (row - dynamic[0], col + dynamic[1])
            elif action == "AD":
                next_state = (row + dynamic[0], col + dynamic[1])
            elif action == "AL":
                next_state = (row + dynamic[1], col - dynamic[0])
            else:
                next_state = (row + dynamic[1], col + dynamic[0])
            next_state = self._check_and_update_state(*next_state, state)
            next_state_probs[next_state] = next_state_probs.get(next_state, 0) + prob
        return next_state_probs

    # Get the reward given a state, its action and the next state
    def get_reward(self, state, action, next_state):
        if self.get_state(*state) in self.terminal_states:
            return 0
        next_state = self.get_state(*next_state)
        return self.rewards_dict[next_state]
    
class ValueIteration:
    def __init__(self):
        # Parameters for domain
        self.rewards_dict = {'Empty': -0.05, 'Monster': -8, 'Food': 10}
        self.states_dict = {'Forbidden Furniture': [(2,1),(2,2),(2,3),(3,2)], 'Monster': [(0,3),(4,1)], 'Food': [(4,4)]}
        self.terminal_states = ['Food']
        self.wall_states = ['Forbidden Furniture']
        self.world_size = (5,5)
        # Initializing domain + MDP
        self.game = Cat_vs_Monster(self.world_size, self.states_dict, self.rewards_dict, self.terminal_states, self.wall_states)
        self.actions = ["AU", "AD", "AL", "AR"]
        self.actions_art = ["↑","↓","←","→"]
        self.discount = 0.925
        self.theta = 0.0001

    # Update paramters for domain and reinitialize game
    def update_game(self):
        self.game = Cat_vs_Monster(self.world_size, self.states_dict, self.rewards_dict, self.terminal_states, self.wall_states)
    
    # Get the value function, given state and action
    def _get_value(self, v_map, state, action):
        # Get possible next states and their probabilities
        next_state_probs = self.game.get_next_state_probs(state, action)
        value = 0
        # Iterating through all possible next states
        for key in next_state_probs.keys():
            # Get p - transition probability of next state, given state + action
            transition_prob = next_state_probs[key]
            next_state = list(key)
            # Get r - reward function given state, action, and next state
            reward = self.game.get_reward(state, action, next_state)
            # Get v(s + 1) - discounted value of next state
            discounted_value_next_state = self.discount * v_map[next_state[0]][next_state[1]]
            # Using the bellman equation to get the updated value function
            value += transition_prob * (reward + discounted_value_next_state)
        return value
    
    # Running value iteration
    def _run_value_iteration(self, init_func, type):
        # Get the domain for internal logic...
        nrow, ncol = self.game.nrow, self.game.ncol
        wall_states = [self.states_dict[wall_type] for wall_type in self.wall_states]
        wall_states = [elem for row in wall_states for elem in row]
        # Initialize value map and policy map of states
        v_map = [[init_func(row, col, wall_states) for col in range(ncol)] for row in range(nrow)]
        policy_map = [[None for col in range(ncol)] for row in range(nrow)]
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
                    if self.game.get_state(row, col) not in self.wall_states:
                        # If not, initialize current state and current value
                        state = (row, col)
                        v = v_map[row][col]
                        # Get the value of the state across the action space
                        action_space = [self._get_value(v_map, state, action) for action in self.actions]
                        # Finding the value maximizing action
                        max_action = np.argmax(action_space)
                        if type == "standard":
                            # Update the value and policy map
                            next_v_map[row][col] = float("{:.4f}".format(action_space[max_action]))
                            next_policy_map[row][col] = self.actions_art[max_action]
                            # Calculate delta
                            delta = float(max(delta, abs(v - next_v_map[row][col])))
                        elif type == "in-place":
                            v_map[row][col] = float("{:.4f}".format(action_space[max_action]))
                            policy_map[row][col] = self.actions_art[max_action]
                            delta = float(max(delta, abs(v - v_map[row][col])))
            iterations += 1
            if type == "standard":
                v_map, policy_map = deepcopy(next_v_map), deepcopy(next_policy_map)
            # Check delta for stopping
            if delta < self.theta:
                break
        return v_map, policy_map, iterations

    def run_standard_value_iteration(self, init_func): #init_func: function that initializes the state value. function signature: row: int, col: int, wall_states: list(int) -> value: int
        return self._run_value_iteration(init_func, "standard")

    def run_in_place_value_iteration(self, init_func):
        return self._run_value_iteration(init_func, "in-place")

# How to run:

# def init_func(row, col, wall_states):
#     if (row,col) in wall_states:
#         return None
#     return 0

# instance = ValueIteration()
# v_map, policy_map, iterations = instance.run_standard_value_iteration(init_func)