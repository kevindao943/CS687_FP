import random

class Game:
    def __init__(self, state_space, states_dict, reward_dict, terminal_states, wall_states, transition_func):
        # Initialize domain parameters
        self.nrow, self.ncol = state_space
        self.states = [['Empty' for _ in range(self.ncol)] for _ in range(self.nrow)]
        self.terminal_states = terminal_states
        self.wall_states = wall_states
        # Update self.states corresponding to domain parameters
        self._init_world(states_dict)
        self.rewards_dict = reward_dict
        self.reward = 0
        self.transition_func = transition_func

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
        return self.transition_func(state, action, next_state_probs)

    # Get the reward given a state, its action and the next state
    def get_reward(self, state, action, next_state):
        if self.get_state(*state) in self.terminal_states:
            return 0
        next_state = self.get_state(*next_state)
        return self.rewards_dict[next_state]

    def get_action(self, state, policy):
        row, col = state
        action_probs = policy[row][col]
        actions, probabilities = list(action_probs.keys()), list(action_probs.values())
        action = random.choices(actions, weights=probabilities, k=1)[0]
        return action

    def get_next_state(self, state, action):
        next_state_probs = self.get_next_state_probs(state, action)
        next_states, probabilities = list(next_state_probs.keys()), list(next_state_probs.values())
        next_state = random.choices(next_states, weights=probabilities, k=1)[0]
        return tuple(next_state)

    def run_episode(self, starting_state, policy, discount, starting_action = None):
        cur_state, state_action_visited = starting_state, []
        returns, t = [], 0
        while self.get_state(*cur_state) not in self.terminal_states and t < 200:
            if starting_action is not None:
                cur_action = starting_action
                starting_action = None
            else:
                cur_action = self.get_action(cur_state, policy)
            next_state = self.get_next_state(cur_state, cur_action)
            reward = self.get_reward(cur_state, cur_action, next_state)
            returns.append(reward)
            state_action_visited.append((cur_state, cur_action))
            cur_state = next_state
            t += 1
        return returns, state_action_visited