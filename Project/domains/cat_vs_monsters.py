from domains.game import *

class Cat_vs_Monsters:
    def __init__(self):
        # Parameters for domain
        self.rewards_dict = {'Empty': -0.05, 'Monster': -8, 'Food': 10}
        self.states_dict = {'Forbidden Furniture': [(2,1),(2,2),(2,3),(3,2)], 'Monster': [(0,3),(4,1)], 'Food': [(4,4)]}
        self.terminal_states = ['Food']
        self.wall_states = ['Forbidden Furniture']
        self.monsters = ['Monster']
        self.state_space = (5,5)
        # Initializing domain + MDP
        self.game = Game(self.state_space, self.states_dict, self.rewards_dict, self.terminal_states, self.wall_states, self.transition_func)
        self.action_space = ["AU", "AD", "AL", "AR"]
        self.actions_art = ["↑","↓","←","→"]
        self.discount = 0.925
        self.theta = 0.0001

    def transition_func(self, state, action, next_state_probs):
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
            next_state = self.game._check_and_update_state(*next_state, state)
            next_state_probs[next_state] = next_state_probs.get(next_state, 0) + prob
        return next_state_probs

    # Update paramters for domain and reinitialize game
    def update_game(self):
        self.game = Game(self.state_space, self.states_dict, self.rewards_dict, self.terminal_states, self.wall_states)

    def _get_policy_art(self, row, col, max_action):
        state = self.game.get_state(row, col)
        if state in self.terminal_states:
            return 'G'
        art = self.actions_art[max_action] + str('(M)' if state in self.monsters else '')
        return art