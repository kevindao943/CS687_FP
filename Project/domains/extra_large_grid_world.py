from domains.game import *

class Extra_Large_Grid_World:
    def __init__(self):
        # Parameters for domain
        self.rewards_dict = {'Empty': -0.05, 'Trap': -5, 'Guard': -20, 'Princess': 50}
        self.states_dict = {'Wall': [(0,2), (2,0), (2,4), (4,2), (2,5), (5,2), (2,6), (6,2), (2,7), (7,2), (1,6), (6,1), (3,7), (7,3), (4,7), (7,4), (5,7), (7,5), (6,8), (8,6), (7,8), (8,7)], 
                            'Trap': [(2,1), (1,2), (5,0), (0,5), (8,1), (1,8), (2,9), (9,2), (8,4), (4,8), (9,9), (3,5)], 
                            'Guard': [(4,0), (0,4), (7,0), (0,7), (9,0), (0,9), (3,3), (5,3), (3,5), (9,3), (3,9), (4,4), (5,4), (4,5), (6,5), (5,6), (9,5), (5,9), (7,6), (6,7), (9,8), (8,9)],
                            'Princess': [(6,6)]}
        self.terminal_states = ['Princess']
        self.wall_states = ['Wall']
        self.monsters = ['Guard', 'Trap']
        self.state_space = (10,10)
        # Initializing domain + MDP
        self.game = Game(self.state_space, self.states_dict, self.rewards_dict, self.terminal_states, self.wall_states, self.transition_func)
        self.action_space = ["AU", "AD", "AL", "AR", "AUL", "AUR", "ADL", "ADR"]
        self.actions_art = ["↑","↓","←","→","↖","↗","↙","↘"]
        self.discount = 1
        self.theta = 0.0001

    def transition_func(self, state, action, next_state_probs):
        row, col = state
        dynamics = [(1,0),(0,1),(0,-1),(1,1),(1,-1),(0,0)] # forward, right, left, slant right, slant left, stand
        slant_dynamics = [(1,1), (1,0), (0,1), (1,-2), (-2,1), (0,0)]
        probabilities = [0.6, 0.06, 0.06, 0.12, 0.12, 0.04]
        if action in ["AU", "AD", "AL", "AR"]:
            for item in zip(dynamics, probabilities):
                dynamic, prob = item[0], item[1]
                if action == "AU":
                    next_state = (row - dynamic[0], col + dynamic[1])
                elif action == "AD":
                    next_state = (row + dynamic[0], col + dynamic[1])
                elif action == "AL":
                    next_state = (row + dynamic[1], col - dynamic[0])
                elif action == "AR":
                    next_state = (row + dynamic[1], col + dynamic[0])
                next_state = self.game._check_and_update_state(*next_state, state)
                next_state_probs[next_state] = next_state_probs.get(next_state, 0) + prob
        else:
            for item in zip(slant_dynamics, probabilities):
                dynamic, prob = item[0], item[1]
                if action == "AUL":
                    next_state = (row - dynamic[0], col - dynamic[1])
                elif action == "AUR":
                    next_state = (row - dynamic[1], col + dynamic[0])
                elif action == "ADL":
                    next_state = (row + dynamic[1], col - dynamic[0])
                else:
                    next_state = (row + dynamic[0], col + dynamic[0])
                next_state = self.game._check_and_update_state(*next_state, state)
                next_state_probs[next_state] = next_state_probs.get(next_state, 0) + prob
        return next_state_probs

    # Update paramters for domain and reinitialize game
    def update_game(self):
        self.game = Game(self.state_space, self.states_dict, self.rewards_dict, self.terminal_states, self.wall_states)

    def _get_policy_art(self, row, col, max_action):
        state = self.game.get_state(row, col)
        if state in self.terminal_states:
            return 'P'
        art = self.actions_art[max_action] + str('(G)' if state == self.monsters[0] else ('(T)' if state == self.monsters[1] else ''))
        return art