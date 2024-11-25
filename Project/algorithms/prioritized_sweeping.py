import time
import random

class PrioritizedSweeping:
    def __init__(self, mdp, optimal_v_map):
        self.mdp = mdp
        self.optimal_v_map = optimal_v_map
        self.gamma = self.mdp.discount  # Discount factor
        self.thresh = 0.5
    
    def _arbitrary_init(self):
        n_row, n_col = self.mdp.state_space
        q = [[{action: 0 for action in self.mdp.action_space} for col in range(n_col)] for row in range(n_row)]
        pi = [[{action: 1/len(self.mdp.action_space) for action in self.mdp.action_space} for col in range(n_col)] for row in range(n_row)]
        model = [[{action: {'R': None, 'next_state': None} for action in self.mdp.action_space} for col in range(n_col)] for row in range(n_row)]
        return q, pi, model

    def max_norm(self, q_map):
        nrow, ncol = self.mdp.state_space
        max_norm_val = float('-inf')
        for row in range(nrow):
            for col in range(ncol):
                if self.mdp.game.get_state(row, col) not in self.mdp.wall_states:
                    diff = abs(self.optimal_v_map[row][col] - max(q_map[row][col].values()))
                    max_norm_val = max(diff, max_norm_val)
        return max_norm_val
    
    def compute_priority(self, state, action, q, model):
        row, col = state
        reward = model[row][col][action]['R']
        next_state = model[row][col][action]['next_state']
        if reward is None or next_state is None:
            return 0
        next_row, next_col = next_state
        max_next_q = max(q[next_row][next_col].values())
        current_q = q[row][col][action]
        priority = abs(reward + self.gamma * max_next_q - current_q)
        return priority

    def update_policy(self, q, pi, visited_state_action, epsilon):
        visited_state = []
        for state_action in visited_state_action:
            state, _ = state_action
            if state not in visited_state:
                visited_state.append(state)
                row, col = state
                max_q = max(q[row][col].values())
                opt_actions = [key for key, value in q[row][col].items() if value == max_q]
                for possible_action in self.mdp.action_space:
                    if possible_action in opt_actions:
                        pi[row][col][possible_action] = (1 - epsilon) / len(opt_actions) + epsilon / len(self.mdp.action_space)
                    else:
                        pi[row][col][possible_action] = epsilon / len(self.mdp.action_space)
        return pi

    def prioritized_sweeping(self, theta, alpha, epsilon, episode_length, n, niter = float('inf')):
        n_row, n_col = self.mdp.state_space
        q, pi, model = self._arbitrary_init()

        cur_iter, max_norm_val = 0, float('inf')

        start_time = time.time()
        while max_norm_val >= self.thresh and cur_iter < niter:

            srow, scol = random.randint(0,4), random.randint(0,4)
            
            if self.mdp.game.get_state(srow, scol) not in self.mdp.wall_states:
                
                pqueue = []
                state = (srow, scol)
                visited_state_action = []
        
                for _ in range(episode_length):
                    row, col = state
        
                    action = self.mdp.game.get_action(state, pi)
                    
                    next_state = self.mdp.game.get_next_state(state, action)
        
                    visited_state_action.append((state, action))
                    
                    reward = self.mdp.game.get_reward(state, action, next_state)
        
                    model[row][col][action] = {'R': reward, 'next_state': next_state}
        
                    priority = self.compute_priority(state, action, q, model)
                    if priority > theta:
                        pqueue.append(((row, col), action, priority))
                        pqueue.sort(key=lambda x: -x[2])  # sort by priority
        
                    for _ in range(n):
                        if not pqueue:
                            break
                        (cur_row, cur_col), cur_action, _ = pqueue.pop(0)
                        reward = model[cur_row][cur_col][cur_action]['R']
                        next_state = model[cur_row][cur_col][cur_action]['next_state']
                        if reward is None or next_state is None:
                            continue
                        next_row, next_col = next_state
                        max_next_q = max(q[next_row][next_col].values())
                        current_q = q[cur_row][cur_col][cur_action]
                        # q-learning
                        q[cur_row][cur_col][cur_action] += alpha * (
                            reward + self.gamma * max_next_q - current_q
                        )
        
                        for action in self.mdp.action_space:
                            pred_state = self.mdp.game.get_next_state((cur_row, cur_col), action)
                            pred_row, pred_col = pred_state
                            priority = self.compute_priority((pred_row, pred_col), action, q, model)
                            if priority > theta:
                                pqueue.append(((pred_row, pred_col), action, priority))
                                pqueue.sort(key=lambda x: -x[2])
        
                    state = next_state
                    if self.mdp.game.get_state(*state) in self.mdp.terminal_states:
                        break

                pi = self.update_policy(q, pi, visited_state_action, epsilon)
    
                max_norm_val = self.max_norm(q)

                # debug
                # if cur_iter % 500 == 0:
                #     print(f"{cur_iter=}, {max_norm_val=}")

                cur_iter += 1

        end_time = time.time()

        elapsed_time = end_time - start_time
            
        return q, max_norm_val, cur_iter, elapsed_time