import time
import random
import heapq

class PrioritizedSweeping:
    def __init__(self, mdp, optimal_v_map):
        self.mdp = mdp
        self.optimal_v_map = optimal_v_map
        self.gamma = self.mdp.discount  # Discount factor
        self.thresh = 0.5
        self.actions = self.mdp.action_space
        self.wall_states = self.mdp.wall_states
        self.terminal_states = self.mdp.terminal_states
        self.game = self.mdp.game

    def _arbitrary_init(self):
        nrow, ncol = self.mdp.state_space
        actions = self.actions
        qmap = [[{action: 0.0 for action in actions} for _ in range(ncol)] for _ in range(nrow)]
        pi = [[{action: 1.0 / len(self.actions) for action in actions} for _ in range(ncol)] for _ in range(nrow)]
        model = [[{action: {'reward': None, 'next_state': None} for action in actions} for _ in range(ncol)] for _ in range(nrow)]
        return qmap, pi, model

    def max_norm(self, q_map):
        nrow, ncol = self.mdp.state_space
        max_diff = float('-inf')
        optimal_v_map = self.optimal_v_map
        get_state = self.game.get_state
        wall_states = self.wall_states
        for row in range(nrow):
            for col in range(ncol):
                if get_state(row, col) not in wall_states:
                    max_q_val = max(q_map[row][col].values())
                    actual_val = optimal_v_map[row][col]
                    diff = abs(actual_val - max_q_val)
                    if diff > max_diff:
                        max_diff = diff
        return max_diff

    def compute_priority(self, state, action, qmap, model):
        row, col = state
        data = model[row][col][action]
        reward, next_state = data['reward'], data['next_state']
        if reward is None or next_state is None:
            return 0
        next_row, next_col = next_state
        max_next_q = max(qmap[next_row][next_col].values())
        current_q = qmap[row][col][action]
        priority = abs(reward + self.gamma * max_next_q - current_q)
        return priority

    def update_policy(self, q, pi, visited_state_action, epsilon):
        visited_states_set = set()
        actions = self.actions
        for (state, _) in visited_state_action:
            if state not in visited_states_set:
                visited_states_set.add(state)
                row, col = state
                q_vals = q[row][col]
                max_q = max(q_vals.values())
                opt_actions = [a for a, val in q_vals.items() if val == max_q]
                for possible_action in actions:
                    if possible_action in opt_actions:
                        pi[row][col][possible_action] = (1 - epsilon) / len(opt_actions) + epsilon / len(self.actions)
                    else:
                        pi[row][col][possible_action] = epsilon / len(self.actions)
        return pi

    def prioritized_sweeping(self, theta, alpha, epsilon, n=5, niter=1000, episode_length=150):
        # Retrieving all functions for fast access
        nrow, ncol = self.mdp.state_space
        qmap, pi, model = self._arbitrary_init()
        get_state = self.game.get_state
        get_action = self.game.get_action
        get_next_state = self.game.get_next_state
        get_reward = self.game.get_reward
        terminal_states = self.terminal_states
        wall_states = self.wall_states
        actions = self.actions
        compute_priority = self.compute_priority
        cur_iter = 0
        max_norm_val = float('inf')
        start_time = time.time()
        while max_norm_val >= self.thresh and cur_iter < niter:
            while True:
                srow = random.randint(0, nrow - 1)
                scol = random.randint(0, ncol - 1)
                if get_state(srow, scol) not in wall_states:
                    break
            state = (srow, scol)
            visited_state_action = []
            pqueue = []
            for _ in range(episode_length):
                row, col = state
                action = get_action(state, pi)
                next_state = get_next_state(state, action)
                visited_state_action.append((state, action))
                reward = get_reward(state, action, next_state)
                model[row][col][action]['reward'] = reward
                model[row][col][action]['next_state'] = next_state
                priority = compute_priority(state, action, qmap, model)
                if priority > theta:
                    heapq.heappush(pqueue, (-priority, (row, col, action)))
                for _ in range(n):
                    if not pqueue:
                        break
                    _, (cur_row, cur_col, cur_action) = heapq.heappop(pqueue)
                    data = model[cur_row][cur_col][cur_action]
                    temp_reward = data['reward']
                    temp_next_state = data['next_state']
                    if temp_reward is None or temp_next_state is None:
                        continue
                    next_row, next_col = temp_next_state
                    max_next_q = max(qmap[next_row][next_col].values())
                    current_q = qmap[cur_row][cur_col][cur_action]
                    qmap[cur_row][cur_col][cur_action] += alpha * (temp_reward + self.gamma * max_next_q - current_q)
                    for action in actions:
                        pred_state = get_next_state((cur_row, cur_col), action)
                        pred_row, pred_col = pred_state
                        pred_priority = compute_priority((pred_row, pred_col), action, qmap, model)
                        if pred_priority > theta:
                            heapq.heappush(pqueue, (-pred_priority, (pred_row, pred_col, action)))
                state = next_state
                if get_state(*state) in terminal_states:
                    break
            pi = self.update_policy(qmap, pi, visited_state_action, epsilon)
            max_norm_val = self.max_norm(qmap)
            cur_iter += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        return qmap, max_norm_val, cur_iter, elapsed_time
