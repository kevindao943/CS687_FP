import math
import time

import numpy as np

from KD_Cat_vs_Monster import Cat_vs_Monster


class MCTS():
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
        self.Nodes = {}
        self.C_p = 2#math.sqrt(2)
        for i in range(5):
            for j in range(5):
                state = (i,j)
                state_node = Node(state)
                self.Nodes[(i,j)] = state_node

    def get_action_UCT(self, node):
        good_action_list = []
        curr_best = -math.inf
        for a in ["AU", "AD", "AL", "AR"]:
            UCB1 = node.q[a] + 2 * self.C_p * math.sqrt((2 * math.log(node.get_num_visits())) / node.get_num_action_visits(a))
            if UCB1 > curr_best:
                curr_best = UCB1
                good_action_list = [a]
            elif UCB1 == curr_best:
                good_action_list.append(a)
        return np.random.choice(good_action_list)

    def get_outcome(self, s, a):
        state_probs = self.game.get_next_state_probs(s, a)
        return list(state_probs.keys())[np.random.choice(len(state_probs), p = list(state_probs.values()))]

    def Select(self, node):
        history = []
        # Keeps getting stuck in an infinite loop :(
        count = 0
        while node.fully_expanded():
            a = self.get_action_UCT(node)
            s_prime = self.get_outcome(node.state, a)
            history.append((node.state, a, s_prime))
            node = self.Nodes[s_prime]
            count += 1
            print(count)
        return node, history


    def Expand(self, node):
        if self.game.get_state(node.state[0], node.state[1]) == "Food":
            return node, []
        else:
            unexpanded_actions = node.get_unexpanded_actions()
            a = np.random.choice(unexpanded_actions)
            node.children[a] = self.game.get_next_state_probs(node.state, a)
            s_prime = self.get_outcome(node.state, a)
            history_addition = [(node.state, a, s_prime)]
            return self.Nodes[s_prime], history_addition

    def Simulate(self, node):
        count = 0
        G = 0
        while self.game.get_state(node.state[0], node.state[1]) != "Food" and count < 500:
            a = np.random.choice(["AU", "AD", "AL", "AR"], p = [0.1, 0.4, 0.1, 0.4])
            s_prime = self.get_outcome(node.state, a)
            G += self.discount ** count * self.game.get_reward(node.state, None, s_prime)
            node = self.Nodes[s_prime]
            count += 1
        return G

    def Backpropagate(self, history, G):
        for i in range(len(history) - 1, -1, -1):
            s = history[i][0]
            a = history[i][1]
            s_prime = history[i][2]
            r = self.game.get_reward(s, None, s_prime)
            node = self.Nodes[s]
            # Update N(s,a)
            node.num_action_visits[a] += 1
            G = r + self.discount * G
            # Update Q(s,a)
            node.q[a] += (1 / node.num_action_visits[a]) * (G - node.q[a])

    def run_mcts(self, s):
        for i in range(5):
            for j in range(5):
                state = (i,j)
                state_node = Node(state)
                self.Nodes[(i,j)] = state_node
        node = self.Nodes[s]
        t_end = time.time() + 5
        # while time.time() < t_end:
        for i in range(30):
            selected_node, history = self.Select(node)
            child, history_addition = self.Expand(selected_node)
            history = history + history_addition
            G = self.Simulate(child)
            self.Backpropagate(history, G)
        return node.q

    def mcts(self):
        s = (1,1)
        while s != (4,4):
            q = self.run_mcts(s)
            a = max(q, key=q.get)
            s = self.get_outcome(s,a)
            print((s,a))



class Node():
    def __init__(self, state):
        self.state = state
        self.children = {}
        # self.num_visits = 0
        self.num_action_visits = {'AU': 0, "AL": 0, "AD": 0, "AR": 0}
        self.q = {'AU': 0, "AL": 0, "AD": 0, "AR": 0}
        self.G = 0

    def set_children(self, children):
        self.children = children

    def fully_expanded(self):
        return len(self.children) == 4
    
    def get_state(self):
        return self.state
    
    def get_num_visits(self):
        return sum(list(self.num_action_visits.values()))
    
    def get_num_action_visits(self, action):
        return self.num_action_visits[action]
    
    def get_G(self):
        return self.G
    
    def get_unexpanded_actions(self):
        actions = ["AU", "AD", "AL", "AR"]
        for act in list(self.children.keys()):
            actions.remove(act)
        return actions

mcts = MCTS()
mcts.mcts()