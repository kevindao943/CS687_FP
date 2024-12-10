import math
import time

import matplotlib.pyplot as plt
import numpy as np
# from PrettyPrint import PrettyPrintTree

# from Project.domains.cat_vs_monsters import Cat_vs_Monsters
# from Project.domains.gridworld import Gridworld
# from Project.domains.really_simple_node import Really_Simple_GridWorld


class MCTS():
    def __init__(self, game, NodeClass, c_p, iterations, policy):
        self.game = game
        self.actions = self.game.action_space
        self.actions_art = self.game.actions_art
        self.discount = 0.925
        self.C_p = c_p
        self.NodeClass = NodeClass
        self.iterations = iterations
        self.policy = policy

    def get_action_UCT(self, node):
        good_action_list = []
        curr_best = -math.inf
        for a in self.actions:
            UCB1 = node.q[a] + 2 * self.C_p * math.sqrt((2 * math.log(node.get_num_visits())) / node.get_num_action_visits(a))
            if UCB1 > curr_best:
                curr_best = UCB1
                good_action_list = [a]
            elif UCB1 == curr_best:
                good_action_list.append(a)
        return np.random.choice(good_action_list)

    def get_outcome(self, s, a):
        state_probs = self.game.game.get_next_state_probs(s, a)
        return list(state_probs.keys())[np.random.choice(len(state_probs), p = list(state_probs.values()))]

    def Select(self, node):
        # Keeps getting stuck in an infinite loop :(
        count = 0
        while node.fully_expanded():
            a = self.get_action_UCT(node)
            s_prime = self.get_outcome(node.state, a)
            node = node.children[a][s_prime] # self.NodeClass(s_prime, a, node)
            count += 1
        return node


    def Expand(self, node):
        if self.game.game.get_state(node.state[0], node.state[1]) == self.game.game.terminal_states[0]:
            return node
        else:
            unexpanded_actions = node.get_unexpanded_actions()
            a = np.random.choice(unexpanded_actions)
            node.children_dicts[a] = self.game.game.get_next_state_probs(node.state, a)
            child_nodes = {}
            for state in list(node.children_dicts[a].keys()):
                child_node = self.NodeClass(state, a, node)
                child_nodes[state] = child_node
            s_prime = self.get_outcome(node.state, a)
            node.children[a] = child_nodes
            child_node = node.children[a][s_prime]
            return child_node

    def Simulate(self, node):
        count = 0
        G = 0
        while self.game.game.get_state(node.state[0], node.state[1]) != self.game.game.terminal_states[0] and count < 500:
            if self.policy == "Down Left":
                max_a = np.random.choice(self.actions, p = [0.15, 0.35, 0.15, 0.35])
            elif self.policy == "state_score":
                max_score = -math.inf
                max_a = -1
                for a in self.actions:
                    state_probs = self.game.game.get_next_state_probs(node.state, a)
                    curr_score = 0
                    for key in state_probs.keys():
                        curr_score += state_probs[key] * node.transition_score(node.state, key)
                    if curr_score > max_score:
                        max_score = curr_score
                        max_a = a
            elif self.policy == "state_score2":
                max_score = -math.inf
                max_a = -1
                for a in self.actions:
                    state_probs = self.game.game.get_next_state_probs(node.state, a)
                    curr_score = 0
                    for key in state_probs.keys():
                        curr_score += state_probs[key] * node.transition_score2(node.state, key)
                    if curr_score > max_score:
                        max_score = curr_score
                        max_a = a
            else:
                 max_a = np.random.choice(self.actions)
            s_prime = self.get_outcome(node.state, max_a)
            G += self.discount ** count * self.game.game.get_reward(node.state, None, s_prime)
            # if a in node.children.keys():
            #     node = node.children.keys[a]
            # else:
            node = self.NodeClass(s_prime, max_a, node)
            count += 1
        # print(count)
        return G

    def Backpropagate(self, node, G):
        node.num_action_visits[node.action] += 1
        # # Update Q(s,a)
        node.q[node.action] += (1 / node.num_action_visits[node.action]) * (G - node.q[node.action])
        while node.parent != None:
            s = node.state
            a = node.action
            s_prime = node.parent
            r = self.game.game.get_reward(s_prime.state, a, s)
            node = node.parent
            # Update N(s,a)
            node.num_action_visits[a] += 1
            G = r + self.discount * G
            # Update Q(s,a)
            node.q[a] += (1 / node.num_action_visits[a]) * (G - node.q[a])

    def run_mcts(self, s):
        node = self.NodeClass(s, None, None)
        print("Starting new search")
        # t_end = time.time() + 1.5
        # while time.time() < t_end:
        for i in range(self.iterations):
            selected_node = self.Select(node)
            child = self.Expand(selected_node)
            G = self.Simulate(child)
            self.Backpropagate(child, G)
        print("Ending new search")
        # pt = PrettyPrintTree(lambda x: x.get_pretty_children(), lambda x: x.state)
        # pt(node)
        print(node.q)
        return node.q
    

    def mcts(self):
        s = (0,0)
        score = 0
        while self.game.game.get_state(s[0], s[1]) != self.game.game.terminal_states[0]:
            # count += 1
            q = self.run_mcts(s)
            a = max(q, key=q.get)
            s_prime = self.get_outcome(s,a)
            print((s_prime,a))
            score += self.game.game.get_reward(s, a, s_prime)
            s = s_prime
        return (self.iterations, self.C_p, score)


class Cats_vs_Monsters_Node():
    def __init__(self, state, action, parent):
        self.state = state
        self.children_dicts = {}
        self.children = {}
        self.action = action
        self.parent = parent
        self.num_action_visits = {'AU': 0, "AL": 0, "AD": 0, "AR": 0}
        self.q = {'AU': 0, "AL": 0, "AD": 0, "AR": 0}
        self.G = 0
    
    def get_parent(self):
        return self.parent
    
    def get_pretty_children(self):
        ret_list = []
        for action in self.children.keys():
            for child in self.children[action].values():
                ret_list.append(child)
        return ret_list

    def fully_expanded(self):
        return len(self.children_dicts) == 4
    
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
        for act in list(self.children_dicts.keys()):
            actions.remove(act)
        return actions
    
    def state_score(self,s):
        # Self Loops are bad
            # Outside
        # Closer to goal is better
        manhattan_dist_to_monster = math.inf
        for monster in [(4,1), (0,3)]:
            manhattan_dist_to_monster = min(abs(monster[0] - s[0]) + abs(monster[1] - s[1]), manhattan_dist_to_monster)
        # Closer to monster is worse
        manhattan_dist_to_goal = abs(4 - s[0]) + abs(4 - s[1])
        return 0.8 * manhattan_dist_to_monster - 2 * manhattan_dist_to_goal
    
    def transition_score(self, s, s_prime):
        score = self.state_score(s_prime)
        if s == s_prime:
            return score - 2
        else:
            return score

class Really_Simple_Node():
    def __init__(self, state, action, parent):
        self.state = state
        self.children_dicts = {}
        self.children = {}
        self.action = action
        self.parent = parent
        self.num_action_visits = {'AU': 0, "AL": 0, "AD": 0, "AR": 0}
        self.q = {'AU': 0, "AL": 0, "AD": 0, "AR": 0}
        self.G = 0
    
    def get_parent(self):
        return self.parent
    
    def get_pretty_children(self):
        ret_list = []
        for action in self.children.keys():
            for child in self.children[action].values():
                ret_list.append(child)
        return ret_list

    def fully_expanded(self):
        return len(self.children_dicts) == 4
    
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
        for act in list(self.children_dicts.keys()):
            actions.remove(act)
        return actions
    
    def state_score(self,s):
        # Self Loops are bad
            # Outside
        # Closer to goal is better
        manhattan_dist_to_monster = math.inf
        for monster in [(3,1), (1,4)]:
            manhattan_dist_to_monster = min(abs(monster[0] - s[0]) + abs(monster[1] - s[1]), manhattan_dist_to_monster)
        # Closer to monster is worse
        manhattan_dist_to_goal = abs(4 - s[0]) + abs(4 - s[1])
        return 0.8 * manhattan_dist_to_monster - 2 *manhattan_dist_to_goal
    
    def transition_score(self, s, s_prime):
        score = self.state_score(s_prime)
        if s == s_prime:
            return score - 2
        else:
            return score
        
class GridWorldNode():
    def __init__(self, state, action, parent):
        self.state = state
        self.children_dicts = {}
        self.children = {}
        self.action = action
        self.parent = parent
        self.num_action_visits = {'AU': 0, "AL": 0, "AD": 0, "AR": 0}
        self.q = {'AU': 0, "AL": 0, "AD": 0, "AR": 0}
        self.G = 0

    def get_parent(self):
        return self.parent
    
    def get_pretty_children(self):
        ret_list = []
        for action in self.children.keys():
            for child in self.children[action].values():
                ret_list.append(child)
        return ret_list

    def fully_expanded(self):
        return len(self.children_dicts) == 4
    
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
        for act in list(self.children_dicts.keys()):
            actions.remove(act)
        return actions
    
    def state_score(self,s):
        # Self Loops are bad
            # Outside
        # Closer to goal is better
        manhattan_dist_to_monster = math.inf
        for monster in [(2,2), (3,2)]:
            manhattan_dist_to_monster = min(abs(monster[0] - s[0]) + abs(monster[1] - s[1]), manhattan_dist_to_monster)
        # Closer to monster is worse
        manhattan_dist_to_goal = abs(4 - s[0]) + abs(4 - s[1])
        return manhattan_dist_to_monster - 2 * manhattan_dist_to_goal
    
    def transition_score(self, s, s_prime):
        score = self.state_score(s_prime)
        if s == s_prime:
            return score - 2
        else:
            return score

class Really_Big_Node():
    def __init__(self, state, action, parent):
        self.state = state
        self.children_dicts = {}
        self.children = {}
        self.action = action
        self.parent = parent
        self.num_action_visits = {'AU': 0, "AL": 0, "AD": 0, "AR": 0, "AUL": 0, "AUR": 0, "ADL": 0, "ADR": 0}
        self.q = {'AU': 0, "AL": 0, "AD": 0, "AR": 0, "AUL": 0, "AUR": 0, "ADL": 0, "ADR": 0}
        self.G = 0
    
    def get_parent(self):
        return self.parent
    
    def get_pretty_children(self):
        ret_list = []
        for action in self.children.keys():
            for child in self.children[action].values():
                ret_list.append(child)
        return ret_list

    def fully_expanded(self):
        return len(self.children_dicts) == 8
    
    def get_state(self):
        return self.state
    
    def get_num_visits(self):
        return sum(list(self.num_action_visits.values()))
    
    def get_num_action_visits(self, action):
        return self.num_action_visits[action]
    
    def get_G(self):
        return self.G
    
    def get_unexpanded_actions(self):
        actions = ["AU", "AD", "AL", "AR", "AUL", "AUR", "ADL", "ADR"]
        for act in list(self.children_dicts.keys()):
            actions.remove(act)
        return actions
    
    def state_score(self,s):
        # Self Loops are bad
            # Outside
        # Closer to trap is bad
        manhattan_dist_to_trap = math.inf
        for trap in [(2,1), (1,2), (5,0), (0,5), (8,1), (1,8), (2,9), (9,2), (8,4), (4,8), (9,9), (3,5)]:
            manhattan_dist_to_trap = min(abs(trap[0] - s[0]) + abs(trap[1] - s[1]), manhattan_dist_to_trap)
        # Closer to guard is worse
        manhattan_dist_to_guard = math.inf
        for guard in [(4,0), (0,4), (7,0), (0,7), (9,0), (0,9), (3,3), (5,3), (3,5), (9,3), (3,9), (4,4), (5,4), (4,5), (6,5), (5,6), (9,5), (5,9), (7,6), (6,7), (9,8), (8,9)]:
            manhattan_dist_to_guard = min(abs(guard[0] - s[0]) + abs(guard[1] - s[1]), manhattan_dist_to_guard)
        # Closer to princess is better
        manhattan_dist_to_goal = abs(6 - s[0]) + abs(6 - s[1])
        # Inside the castle is better
        if s[0] >= 3 and s[0] <= 6 and s[1] >= 3 and s[1] <= 6:
            inside_the_castle = 1
        else:
            inside_the_castle = 0
        return 10 * manhattan_dist_to_trap + 20 * manhattan_dist_to_guard - 50 * manhattan_dist_to_goal + 75 * inside_the_castle
    
    def transition_score(self, s, s_prime):
        score = self.state_score(s_prime)
        if s == s_prime:
            return score - 50
        else:
            return score
        
    def state_score2(self,s):
        # Self Loops are bad
            # Outside
        # Closer to trap is bad
        manhattan_dist_to_trap = math.inf
        for trap in [(2,1), (1,2), (5,0), (0,5), (8,1), (1,8), (2,9), (9,2), (8,4), (4,8), (9,9), (3,5)]:
            manhattan_dist_to_trap = min(abs(trap[0] - s[0]) + abs(trap[1] - s[1]), manhattan_dist_to_trap)
        # Closer to guard is worse
        manhattan_dist_to_guard = math.inf
        for guard in [(4,0), (0,4), (7,0), (0,7), (9,0), (0,9), (3,3), (5,3), (3,5), (9,3), (3,9), (4,4), (5,4), (4,5), (6,5), (5,6), (9,5), (5,9), (7,6), (6,7), (9,8), (8,9)]:
            manhattan_dist_to_guard = min(abs(guard[0] - s[0]) + abs(guard[1] - s[1]), manhattan_dist_to_guard)
        # Closer to princess is better
        manhattan_dist_to_goal = abs(6 - s[0]) + abs(6 - s[1])
        # Inside the castle is better
        if s[0] >= 3 and s[0] <= 6 and s[1] >= 3 and s[1] <= 6:
            inside_the_castle = 1
        else:
            inside_the_castle = 0
        # Get away from start
        if s[0] >= 0 and s[0] <= 2 and s[1] >= 0 and s[1] <= 2:
            close_to_start = 1
        else:
            close_to_start = 0
        return 10 * manhattan_dist_to_trap + 20 * manhattan_dist_to_guard + 50 * close_to_start - 50 * manhattan_dist_to_goal + 75 * inside_the_castle - 75 * close_to_start
    
    def transition_score2(self, s, s_prime):
        score = self.state_score(s_prime)
        if s == s_prime:
            return score - 50
        else:
            return score
    
    def show_default(self):
        row_arr = []
        for i in range(10):
            row = []
            for j in range(10):
                row.append(self.state_score((i,j)))
            row_arr.append(row)
        return row_arr
