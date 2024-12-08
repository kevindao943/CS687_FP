import math
import time

import matplotlib.pyplot as plt
import numpy as np
from PrettyPrint import PrettyPrintTree

from domains.gridworld import Gridworld
from domains.KD_Cat_vs_Monster import Cat_vs_Monster
from really_simple_node import Really_Simple_GridWorld


class MCTS():
    def __init__(self, game, NodeClass, c_p, iterations):
        self.game = game
        self.actions = ["AU", "AD", "AL", "AR"]
        self.actions_art = ["↑","↓","←","→"]
        self.discount = 0.925
        self.C_p = c_p
        self.NodeClass = NodeClass
        self.iterations = iterations

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
        # Keeps getting stuck in an infinite loop :(
        count = 0
        while node.fully_expanded():
            a = self.get_action_UCT(node)
            s_prime = self.get_outcome(node.state, a)
            node = node.children[a][s_prime] # self.NodeClass(s_prime, a, node)
            count += 1
        return node


    def Expand(self, node):
        if self.game.get_state(node.state[0], node.state[1]) == "Food":
            return node
        else:
            unexpanded_actions = node.get_unexpanded_actions()
            a = np.random.choice(unexpanded_actions)
            node.children_dicts[a] = self.game.get_next_state_probs(node.state, a)
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
        while self.game.get_state(node.state[0], node.state[1]) != "Food" and count < 500:
            # a = np.random.choice(["AU", "AD", "AL", "AR"], p = [0.15, 0.35, 0.15, 0.35])
            max_score = -math.inf
            max_a = -1
            for a in ["AU", "AD", "AL", "AR"]:
                state_probs = self.game.get_next_state_probs(node.state, a)
                curr_score = 0
                for key in state_probs.keys():
                    curr_score += state_probs[key] * node.transition_score(node.state, key)
                if curr_score > max_score:
                    max_score = curr_score
                    max_a = a
            s_prime = self.get_outcome(node.state, max_a)
            G += self.discount ** count * self.game.get_reward(node.state, None, s_prime)
            # if a in node.children.keys():
            #     node = node.children.keys[a]
            # else:
            node = self.NodeClass(s_prime, a, node)
            count += 1
        return G

    def Backpropagate(self, node, G):
        node.num_action_visits[node.action] += 1
        # # Update Q(s,a)
        node.q[node.action] += (1 / node.num_action_visits[node.action]) * (G - node.q[node.action])
        while node.parent != None:
            s = node.state
            a = node.action
            s_prime = node.parent
            r = self.game.get_reward(s_prime.state, a, s)
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
        count = 0
        while s != (4,4):
            q = self.run_mcts(s)
            a = max(q, key=q.get)
            s_prime = self.get_outcome(s,a)
            print((s_prime,a))
            count += self.game.get_reward(s, a, s_prime)
            s = s_prime
        return (self.C_p, count)


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

if __name__ == "__main__":
    # # Parameters for domain
    rewards_dict = {'Empty': -0.05, 'Monster': -8, 'Food': 10}
    states_dict = {'Forbidden Furniture': [(2,1),(2,2),(2,3),(3,2)], 'Monster': [(0,3),(4,1)], 'Food': [(4,4)]}
    terminal_states = ['Food']
    wall_states = ['Forbidden Furniture']
    world_size = (5,5)
    # Initializing domain + MDP
    game = Cat_vs_Monster(world_size, states_dict, rewards_dict, terminal_states, wall_states)
    # mcts = MCTS(game, Cats_vs_Monsters_Node)
    class example():
        def __init__(self, game) -> None:
            self.game = game
    # game = example(Really_Simple_GridWorld())
    # game.game = Really_Simple_GridWorld()
    #game = Gridworld()
    # mcts = MCTS(game, GridWorldNode)
    iterations_list = []
    count_list = []
    for i in range(50):
        mcts = MCTS(game, Cats_vs_Monsters_Node, np.random.uniform(1, 5), 50)
        iterations, count = mcts.mcts()
        iterations_list.append(iterations)
        count_list.append(count)
    plt.scatter(iterations_list, count_list)
    m, b = np.polyfit(iterations_list, count_list, 1) 
    plt.plot(iterations_list, m*np.array(iterations_list) + b, color='red')
    plt.show()