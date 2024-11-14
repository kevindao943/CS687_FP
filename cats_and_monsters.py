import math
from pprint import pprint
from typing import Dict, List

import numpy as np


class Cats_and_Monsters_MDP():
    AU = 0
    AR = 1
    AD = 2
    AL = 3

    def __init__(self) -> None:
        self.state = (0,0)
        self.action = self.AU  # "AU", "AD", "AL", "AR"
        self.gamma = 0.2
        self.total_reward = 0
        self.v = [[0]*5 for i in range(5)]
        self.pi = [[-1]*5 for i in range(5)]

    def print_state(self, s) -> None:
        arr = [[" "]*5 for i in range(5)]
        arr[2][1] = "F"
        arr[2][2] = "F"
        arr[3][2] = "F"
        arr[2][3] = "F"
        arr[0][3] = "M"
        arr[4][1] = "M"
        arr[s[0]][s[1]] = "x"
        pprint(arr)

    def print_transition(self,s,a,s_prime):
        self.print_state(s)
        print(f"Attempt {a}")
        print(f"Move from {s} to {s_prime}")
        print(self.total_reward)
        self.print_state(s_prime)
        print("****************************")
                         

    def move_straight(self, s,a) -> tuple:
        # May return out of bounds state. p will take care of this
        if a == 0:
            return (s[0] - 1, s[1])
        elif a == 1:
            return (s[0], s[1] + 1)
        elif a == 2: 
            return (s[0] + 1, s[1])
        elif a == 3:
            return (s[0], s[1] - 1)
        else:
            raise Exception("Invalid Action")
        
    def move_right(self, s,a) -> tuple:
        # May return out of bounds state. p will take care of this
        if a == 3:
            return (s[0] - 1, s[1])
        elif a == 0:
            return (s[0], s[1] + 1)
        elif a == 1: 
            return (s[0] + 1, s[1])
        elif a == 2:
            return (s[0], s[1] - 1)
        else:
            raise Exception("Invalid Action")
        
    def move_left(self, s,a) -> tuple:
        # May return out of bounds state. p will take care of this
        if a == 1:
            return (s[0] - 1, s[1])
        elif a == 2:
            return (s[0], s[1] + 1)
        elif a == 3: 
            return (s[0] + 1, s[1])
        elif a == 0:
            return (s[0], s[1] - 1)
        else:
            raise Exception("Invalid Action")
        
    def is_monster(self, s) -> bool:
        if s == (0,3) or s == (4,1):
            return True
        else:
            return False

    def is_furniture(self, s) -> bool:
        if s == (2,1) or s == (2,2) or s == (2,3) or s == (3,2):
            return True
        else:
            return False

    def is_valid(self, s) -> bool:
        if s[0] >= 0 and s[0] <= 4 and s[1] >= 0 and s[1] <= 4:
            return True
        else:
            return False
    
    def is_terminal(self, s) -> bool:
        if s == (4,4): # or s == (0,1):
            return True
        else:
            return False

    def p(self,s,a) -> Dict:
        #Init with chance cat decides not to move
        if self.is_terminal(s):
            return {s:1.0}
        ret_dict = {s:0.06}
        straight = self.move_straight(s,a)
        left = self.move_left(s,a)
        right = self.move_right(s,a)
        if self.is_valid(straight) and not self.is_furniture(straight):
            ret_dict[straight] = 0.70
        else:
            ret_dict[s] += 0.70
        if self.is_valid(left) and not self.is_furniture(left):
            ret_dict[left] = 0.12
        else:
            ret_dict[s] += 0.12
        if self.is_valid(right) and not self.is_furniture(right):
            ret_dict[right] = 0.12
        else:
            ret_dict[s] += 0.12
        return ret_dict
        

    def take_action(self) -> None:
        action_dict = self.p(self.state,self.action)
        s_prime_index = np.random.choice(a=len(list(action_dict.keys())), p=list(action_dict.values()))
        return list(action_dict.keys())[s_prime_index]

    def get_reward(self, s,a,s_prime) -> float:
        if self.is_terminal(s):
            return 0
        # Remove comment for catnip
        # if s_prime == (0,1):
        #     return 3.91
        if self.is_terminal(s_prime):
            return 10
        if self.is_monster(s_prime):
            return -8
        else:
            return -0.05

    def get_action(self) -> int:
        return np.random.choice([0,1,2,3], p=[0.1, 0.4, 0.4, 0.1])

    def run_episode(self) -> float:
        while not self.is_terminal(self.state):
            self.action = self.get_action()
            s_prime = self.take_action()
            self.total_reward += self.get_reward(self.state, self.action, s_prime)
            self.print_transition(self.state, self.action, s_prime)
            self.state = s_prime
        return self.total_reward
    
    def state_iteration(self,s,a):
        total = 0
        p_curr = self.p(s,a)
        for s_prime, prob in p_curr.items():
            total += prob * (self.get_reward(s,a,s_prime) + self.gamma *self.v[s_prime[0]][s_prime[1]])
        return total


    def in_place_value_iteration(self):
        counter = 0
        while True:
            counter += 1
            delta = 0
            for i in range(5):
                for j in range(5):
                    v_curr = self.v[i][j]
                    v_new_max = -math.inf
                    for a in [0,1,2,3]:
                        v_new_max = max(v_new_max, self.state_iteration((i,j),a))
                    self.v[i][j] = v_new_max
                    delta = max(delta, abs(v_curr - v_new_max))
            if delta < 0.0001:
                break
        for i in range(5):
            for j in range(5):
                v_new_max = -math.inf
                best_a = -1
                for a in [0,1,2,3]:
                    v_curr = self.state_iteration((i,j),a)
                    if(v_curr> v_new_max):
                        best_a = a
                        v_new_max = v_curr
                self.pi[i][j] = best_a
        print(counter)

    def standard_value_iteration(self):
        counter = 0
        while True:
            counter += 1
            delta = 0
            new_v = [[0]*5 for i in range(5)]
            for i in range(5):
                for j in range(5):
                    v_curr = self.v[i][j]
                    v_new_max = -math.inf
                    for a in [0,1,2,3]:
                        v_new_max = max(v_new_max, self.state_iteration((i,j),a))
                    new_v[i][j] = v_new_max
                    delta = max(delta, abs(v_curr - v_new_max))
            for i in range(5):
                for j in range(5):
                    self.v[i][j] = new_v[i][j]
            if delta < 0.0001:
                break
        for i in range(5):
            for j in range(5):
                v_new_max = -math.inf
                best_a = -1
                for a in [0,1,2,3]:
                    v_curr = self.state_iteration((i,j),a)
                    if(v_curr> v_new_max):
                        best_a = a
                        v_new_max = v_curr
                self.pi[i][j] = best_a
        print(counter)
    
    def print_value_and_policy(self):
        for i in range(5):
            string = ""
            for j in range(5):
                if self.is_furniture((i,j)):
                    string += str(round(0.0,4)).ljust(6) + " "
                else:
                    string += str(round(self.v[i][j],4)).ljust(6) + " "
            print(string)
        for i in range(5):
            string = u""
            for j in range(5):
                if self.is_furniture((i,j)):
                    string += u"\u2610 "
                elif self.is_terminal((i,j)):
                    if (i,j) == (4,4):
                        string += "G"
                    else:
                        string += "C "
                else:
                    if self.pi[i][j] == 0:
                        string += u"\u2191 "
                    elif self.pi[i][j] == 1:
                        string += u"\u2192 "
                    elif self.pi[i][j] == 2:
                        string += u"\u2193 "
                    else:
                        string += u"\u2190 "
            print(string)