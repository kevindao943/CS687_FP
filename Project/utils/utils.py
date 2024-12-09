import numpy as np

def get_vmap_policy_map(q, instance):
    nrow, ncol = instance.mdp.state_space

    vmap = [[None for _ in range(ncol)] for _ in range(nrow)]
    policy_map = [[None for _ in range(ncol)] for _ in range(nrow)]
    
    for row in range(nrow):
        for col in range(ncol):
            if instance.mdp.game.get_state(row, col) in instance.mdp.wall_states:
                vmap[row][col] = 'X'
                policy_map[row][col] = 'X'
            else:
                vmap[row][col] = max(q[row][col].values())
                max_action = np.argmax([value for key, value in q[row][col].items()])
                policy_map[row][col] = instance.mdp._get_policy_art(row, col, max_action)
    return vmap, policy_map

def init_func(row, col, wall_states, str_type):
    if (row,col) in wall_states:
        return 'X'
    return 0 if str_type == 'values' else None