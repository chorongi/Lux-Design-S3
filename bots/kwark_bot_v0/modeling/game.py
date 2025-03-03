"""
Game Steps
1. Move
2. Sap
3. Collisions & Energy Void fields
(2, 3 happen simultaneously)
4. Update all energy of units
5. Spawn / Remove units
6. Compute team vision & masks
7. Move Asteroid / Nebula / Energy Nodes
8. Compute team points

Repeat Until total steps = params.max_steps_in_match
"""

"""
Game parameters
--------------------------
# Observable parameters
1. unit move cost
2. unit sap cost
3. unit sap range
--------------------------
# Secret parameters
4. unit sensor range
5. nebula vision reduction fn
6. nebula energy reduction fn
7. sap drop off factor
8. nebula drift speed
9. energy node drift speed
10. energy node drift magnitude

env_params_ranges = dict(
    # map_type=[1],
    unit_move_cost=list(range(1, 6)),
    unit_sensor_range=[1, 2, 3, 4],
    nebula_tile_vision_reduction=list(range(0, 8)),
    nebula_tile_energy_reduction=[0, 1, 2, 3, 5, 25],
    unit_sap_cost=list(range(30, 51)),
    unit_sap_range=list(range(3, 8)),
    unit_sap_dropoff_factor=[0.25, 0.5, 1],
    unit_energy_void_factor=[0.0625, 0.125, 0.25, 0.375],
    # map randomizations
    nebula_tile_drift_speed=[-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15],
    energy_node_drift_speed=[0.01, 0.02, 0.03, 0.04, 0.05],
    energy_node_drift_magnitude=list(range(3, 6)),
)
"""

"""
Debugging
consol.error("hello")
print("hello", file=sys.stderr)

Use remaining OverageTiem - for how much time you have to make a movement
"""

from dataclasses import dataclass


@dataclass
class GameParams:
    sensor_range: int
    sap_drop_off_factor: float
    nebula_drift_speed: float
    energy_node_drift_speed: float
    energy_node_drift_magnitude: int


# player_1, ./bots/kwark_bot_v0//main.py: sensor_mask: [[False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False  True  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False  True
# player_1, ./bots/kwark_bot_v0//main.py:    True  True  True False False  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False  True
# player_1, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False  True
# player_1, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False  True
# player_1, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False  True
# player_1, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False  True
# player_1, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False  True
# player_1, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False  True  True  True  True  True  True  True  True]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_1, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]]
# player_0, ./bots/kwark_bot_v0//main.py: units_mask: [[ True  True  True  True  True  True  True  True  True  True  True  True
# player_0, ./bots/kwark_bot_v0//main.py:    True  True  True  True]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False]]
# player_0, ./bots/kwark_bot_v0//main.py: sensor_mask: [[False False  True  True  True  True  True  True  True  True  True  True
# player_0, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False  True  True  True  True  True  True  True  True  True  True
# player_0, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False  True  True  True  True  True  True  True  True  True  True
# player_0, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False  True  True  True  True  True  True  True  True  True  True
# player_0, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False  True  True  True  True  True  True  True  True  True  True
# player_0, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False  True  True  True  True  True  True  True  True  True  True
# player_0, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False  True  True  True  True  True  True
# player_0, ./bots/kwark_bot_v0//main.py:    True  True  True  True  True  True  True False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False  True  True
# player_0, ./bots/kwark_bot_v0//main.py:   False False False  True  True  True  True False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False  True  True  True  True  True  True  True False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False  True  True  True  True  True  True  True False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False  True  True  True  True  True  True  True False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False  True  True  True  True  True  True  True False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False  True  True  True  True  True False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False  True  True  True  True  True False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False  True  True False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]
# player_0, ./bots/kwark_bot_v0//main.py:  [False False False False False False False False False False False False
# player_0, ./bots/kwark_bot_v0//main.py:   False False False False False False False False False False False False]]
# player_1, ./bots/kwark_bot_v0//main.py: map_features: {'energy': array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1,  6, 10,  8,  1, -5],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1,  6, 10,  7, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -5,  1,  8, 10,  4],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         10,  6, -1, -6, -4,  3,  9,  8],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1,  5, -3, -6, -2,  6, 10],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  9,  4, -2, -6, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1,  9,  9,  2, -5, -5,  2,  9],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  9,  9,  4, -3, -6,
# player_1, ./bots/kwark_bot_v0//main.py:         -2,  5, 10,  6, -2, -6, -2,  6],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  4,  9,  9,  2, -5,
# player_1, ./bots/kwark_bot_v0//main.py:         -6,  1,  8,  9,  2, -5, -5,  3],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  5, 10,  7, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -6, -3,  5, 10,  6, -3, -6,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -6,  0,  8, 10,  4,
# player_1, ./bots/kwark_bot_v0//main.py:         -4, -5,  2,  9,  8,  0, -6, -2],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -6, -4,  4, 10,  7,
# player_1, ./bots/kwark_bot_v0//main.py:         -2, -6, -1,  7,  9,  2, -5, -4],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -3, -6,  0,  8,  9,
# player_1, ./bots/kwark_bot_v0//main.py:          1, -6, -3,  6, 10,  4, -4, -5],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:          3, -5, -4,  4, 10,  5, -4, -6],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1]]), 'tile_type': array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1,  0,  0,  0,  2,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1,  0,  0,  0,  2,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1,  0,  0,  0,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:          0,  0,  0,  0,  0,  0,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1,  1,  1,  0,  0,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1,  1,  1,  0,  0,  0,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,
# player_1, ./bots/kwark_bot_v0//main.py:          1,  1,  0,  0,  0,  0,  0,  2],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  2,  0,
# player_1, ./bots/kwark_bot_v0//main.py:          0,  0,  0,  0,  0,  2,  2,  2],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  2,  2,
# player_1, ./bots/kwark_bot_v0//main.py:          0,  0,  0,  0,  2,  0,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  2,  2,
# player_1, ./bots/kwark_bot_v0//main.py:          0,  0,  0,  0,  0,  0,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  0,  0,  2,  2,
# player_1, ./bots/kwark_bot_v0//main.py:          0,  0,  0,  0,  0,  0,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  2,  0,  0,  0,
# player_1, ./bots/kwark_bot_v0//main.py:          0,  0,  0,  0,  0,  0,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:          0,  0,  0,  0,  0,  2,  0,  0],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1],
# player_1, ./bots/kwark_bot_v0//main.py:        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# player_1, ./bots/kwark_bot_v0//main.py:         -1, -1, -1, -1, -1, -1, -1, -1]])}
# player_1, ./bots/kwark_bot_v0//main.py: relic_nodes: [[11 21]
# player_1, ./bots/kwark_bot_v0//main.py:  [-1 -1]
# player_1, ./bots/kwark_bot_v0//main.py:  [-1 -1]
# player_1, ./bots/kwark_bot_v0//main.py:  [-1 -1]
# player_1, ./bots/kwark_bot_v0//main.py:  [-1 -1]
# player_1, ./bots/kwark_bot_v0//main.py:  [-1 -1]]
# player_1, ./bots/kwark_bot_v0//main.py: relic_nodes_mask: [ True False False False False False]
# player_1, ./bots/kwark_bot_v0//main.py: team_points: [102 102]
# player_1, ./bots/kwark_bot_v0//main.py: team_wins: [0 0]
# player_1, ./bots/kwark_bot_v0//main.py: steps: 80
# player_1, ./bots/kwark_bot_v0//main.py: match_steps: 80
