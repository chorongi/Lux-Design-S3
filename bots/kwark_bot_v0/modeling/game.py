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
to print stuff

Use remaining OverageTiem - for how much time you have to make a movement
"""
