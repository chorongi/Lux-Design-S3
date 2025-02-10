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
"""

"""
Debugging
consol.error("hello")
to print stuff

Use remaining OverageTiem - for how much time you have to make a movement
"""
