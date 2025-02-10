"""
Agent movements
There are 5 movements available; Up, Down, Left, Right, center
Center movemnt is free, all other movements cost energy. Even a illegal move will cost energy (but movement will not be performed)
"""

"""
Agent Actions

Action is defined as a (N, 3) array
# Action[:, 0] = type of action {0: stay, 1: up, 2: right, 3: down, 4: lefet, 5: sap}
# Action[:, 1, 2] = dx, dy of sap location

Sap
    * randomized fixed sap range & sap drop off factor: params.unit_sap_range, params.unit_sap_dropoff_factor
    * Valid Sap: params.unit_sap_range >= dx && params.unit_sap_range >= dy
    * saps only opposing unit
    * Saps neighboring tiles as well (splash damage available)
* Units in the center gets damaged by exact amount of sap_energy
* Units in the neighboring 8 tiles get sapped by sap_energy * params.unit_sap_dropoff_factor
"""

"""
Unit Creation
Num of units: self.param_max_units
A unit is created every 3 steps
Ids: [0, self.param_max_units - 1]
Unit is always spawned on [0, 0] for player 0 and [23, 23] for player 1
"""

"""
Unit Vision
Unit vision are aggregatable. This can be used to observe through nebula tiles
"""

"""
Unit Collision
If collided with opposing units,
Compare(Sum(E_all), Sum(E_opp)) and lower units all die.
Draw will destroy all units
"""

"""
Unit Energy Voids
Neighboring enemy units will be affected with params.unit_energy_void_factor
Neighboring tiles (up, down, left, right only) enemy units will decrease energy: params.unit_energy_void_factor * unit_energy
"""
