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
    * randomized fixed sap range: params.unit_sap_range
    * params.unit_sap_cost, params.unit_sap_cost * sap drop off factor
        * params.unit_sap_range >= dx && params.unit_sap_range >= dy
    * saps only opposing unit
    * Saps neighboring tiles as well (splash damage available)
??? What is the exact fn for sap damage computation (including splash damage)
??? What is the distance sap drop off actor
"""

"""
Unit Creation
Num of units: self.param_max_units
Ids: [0, self.param_max_units - 1]
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
??? (up, down, left, right only) are affected with energy voids?
"""
