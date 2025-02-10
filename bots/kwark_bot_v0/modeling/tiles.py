"""
Asetoroid Tiles
These are tiles that are not unit-placeable
??? Creation Rule
    Always seem to be symmetric
    Also seems to repeat after certaion number of iterations --> which means the landscape is predictable
"""

"""
Nebula Tiles
These are tiles that reduce energy & reduce vision
??? Vision Reduction Fn
??? Energy Reduction Fn
??? Creation Rule
"""

"""
Energy Nodes
These are energy source tile with a unique energy function that provides energy to tiles in distance
Energy of a tile = sum(fn(distance from source))
??? Energy Distance Fn
??? Creation Rule (e.g. How many nodes?)
"""

"""
Relic Nodes
This is a harbor node where you can find a Point Node nearby
Rule for creation
    1. Sample k: [1, 3]
    2. Until round k, create a Relic Node in each side symmetrically on the map
    3. From round k + 1 no more relic nodes are created (we run simulation with the existing k nodes)
??? Do relic nodes move?
"""

"""
Point Node
These are tiles that allow agents to score points from energy
These are always placed within a 5x5 squer of neighboring nodes
??? Do point nodes move?
"""
