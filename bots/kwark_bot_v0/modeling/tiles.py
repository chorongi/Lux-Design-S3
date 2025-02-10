"""
Asetoroid Tiles
These are tiles that are not unit-placeable

* Moving logic - same for nebula tiles
new_tile_types_map = jnp.roll(
    state.map_features.tile_type,
    shift=(
        1 * jnp.sign(params.nebula_tile_drift_speed),
        -1 * jnp.sign(params.nebula_tile_drift_speed),
    ),
    axis=(0, 1),
)
Which is performing
[[1, 2, 3, 4],
[5, 6, 7, 8],
[9, 10, 11, 12]]
-->
[[10, 11, 12, 9],
[2, 3, 4, 1],
[6, 7, 8, 9]]
and Vice versa if the drift speed is negative

Always generated to be symmetric
"""

"""
Nebula Tiles
These are tiles that reduce energy & reduce vision
Vision Reduction
* vision_power_map - (state.map_features.tile_type == NEBULA_TILE) * params.nebula_tile_vision_reduction
Energy Reduction
* energy_map - (state.map_features.tile_type[x, y] == NEBULA_TILE) * params.nebula_tile_energy_reduction

Nebula Tiles always move together with asteroid tiles as well
"""

"""
Energy Nodes
These are energy source tile with a unique energy function that provides energy to tiles in distance
Energy of a tile = sum(fn(distance from source))
Energy Distance Fn
* ENERGY_NODE_FNS = [lambda d, x, y, z: jnp.sin(d * x + y) * z, lambda d, x, y, z: (x / (d + 1) + y) * z]
# fn_i, x, y, z
energy_node_fns = jnp.array(
    [
        [0, 1.2, 1, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        # [1, 4, 0, 2],
        [0, 1.2, 1, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        # [1, 4, 0, 0]
    ]
)
energy_field: (6, 24, 24)
-> shift such that mean is >= 0.25
-> sum on dim(0)
-> final_energy_field: (24, 24)

Creation Rule (e.g. How many nodes?)
* noise = generate_perlin_noise_2d(subkey, (map_height, map_width), (4, 4))
* highest_positions = jnp.column_stack(jnp.unravel_index(flat_indices, noise.shape)).astype(jnp.int16)
* energy_nodes = jnp.concat([highest_positions, mirrored_positions], axis=0)
"""

"""
Relic Nodes
This is a harbor node where you can find a Point Node nearby
Rule for creation
    1. Sample k: [1, 3]
    2. Until round k, create a Relic Node in each side symmetrically on the map
    3. From round k + 1 no more relic nodes are created (we run simulation with the existing k nodes)
Relic Nodes do not move throughout the match
"""

"""
Point Node
These are tiles that allow agents to score points from energy
These are always placed within a "5x5 square" of neighboring nodes
Density of 20% nearby tiles can yield points
Fixed throughout each round. (Doesn't move)
"""
