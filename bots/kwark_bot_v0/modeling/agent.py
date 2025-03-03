from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict
from numpy.typing import NDArray
from .tiles import TileMap, RelicNode


class AgentMove(Enum):
    """
    Agent movements
    There are 5 movements available; Up, Down, Left, Right, center
    Center movemnt is free, all other movements cost energy. Even a illegal move will cost energy (but movement will not be performed)
    """

    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SAP = 5


@dataclass
class AgentAction:
    """
    Agent Actions

    Action is defined as a (N, 3) array
    # Action[:, 0] = type of action {0: stay, 1: up, 2: right, 3: down, 4: left, 5: sap}
    # Action[:, 1, 2] = dx, dy of sap location

    Sap
        * randomized fixed sap range & sap drop off factor: params.unit_sap_range, params.unit_sap_dropoff_factor
        * Valid Sap: params.unit_sap_range >= dx && params.unit_sap_range >= dy
        * saps only opposing unit
        * Saps neighboring tiles as well (splash damage available)
    * Units in the center gets damaged by exact amount of sap_energy
    * Units in the neighboring 8 tiles get sapped by sap_energy * params.unit_sap_dropoff_factor
    * If no sap is to be performed, put 0, 0 in dx and dy
    """

    agent: Agent
    move: AgentMove
    sap: bool
    sap_target: Tuple[int, int]  # dx, dy of sap action

    @staticmethod
    def to_numpy(agent_actions: List[AgentAction]) -> NDArray:
        num_agents = len(agent_actions)
        all_actions = np.zeros((num_agents, 3))
        for action in agent_actions:
            i = action.agent.id
            all_actions[i, 0] = action.move.value
            if action.sap:
                dx, dy = (
                    action.agent.pos[0] - action.sap_target[0],
                    action.agent.pos[1] - action.sap_target[1],
                )
                all_actions[i, 1] = dx
                all_actions[i, 2] = dy
        return all_actions


@dataclass
class Agent:
    id: int  # ID must be 0 ~ self.param_max_units - 1
    pos: Tuple[int, int]
    energy: int
    visible: bool  # Obtained from obs["units_mask"]


class Observation:
    # ['units', 'units_mask', 'sensor_mask', 'map_features', 'relic_nodes', 'relic_nodes_mask', 'team_points', 'team_wins', 'steps', 'match_steps']
    units: List[Agent]
    units_mask: NDArray
    sensor_mask: NDArray
    map_features: Dict[str, float]
    relic_nodes: List[RelicNode]
    relic_nodes_mask: NDArray
    team_points: Tuple[int, int]
    team_wins: int
    steps: int
    match_steps: int


class CommandCenter:
    """
    class that observes the game board and orders the best action to each agent.
    :param agents: List[Agent]
    :param opp_agents: List[Agent]
    :param action_history: Dict[int, List[AgentAction]] list of actions taken by agents in the past


    CommandCenter is responsible of
    * predicting unrevealed game parameters
    * deciding the best move for each agent
    * deciding the best sap target for each agent
    * decide to sap vs. collide

    # Find best distribution of scv vs. marines
    # Find best RelicNode harbor location
    # Find bets EnergyNode location
    """

    def __init__(self):
        self.agents = []
        self.opp_agents = []
        self.action_history = {}
        self.obs_history = []
        self.observed_relic_nodes = []
        self.tile_map = TileMap()
        self.team_points = 0
        self.opp_team_points = 0
        self.steps = 0

    def observe(self, obs: Observation):
        pass
        # Update the game state with the new observation
        # self.tile_map.update(obs.map_features)


# All opponent information is masked with -1 values
# whether the unit exists and is visible to you. units_mask[t][i] is whether team t's unit i can be seen and exists.
#  dict_keys(['units', 'units_mask', 'sensor_mask', 'map_features', 'relic_nodes', 'relic_nodes_mask', 'team_points', 'team_wins', 'steps', 'match_steps'])


# // T is the number of teams (default is 2)
# // N is the max number of units per team
# // W, H are the width and height of the map
# // R is the max number of relic nodes
# {
#   "obs": {
#     "units": {
#       "position": Array(T, N, 2),
#       "energy": Array(T, N, 1)
#     },
#     // whether the unit exists and is visible to you. units_mask[t][i] is whether team t's unit i can be seen and exists.
#     "units_mask": Array(T, N),
#     // whether the tile is visible to the unit for that team
#     "sensor_mask": Array(W, H),
#     "map_features": {
#         // amount of energy on the tile
#         "energy": Array(W, H),
#         // type of the tile. 0 is empty, 1 is a nebula tile, 2 is asteroid
#         "tile_type": Array(W, H)
#     },
#     // whether the relic node exists and is visible to you.
#     "relic_nodes_mask": Array(R),
#     // position of the relic nodes.
#     "relic_nodes": Array(R, 2),
#     // points scored by each team in the current match
#     "team_points": Array(T),
#     // number of wins each team has in the current game/episode
#     "team_wins": Array(T),
#     // number of steps taken in the current game/episode
#     "steps": int,
#     // number of steps taken in the current match
#     "match_steps": int
#   },
#   // number of steps taken in the current game/episode
#   "remainingOverageTime": int, // total amount of time your bot can use whenever it exceeds 2s in a turn
#   "player": str, // your player id
#   "info": {
#     "env_cfg": dict // some of the game's visible parameters
#   }

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
