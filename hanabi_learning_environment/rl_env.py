# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RL environment for Hanabi, using an API similar to OpenAI Gym."""

from __future__ import absolute_import
from __future__ import division
import random

from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment.pyhanabi import color_char_to_idx, HanabiMoveType

import tensorflow as tf
from rainbow_agent import RainbowAgent
from dqn_agent import DQNAgent
from run_experiment import ObservationStacker
        
MOVE_TYPES = [_.name for _ in pyhanabi.HanabiMoveType]

# -------------------------------------------------------------------------------
# Environment API
# -------------------------------------------------------------------------------


class Environment(object):
    """Abstract Environment interface.

    All concrete implementations of an environment should derive from this
    interface and implement the method stubs.
    """

    def reset(self, config):
        """Reset the environment with a new config.

        Signals environment handlers to reset and restart the environment using
        a config dict.

        Args:
          config: dict, specifying the parameters of the environment to be
            generated.

        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.

        Args:
          action: dict, mapping to an action taken by an agent.

        Returns:
          observation: dict, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")


class HanabiEnv(Environment):
    """RL interface to a Hanabi environment.

    ```python

    environment = rl_env.make()
    config = { 'players': 5 }
    observation = environment.reset(config)
    while not done:
        # Agent takes action
        action =  ...
        # Environment take a step
        observation, reward, done, info = environment.step(action)
    ```
    """

    def __init__(self, config):
        r"""Creates an environment with the given game configuration.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - colors: int, Number of colors \in [2,5].
              - ranks: int, Number of ranks \in [2,5].
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0).
              - max_life_tokens: int, Number of life tokens (>=1).
              - observation_type: int.
                0: Minimal observation.
                1: First-order common knowledge observation.
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
        """
        assert isinstance(config, dict), "Expected config to be of type dict."
        self.game = pyhanabi.HanabiGame(config)

        self.observation_encoder = pyhanabi.ObservationEncoder(
            self.game, pyhanabi.ObservationEncoderType.CANONICAL)
        self.players = self.game.num_players()

    def reset(self):
        r"""Resets the environment for a new game.

        Returns:
          observation: dict, containing the full observation about the game at the
            current step. *WARNING* This observation contains all the hands of the
            players and should not be passed to the agents.
            An example observation:
            {'current_player': 0,
             'player_observations': [{'current_player': 0,
                                      'current_player_offset': 0,
                                      'deck_size': 40,
                                      'discard_pile': [],
                                      'fireworks': {'B': 0,
                                                    'G': 0,
                                                    'R': 0,
                                                    'W': 0,
                                                    'Y': 0},
                                      'information_tokens': 8,
                                      'legal_moves': [{'action_type': 'PLAY',
                                                       'card_index': 0},
                                                      {'action_type': 'PLAY',
                                                       'card_index': 1},
                                                      {'action_type': 'PLAY',
                                                       'card_index': 2},
                                                      {'action_type': 'PLAY',
                                                       'card_index': 3},
                                                      {'action_type': 'PLAY',
                                                       'card_index': 4},
                                                      {'action_type':
                                                      'REVEAL_COLOR',
                                                       'color': 'R',
                                                       'target_offset': 1},
                                                      {'action_type':
                                                      'REVEAL_COLOR',
                                                       'color': 'G',
                                                       'target_offset': 1},
                                                      {'action_type':
                                                      'REVEAL_COLOR',
                                                       'color': 'B',
                                                       'target_offset': 1},
                                                      {'action_type': 'REVEAL_RANK',
                                                       'rank': 0,
                                                       'target_offset': 1},
                                                      {'action_type': 'REVEAL_RANK',
                                                       'rank': 1,
                                                       'target_offset': 1},
                                                      {'action_type': 'REVEAL_RANK',
                                                       'rank': 2,
                                                       'target_offset': 1}],
                                      'life_tokens': 3,
                                      'observed_hands': [[{'color': None, 'rank':
                                      -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1}],
                                                         [{'color': 'G', 'rank': 2},
                                                          {'color': 'R', 'rank': 0},
                                                          {'color': 'R', 'rank': 1},
                                                          {'color': 'B', 'rank': 0},
                                                          {'color': 'R', 'rank':
                                                          1}]],
                                      'num_players': 2,
                                      'vectorized': [ 0, 0, 1, ... ]},
                                     {'current_player': 0,
                                      'current_player_offset': 1,
                                      'deck_size': 40,
                                      'discard_pile': [],
                                      'fireworks': {'B': 0,
                                                    'G': 0,
                                                    'R': 0,
                                                    'W': 0,
                                                    'Y': 0},
                                      'information_tokens': 8,
                                      'legal_moves': [],
                                      'life_tokens': 3,
                                      'observed_hands': [[{'color': None, 'rank':
                                      -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1}],
                                                         [{'color': 'W', 'rank': 2},
                                                          {'color': 'Y', 'rank': 4},
                                                          {'color': 'Y', 'rank': 2},
                                                          {'color': 'G', 'rank': 0},
                                                          {'color': 'W', 'rank':
                                                          1}]],
                                      'num_players': 2,
                                      'vectorized': [ 0, 0, 1, ... ]}]}
        """
        # print("Pass in HanabiEnv ({class_name}) - reset".format(class_name=self.__class__))
        
        self.state = self.game.new_initial_state()

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        obs = self._make_observation_all_players()
        obs["current_player"] = self.state.cur_player()
        return obs

    def vectorized_observation_shape(self):
        """Returns the shape of the vectorized observation.

        Returns:
          A list of integer dimensions describing the observation shape.
        """
        return self.observation_encoder.shape()

    def num_moves(self, player_id=None):
        """Returns the total number of moves in this game (legal or not).

        Returns:
          Integer, number of moves.
        """
        return self.game.max_moves()

    def step(self, action):
        """Take one step in the game.

        Args:
          action: dict, mapping to a legal action taken by an agent. The following
            actions are supported:
              - { 'action_type': 'PLAY', 'card_index': int }
              - { 'action_type': 'DISCARD', 'card_index': int }
              - {
                  'action_type': 'REVEAL_COLOR',
                  'color': str,
                  'target_offset': int >=0
                }
              - {
                  'action_type': 'REVEAL_RANK',
                  'rank': str,
                  'target_offset': int >=0
                }
            Alternatively, action may be an int in range [0, num_moves()).

        Returns:
          observation: dict, containing the full observation about the game at the
            current step. *WARNING* This observation contains all the hands of the
            players and should not be passed to the agents.
            An example observation:
            {'current_player': 0,
             'player_observations': [{'current_player': 0,
                                'current_player_offset': 0,
                                'deck_size': 40,
                                'discard_pile': [],
                                'fireworks': {'B': 0,
                                          'G': 0,
                                          'R': 0,
                                          'W': 0,
                                          'Y': 0},
                                'information_tokens': 8,
                                'legal_moves': [{'action_type': 'PLAY',
                                             'card_index': 0},
                                            {'action_type': 'PLAY',
                                             'card_index': 1},
                                            {'action_type': 'PLAY',
                                             'card_index': 2},
                                            {'action_type': 'PLAY',
                                             'card_index': 3},
                                            {'action_type': 'PLAY',
                                             'card_index': 4},
                                            {'action_type': 'REVEAL_COLOR',
                                             'color': 'R',
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_COLOR',
                                             'color': 'G',
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_COLOR',
                                             'color': 'B',
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_RANK',
                                             'rank': 0,
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_RANK',
                                             'rank': 1,
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_RANK',
                                             'rank': 2,
                                             'target_offset': 1}],
                                'life_tokens': 3,
                                'observed_hands': [[{'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1}],
                                               [{'color': 'G', 'rank': 2},
                                                {'color': 'R', 'rank': 0},
                                                {'color': 'R', 'rank': 1},
                                                {'color': 'B', 'rank': 0},
                                                {'color': 'R', 'rank': 1}]],
                                'num_players': 2,
                                'vectorized': [ 0, 0, 1, ... ]},
                               {'current_player': 0,
                                'current_player_offset': 1,
                                'deck_size': 40,
                                'discard_pile': [],
                                'fireworks': {'B': 0,
                                          'G': 0,
                                          'R': 0,
                                          'W': 0,
                                          'Y': 0},
                                'information_tokens': 8,
                                'legal_moves': [],
                                'life_tokens': 3,
                                'observed_hands': [[{'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1}],
                                               [{'color': 'W', 'rank': 2},
                                                {'color': 'Y', 'rank': 4},
                                                {'color': 'Y', 'rank': 2},
                                                {'color': 'G', 'rank': 0},
                                                {'color': 'W', 'rank': 1}]],
                                'num_players': 2,
                                'vectorized': [ 0, 0, 1, ... ]}]}
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        if isinstance(action, dict):
            # Convert dict action HanabiMove
            action = self._build_move(action)
        elif isinstance(action, int):
            # Convert int action into a Hanabi move.
            action = self.game.get_move(action)
        else:
            raise ValueError("Expected action as dict or int, got: {}".format(
                action))

        last_score = self.state.score()
        # Apply the action to the state.
        self.state.apply_move(action)

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        observation = self._make_observation_all_players()
        done = self.state.is_terminal()
        # Reward is score differential. May be large and negative at game end.
        reward = self.state.score() - last_score
        info = {}

        return (observation, reward, done, info)

    def _make_observation_all_players(self):
        """Make observation for all players.

        Returns:
          dict, containing observations for all players.
        """
        # print("Pass in HanabiEnv ({class_name}) - make obs".format(class_name=self.__class__))
        
        obs = {}
        player_observations = [self._extract_dict_from_backend(
            player_id, self.state.observation(player_id))
            for player_id in range(self.players)]  # pylint: disable=bad-continuation
        obs["player_observations"] = player_observations
        obs["current_player"] = self.state.cur_player()
        return obs

    def _extract_dict_from_backend(self, player_id, observation):
        """Extract a dict of features from an observation from the backend.

        Args:
          player_id: Int, player from whose perspective we generate the observation.
          observation: A `pyhanabi.HanabiObservation` object.

        Returns:
          obs_dict: dict, mapping from HanabiObservation to a dict.
        """
        # print("Pass in HanabiEnv ({class_name}) - extract dict".format(class_name=self.__class__))
        obs_dict = {}
        obs_dict["current_player"] = self.state.cur_player()
        obs_dict["current_player_offset"] = observation.cur_player_offset()
        obs_dict["life_tokens"] = observation.life_tokens()
        obs_dict["information_tokens"] = observation.information_tokens()
        obs_dict["num_players"] = observation.num_players()
        obs_dict["deck_size"] = observation.deck_size()

        obs_dict["fireworks"] = {}
        fireworks = self.state.fireworks()
        for color, firework in zip(pyhanabi.COLOR_CHAR, fireworks):
            obs_dict["fireworks"][color] = firework

        obs_dict["legal_moves"] = []
        obs_dict["legal_moves_as_int"] = []
        for move in observation.legal_moves():
            obs_dict["legal_moves"].append(move.to_dict())
            obs_dict["legal_moves_as_int"].append(self.game.get_move_uid(move))

        obs_dict["observed_hands"] = []
        for player_hand in observation.observed_hands():
            cards = [card.to_dict() for card in player_hand]
            obs_dict["observed_hands"].append(cards)

        obs_dict["discard_pile"] = [
            card.to_dict() for card in observation.discard_pile()
        ]

        # Return hints received.
        obs_dict["card_knowledge"] = []
        for player_hints in observation.card_knowledge():
            player_hints_as_dicts = []
            for hint in player_hints:
                hint_d = {}
                if hint.color() is not None:
                    hint_d["color"] = pyhanabi.color_idx_to_char(hint.color())
                else:
                    hint_d["color"] = None
                hint_d["rank"] = hint.rank()
                player_hints_as_dicts.append(hint_d)
            obs_dict["card_knowledge"].append(player_hints_as_dicts)

        # ipdb.set_trace()
        obs_dict["vectorized"] = self.observation_encoder.encode(observation)
        obs_dict["pyhanabi"] = observation

        return obs_dict

    def _build_move(self, action):
        """Build a move from an action dict.

        Args:
          action: dict, mapping to a legal action taken by an agent. The following
            actions are supported:
              - { 'action_type': 'PLAY', 'card_index': int }
              - { 'action_type': 'DISCARD', 'card_index': int }
              - {
                  'action_type': 'REVEAL_COLOR',
                  'color': str,
                  'target_offset': int >=0
                }
              - {
                  'action_type': 'REVEAL_RANK',
                  'rank': str,
                  'target_offset': int >=0
                }

        Returns:
          move: A `HanabiMove` object constructed from action.

        Raises:
          ValueError: Unknown action type.
        """
        assert isinstance(
            action, dict), "Expected dict, got: {}".format(action)
        assert "action_type" in action, ("Action should contain `action_type`. "
                                         "action: {}").format(action)
        action_type = action["action_type"]
        assert (action_type in MOVE_TYPES), (
            "action_type: {} should be one of: {}".format(action_type, MOVE_TYPES))

        if action_type == "PLAY":
            card_index = action["card_index"]
            move = pyhanabi.HanabiMove.get_play_move(card_index=card_index)
        elif action_type == "DISCARD":
            card_index = action["card_index"]
            move = pyhanabi.HanabiMove.get_discard_move(card_index=card_index)
        elif action_type == "REVEAL_RANK":
            target_offset = action["target_offset"]
            rank = action["rank"]
            move = pyhanabi.HanabiMove.get_reveal_rank_move(
                target_offset=target_offset, rank=rank)
        elif action_type == "REVEAL_COLOR":
            target_offset = action["target_offset"]
            assert isinstance(action["color"], str)
            color = color_char_to_idx(action["color"])
            move = pyhanabi.HanabiMove.get_reveal_color_move(
                target_offset=target_offset, color=color)
        else:
            raise ValueError("Unknown action_type: {}".format(action_type))

        legal_moves = self.state.legal_moves()
        assert (str(move) in map(
            str,
            legal_moves)), "Illegal action: {}. Move should be one of : {}".format(
                move, legal_moves)

        return move

class HanabiInt2ActEnv(HanabiEnv):
    
    def __init__(self, config):
        super().__init__(config)

        self.history_size = 1

        self.base_dir = config.get('base_dir', None)
        self.agent_pos = config.get('pos', self.players-1)
        self.trust_rate = config.get('trust_rate', 0.2)
        
        self.intent_type = config.get(
            'intent_type', [HanabiIntentEnv.default_intent for _ in range(self.players)])
        if not isinstance(self.intent_type, list) and isinstance(self.intent_type, str):
            self.intent_type = [self.intent_type for _ in range(self.players)]
        # Check if valid intent type
        for i, o_t in enumerate(self.intent_type):
            assert o_t in HanabiIntentEnv.intent_classes.keys(), "'intent_type[{}]' must be in {}, got {}".format(i, HanabiIntentEnv.intent_classes.keys(), o_t)
        
        
        obs_stacker = ObservationStacker(self.history_size,
                                    super().vectorized_observation_shape()[0],
                                    self.players)

        self.intent_agent  = RainbowAgent(observation_size=obs_stacker.observation_size(),
                            num_actions=len(HanabiIntentEnv.intent_classes[self.intent_type[self.agent_pos]]),
                            num_players=self.players)
        
        self.intent_agent.obs_stacker = obs_stacker
        
        if self.base_dir is not None:
            self.load_intent_policy(self.base_dir)
            
        self.intent2actions = {"play":[HanabiMoveType.PLAY], 
                               "discard":[HanabiMoveType.DISCARD],
                               "no_intent":[HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK, HanabiMoveType.DEAL, HanabiMoveType.INVALID]}

    def reset(self):
        assert self.base_dir is not None, "An IntentAgent policy must be loaded"
        return super().reset()

    def load_intent_policy(self, base_dir):
        self.base_dir = base_dir
        checkpoint_dir = '{}/checkpoints'.format(self.base_dir)

        self.intent_agent._saver.restore(self.intent_agent._sess, tf.train.latest_checkpoint(checkpoint_dir))     
    
    def vectorized_observation_shape(self, pos=-1):
        return (super().vectorized_observation_shape()[0] + len(HanabiIntentEnv.intent_classes[self.intent_type[pos]]),)
    
    def step(self, action):
        if isinstance(action, dict):
            # Convert dict action HanabiMove
            tmp_a = self._build_move(action)
        elif isinstance(action, int):
            # Convert int action into a Hanabi move.
            tmp_a = self.game.get_move(action)
        else:
            raise ValueError("Expected action as dict or int, got: {}".format(action))
        penalty = 0
        if self.state.cur_player() == self.agent_pos:
            if tmp_a.type() not in self.intent2actions[self.prev_intent]:
                penalty = self.trust_rate
        obs, rew, done, info = super().step(action)
        return (obs, rew - penalty, done, info)
    
    def _extract_dict_from_backend(self, player_id, observation):
        # print("Pass in HanabiInt2ActEnv ({class_name}) - extract dict".format(class_name=self.__class__))
        obs_dict = super()._extract_dict_from_backend(player_id, observation)
        if player_id == self.agent_pos:
            # Add intent at the end of the vectorized observation
            intent_class = HanabiIntentEnv.intent_classes[self.intent_type[player_id]]
            
            intent = self._get_intent(player_id, obs_dict)
            # print("!!!!!!!!!INTENT",intent)
            if obs_dict["current_player"] == player_id:
                self.prev_intent = intent_class[intent]
                obs_dict['vectorized'] += [int(i==intent) for i in range(len(intent_class))]
            else:
                obs_dict['vectorized'] += [0 for _ in range(len(intent_class))]
        else:
            obs_dict['vectorized'] += [0 for _ in range(len(HanabiIntentEnv.intent_classes[self.intent_type[self.agent_pos]]))]
            
        return obs_dict
        
    def _get_intent(self, player_id, obs_dict):
        # Create observation dict for IntentAgent
        int_obs_dict = obs_dict.copy()
        intent_class = HanabiIntentEnv.intent_classes[self.intent_type[player_id]]
        
        # Transform legal_moves to fit IntentAgent
        bool_act = [False]*len(intent_class)
        valide_act = dict(zip(intent_class, bool_act))
        
        # Check valide intents
        for legal_move_as_dict in obs_dict["legal_moves"]:
            # legal_move_as_dict = move.to_dict()
            if legal_move_as_dict['action_type'] is "PLAY":
                valide_act["play"] = True
            if legal_move_as_dict['action_type'] is "DISCARD":
                valide_act["discard"] = True
            if "REVEAL" in legal_move_as_dict['action_type']:
                valide_act["no_intent"] = True
                
        # Create lists of legal intents
        int_obs_dict["legal_moves"] = []
        int_obs_dict["legal_moves_as_int"] = []
        for ind, (key, valide) in enumerate(valide_act.items()):
            if valide:
                int_obs_dict["legal_moves"].append(key)
                int_obs_dict["legal_moves_as_int"].append(ind)
        
        return self.intent_agent.greedy_action(int_obs_dict)
            
class HanabiIntentEnv(HanabiEnv):
    """RL interface to a Hanabi environment.

    ```python

    environment = rl_env.make("Hanabi-Intent")
    config = { 'players': 2 }
    observation = environment.reset(config)
    while not done:
        # Agent takes action
        action =  ...
        # Environment take a step
        observation, reward, done, info = environment.step(action)
    ```
    """
    intent_classes = {
                    "any": [],
                    "pdn": ["play", "discard", "no_intent"],
                    "pdn_num": ["play_0", "play_1","play_2", "play_3", "play_4", "discard_0", "discard_1","discard_2", "discard_3", "discard_4", "no_intent"]
                    }

    default_intent = "any"

    def __init__(self, config={}):
        super(HanabiIntentEnv, self).__init__(config=config)
        self.intent_type = config.get(
            'intent_type', [HanabiIntentEnv.default_intent for _ in range(self.players)])
        if not isinstance(self.intent_type, list) and isinstance(self.intent_type, str):
            self.intent_type = [self.intent_type for _ in range(self.players)]
        # Check if valid intent type
        for i, o_t in enumerate(self.intent_type):
            assert o_t in HanabiIntentEnv.intent_classes.keys(), "'intent_type[{}]' must be in {}, got {}".format(i, HanabiIntentEnv.intent_classes.keys(), o_t)
        
        self.intents = [HanabiIntentEnv.intent_classes[i] for i in self.intent_type]

    def num_moves(self, player_id=None):
        return len(self.intents[player_id]) if player_id is not None and len(self.intents[player_id]) > 0 else self.game.max_moves()

    def step(self, intent):
        curr_player = self.state.cur_player()

        action = self._intent2action(intent, curr_player)

        return super(HanabiIntentEnv, self).step(action)

    def _intent2action(self, intent, player):
        if self.intent_type[player] == "any":
            return intent
        
        # Get predicted intent as string
        if isinstance(intent, str):
            intent_name = intent
        elif isinstance(intent, int):
            intent_name = self.intents[player][intent]
        else:
            raise ValueError("Expected action as dict or int, got: {}".format(intent))

        # Get legal moves
        legal_moves = []
        for move in self.state.observation(player).legal_moves():
            legal_moves.append(move.to_dict())
            

        # PLAY or DISCARD intent
        if intent_name == "play" or intent_name == "discard":
            # Get infos about recent moves
            move_history = self.state.move_history()
            # print(move_history)
            
            for move in reversed(move_history):
                if move.player() != player and "REVEAL" in move.move().to_dict()["action_type"]:
                    new_reav = move.card_info_revealed()
                    # print(new_reav)
                    act_d = {'action_type': intent_name.upper(), 'card_index': new_reav[-1]}
                    if act_d in legal_moves:
                        return act_d 
            # Check if play/discard the 1st card is legal
            if {'action_type': intent_name.upper(), 'card_index': 0} in legal_moves:
                return {'action_type': intent_name.upper(), 'card_index': 0}
            else:
                return self._get_random_move(legal_moves)
        elif "play_" in intent_name:
            act_d = {'action_type': 'PLAY' ,'card_index': int(intent_name[-1])}
            if act_d in legal_moves:
                return act_d
            else:
                return self._get_random_move(legal_moves)
        elif "discard_" in intent_name:
            act_d = {'action_type': 'DISCARD' ,'card_index': int(intent_name[-1])}
            if act_d in legal_moves:
                return act_d
            else:
                return self._get_random_move(legal_moves)
        # NO INTENT 
        elif intent_name == "no_intent":
            hint = self.best_hint_to_give(super()._extract_dict_from_backend(player, self.state.observation(player)))
            if hint is not None:
                return hint
            else:
                return self._get_random_move(legal_moves)

    def playable_card(self, card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card['rank'] == fireworks[card['color']]
        
    def best_hint_to_give(self, observation):
        """ Select the best hint to give"""
        if observation['information_tokens'] == 0:
            return None
        
        hint = None
        best_so_far = 0
        player_to_hint = None
        color_to_hint = None
        rank_to_hint = None
        
        fireworks = observation['fireworks']
        # Check if there are any playable cards in the hands of the opponents.
        for player_offset in range(1, observation['num_players']):
            player_hand = observation['observed_hands'][player_offset]
            player_hints = observation['card_knowledge'][player_offset]
            
            is_really_playable = []
            for card in player_hand :
                is_really_playable.append(True if self.playable_card(card, fireworks) else False)
            
            for color in fireworks.keys():
                informative_content = 0
                misinformative = False
                for c, card in enumerate(player_hand) :
                    if card['color'] != color:
                        continue
                    if is_really_playable[c] and player_hints[c]['color'] is None:
                        informative_content += 1
                    elif not is_really_playable[c]:
                        misinformative = True
                        break
                if misinformative:
                    continue
                if informative_content > best_so_far:
                    best_so_far = informative_content
                    color_to_hint = color
                    rank_to_hint = None
                    player_to_hint = player_offset
                    
            for rank in range(5):
                informative_content = 0
                misinformative = False
                for c, card in enumerate(player_hand) :
                    if card['rank'] != rank:
                        continue
                    if is_really_playable[c] and player_hints[c]['rank'] is None:
                        informative_content += 1
                    elif not is_really_playable[c]:
                        misinformative = True
                        break
                if misinformative:
                    continue
                if informative_content > best_so_far:
                    best_so_far = informative_content
                    color_to_hint = None
                    rank_to_hint = rank
                    player_to_hint = player_offset
            
        if best_so_far == 0:
            hint = None
        else:
            if rank_to_hint is not None:
                hint = {
                        'action_type': 'REVEAL_RANK',
                        'rank': rank_to_hint,
                        'target_offset': player_to_hint
                    }
            elif color_to_hint is not None :
                hint = {
                        'action_type': 'REVEAL_COLOR',
                        'color': color_to_hint,
                        'target_offset': player_to_hint
                    }
        return hint
    
        
    def _get_random_move(self, legal_moves):
        # Choose random action
        return random.choice(legal_moves)
    
    def _extract_dict_from_backend(self, player_id, observation):
        """Extract a dict of features from an observation from the backend.

        Args:
          player_id: Int, player from whose perspective we generate the observation.
          observation: A `pyhanabi.HanabiObservation` object.

        Returns:
          obs_dict: dict, mapping from HanabiObservation to a dict.
        """
        obs_dict = super(HanabiIntentEnv, self)._extract_dict_from_backend(player_id, observation)

        if len(self.intents[player_id]) == 0:
            return obs_dict
        else:
            bool_act = [False]*len(self.intents[player_id])
            valide_act = dict(zip(self.intents[player_id], bool_act))
            
            if self.intent_type[player_id] == 'pdn':
                if player_id == obs_dict['current_player']:
                    for i in self.intents[player_id][2:]:
                        valide_act[i] = True
                for move in observation.legal_moves():
                    legal_move_as_dict = move.to_dict()
                    if legal_move_as_dict['action_type'] is "PLAY":
                        valide_act["play"] = True
                    if legal_move_as_dict['action_type'] is "DISCARD":
                        valide_act["discard"] = True
            elif self.intent_type[player_id] == 'pdn_num':
                if player_id == obs_dict['current_player']:
                    valide_act['no_intent'] = True
                for move in observation.legal_moves():
                    legal_move_as_dict = move.to_dict()
                    if legal_move_as_dict['action_type'] is "PLAY":
                        valide_act["play_{}".format(legal_move_as_dict['card_index'])] = True
                    if legal_move_as_dict['action_type'] is "DISCARD":
                        valide_act["discard_{}".format(legal_move_as_dict['card_index'])] = True
            else:
                raise ValueError('intent_type key unknown : {}'.format(self.intent_type[player_id]))
                
            obs_dict["legal_moves"] = []
            obs_dict["legal_moves_as_int"] = []
            for ind, (key, valide) in enumerate(valide_act.items()):
                if valide:
                    obs_dict["legal_moves"].append(key)
                    obs_dict["legal_moves_as_int"].append(ind)
                        
            return obs_dict

class HanabiIntentMarlEnv(HanabiIntentEnv):
    
    def __init__(self, config={}):
        super(HanabiIntentMarlEnv, self).__init__(config)
    
    def reset(self):
        obs = HanabiIntentEnv.reset(self)
        return obs['player_observations']
    
    def step(self, action):
        current_player = self.state.cur_player()
        single_action = action[current_player]

        obs, reward, done, info = HanabiIntentEnv.step(self, single_action)
        
        rewrite_obs = obs['player_observations']
                
        return rewrite_obs, [reward]*self.players, [done]*self.players, info


class MarlWrapperEnv(object):
    
    def __init__(self, env):
        self.hanabi_env = env
    
    @property
    def state(self):
        return self.hanabi_env.state
    
    @property
    def game(self):
        return self.hanabi_env.game
    
    def num_moves(self, player_id=None):
        return self.hanabi_env.num_moves(player_id)
        
    def reset(self):
        obs = self.hanabi_env.reset()
        return obs['player_observations']
    
    def step(self, action):
        current_player = self.hanabi_env.state.cur_player()
        single_action = action[current_player]

        obs, reward, done, info = self.hanabi_env.step(single_action)
        
        rewrite_obs = obs['player_observations']
                
        return rewrite_obs, [reward]*self.hanabi_env.players, [done]*self.hanabi_env.players, info
    
    def render(self):
        print(self.hanabi_env.state.observation(self.hanabi_env.state.cur_player()))


def make(environment_name="Hanabi-Full", num_players=2, pyhanabi_path=None):
    """Make an environment.

    Args:
      environment_name: str, Name of the environment to instantiate.
      num_players: int, Number of players in this game.
      pyhanabi_path: str, absolute path to header files for c code linkage.

    Returns:
      env: An `Environment` object.

    Raises:
      ValueError: Unknown environment name.
    """

    import re
    if pyhanabi_path is not None:
        prefixes = (pyhanabi_path,)
        assert pyhanabi.try_cdef(prefixes=prefixes), "cdef failed to load"
        assert pyhanabi.try_load(prefixes=prefixes), "library failed to load"

    if (environment_name == "Hanabi-Full" or
            environment_name == "Hanabi-Full-CardKnowledge"):
        return HanabiEnv(
            config={
                "colors":
                    5,
                "ranks":
                    5,
                "players":
                    num_players,
                "max_information_tokens":
                    8,
                "max_life_tokens":
                    3,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
    elif "Hanabi-Intent" in environment_name:
        fields = re.split(r'-', environment_name)
        intent_type=fields[-num_players:]
        my_class = HanabiIntentMarlEnv if "Marl" in environment_name else HanabiIntentEnv
        if "Very-Small" in environment_name:
            return my_class(
            config={
                "colors":
                    1,
                "ranks":
                    5,
                "players":
                    num_players,
                "intent_type":
                    intent_type,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
        elif "Small" in environment_name:
            return my_class(
            config={
                "colors":
                    2,
                "ranks":
                    5,
                "players":
                    num_players,
                "intent_type":
                    intent_type,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
        else:
            return my_class(
                config={
                    "colors":
                        5,
                    "ranks":
                        5,
                    "players":
                        num_players,
                    "intent_type":
                        intent_type,
                    "max_information_tokens":
                        8,
                    "max_life_tokens":
                        3,
                    "observation_type":
                        pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
                })
            
    elif "Hanabi-Int2Act" in environment_name:
        fields = re.split(r'-', environment_name)
        intent_type=fields[-num_players:]
        my_class = HanabiInt2ActEnv#HanabiInt2ActMarlEnv if "Marl" in environment_name else HanabiInt2ActEnv
        if "Very-Small" in environment_name:
            env = my_class(
            config={
                "colors":
                    1,
                "ranks":
                    5,
                "players":
                    num_players,
                "intent_type":
                    intent_type,
                "pos":
                    1,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
        elif "Small" in environment_name:
            env = my_class(
            config={
                "colors":
                    2,
                "ranks":
                    5,
                "players":
                    num_players,
                "intent_type":
                    intent_type,
                "pos":
                    1,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
        else:
            env = my_class(
                config={
                    "colors":
                        5,
                    "ranks":
                        5,
                    "players":
                        num_players,
                    "intent_type":
                        intent_type,
                    "pos":
                        1,
                    "max_information_tokens":
                        8,
                    "max_life_tokens":
                        3,
                    "observation_type":
                        pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
                })
        if "Marl" in environment_name:
            env = MarlWrapperEnv(env)
        return env
    elif environment_name == "Hanabi-Full-Minimal":
        return HanabiEnv(
            config={
                "colors": 5,
                "ranks": 5,
                "players": num_players,
                "max_information_tokens": 8,
                "max_life_tokens": 3,
                "observation_type": pyhanabi.AgentObservationType.MINIMAL.value
            })
    elif environment_name == "Hanabi-Small":
        return HanabiEnv(
            config={
                "colors":
                    2,
                "ranks":
                    5,
                "players":
                    num_players,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
    elif environment_name == "Hanabi-Very-Small":
        return HanabiEnv(
            config={
                "colors":
                    1,
                "ranks":
                    5,
                "players":
                    num_players,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })

    else:
        raise ValueError("Unknown environment {}".format(environment_name))


# -------------------------------------------------------------------------------
# Hanabi Agent API
# -------------------------------------------------------------------------------


class Agent(object):
    """Agent interface.

    All concrete implementations of an Agent should derive from this interface
    and implement the method stubs.


    ```python

    class MyAgent(Agent):
      ...

    agents = [MyAgent(config) for _ in range(players)]
    while not done:
      ...
      for agent_id, agent in enumerate(agents):
        action = agent.act(observation)
        if obs.current_player == agent_id:
          assert action is not None
        else
          assert action is None
      ...
    ```
    """

    def __init__(self, config, *args, **kwargs):
        r"""Initialize the agent.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - colors: int, Number of colors \in [2,5].
              - ranks: int, Number of ranks \in [2,5].
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0)
              - max_life_tokens: int, Number of life tokens (>=0)
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
          *args: Optional arguments
          **kwargs: Optional keyword arguments.

        Raises:
          AgentError: Custom exceptions.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def reset(self, config):
        r"""Reset the agent with a new config.

        Signals agent to reset and restart using a config dict.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - colors: int, Number of colors \in [2,5].
              - ranks: int, Number of ranks \in [2,5].
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0)
              - max_life_tokens: int, Number of life tokens (>=0)
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def act(self, observation):
        """Act based on an observation.

        Args:
          observation: dict, containing observation from the view of this agent.
            An example:
            {'current_player': 0,
             'current_player_offset': 1,
             'deck_size': 40,
             'discard_pile': [],
             'fireworks': {'B': 0,
                       'G': 0,
                       'R': 0,
                       'W': 0,
                       'Y': 0},
             'information_tokens': 8,
             'legal_moves': [],
             'life_tokens': 3,
             'observed_hands': [[{'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1}],
                            [{'color': 'W', 'rank': 2},
                             {'color': 'Y', 'rank': 4},
                             {'color': 'Y', 'rank': 2},
                             {'color': 'G', 'rank': 0},
                             {'color': 'W', 'rank': 1}]],
             'num_players': 2}]}

        Returns:
          action: dict, mapping to a legal action taken by this agent. The following
            actions are supported:
              - { 'action_type': 'PLAY', 'card_index': int }
              - { 'action_type': 'DISCARD', 'card_index': int }
              - {
                  'action_type': 'REVEAL_COLOR',
                  'color': str,
                  'target_offset': int >=0
                }
              - {
                  'action_type': 'REVEAL_RANK',
                  'rank': str,
                  'target_offset': int >=0
                }
        """
        raise NotImplementedError("Not implemented in Abstract Base class")
