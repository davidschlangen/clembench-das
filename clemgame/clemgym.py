from __future__ import annotations

import collections
from typing import List, Dict

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID, ObsType, ActionType

from backends import Model
from clemgame.clemgame import Player, GameMaster, DialogueGameMaster


class GymMaster(GameMaster, AECEnv):
    """    Connects backends.Model and GameRecorder with AECEnv    """

    def __init__(self, name: str, experiment: Dict, player_models: List[Model] = None):
        super().__init__(name)
        self.experiment: Dict = experiment
        self.player_models: List[Model] = player_models

    def play(self) -> None:
        turn_idx = 0
        termination = False
        while not termination:
            for player_descriptor in self.agent_iter():
                observation, reward, termination, truncation, info = self.last()
                if termination or truncation:
                    action = None
                else:
                    player = self.get_player(player_descriptor)
                    action = player(observation, turn_idx)
                self.step(action)
            turn_idx += 1
        self.close()

    def get_player(self, player_descriptor: str) -> Player:
        raise NotImplementedError

    def setup(self, **kwargs):
        self.reset(seed=42, options=kwargs)

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """Resets the environment to a starting state."""
        raise NotImplementedError

    def observe(self, agent: AgentID) -> ObsType | None:
        """Returns the observation an agent currently can make.

        `last()` calls this function.
        """
        raise NotImplementedError

    def step(self, action: ActionType) -> None:
        """Accepts and executes the action of the current agent_selection in the environment.

        Automatically switches control to the next agent.
        """
        raise NotImplementedError

    def render(self) -> None | np.ndarray | str | list:
        """Renders the environment as specified by self.render_mode.

        Render mode can be `human` to display a window.
        Other render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside of classic,
        and `'ansi'` which returns the strings printed (specific to classic environments).
        """
        raise NotImplementedError

    def state(self) -> np.ndarray:
        """State returns a global view of the environment.

        It is appropriate for centralized training decentralized execution methods like QMIX
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )


class DialogueGymMaster(DialogueGameMaster, AECEnv):

    def __init__(self, name: str, experiment: dict, player_models: List[Model]):
        super().__init__(name, experiment, player_models)
        # the logging works with an internal mapping of "Player N" -> Player
        self.players_by_names: Dict[str, Player] = collections.OrderedDict()
        self.messages_by_names: Dict[str, List] = dict()
        self.current_turn: int = 0
        self.is_reprompt = False
        self._agent_selector = None

    def get_player(self, player_descriptor: str) -> Player:
        return self.players_by_names[player_descriptor]

    def setup(self, **kwargs):
        self.reset(seed=42, options=kwargs)

    def _on_setup(self, **kwargs):
        """
        Template method: must be implemented

        Use add_player() here to add the players.

        :param kwargs: of the game instance
        """
        raise NotImplementedError()

    def reset(self, seed: int | None = None, options: dict | None = None, ) -> None:
        self._on_setup(**options)
        # log players
        players_descriptions = collections.OrderedDict(GM=f"Game master for {self.name}")
        for name, player in self.players_by_names.items():
            players_descriptions[name] = player.get_description()
        self.log_players(players_descriptions)
        # gym attributes
        self.agents = list(self.players_by_names.keys())
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = dict(zip(self.agents, [0.0] * len(self.agents)))
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = dict(zip(self.agents, [False] * len(self.agents)))
        self.truncations = dict(zip(self.agents, [False] * len(self.agents)))
        self.infos = dict(zip(self.agents, [{}] * len(self.agents)))

    def step(self, action: ActionType) -> None:
        # GM -> GM
        player = self.players_by_names[self.agent_selection]
        self._validate_parse_and_add_player_response(player, action)
        self.terminations = dict.fromkeys(self.terminations, not self._does_game_proceed())  # applies to all players

        if self._should_reprompt(player):  # player has additional actions
            self._on_before_reprompt(player)
            self.is_reprompt = True
        else:
            self.is_reprompt = False
            self.agent_selection = self._agent_selector.next()

    def observe(self, agent: AgentID) -> ObsType | None:
        # GM -> Player
        history = self.messages_by_names[agent]
        assert history, f"messages history must not be empty for {agent}"

        last_entry = history[-1]
        assert last_entry["role"] != "assistant", "Last entry should not be assistant " \
                                                  "b.c. this would be the role of the current player"
        return history

    def play(self) -> None:
        self._on_before_game()
        termination = False
        while not termination:
            self.log_next_turn()  # not sure if we want to do this always here (or add to _on_before_turn)
            self._on_before_turn(self.current_turn)
            self.logger.info(f"{self.name}: %s turn: %d", self.name, self.current_turn)

            for player_name in self.agent_iter():
                observation, _, _, _, info = self.last()
                action = self.prompt(player_name, observation)
                self.step(action)

                # check for game end after player turn
                _, reward, termination, truncation, info = self.last(observe=False)
                if termination:
                    break

            self._on_after_turn(self.current_turn)
            self.current_turn += 1
        self._on_after_game()

    def prompt(self, player_name: str, history: List[Dict]) -> str:

        last_message = history[-1]["content"]
        action_type = 'send message' if not self.is_reprompt else 'send message (reprompt)'
        action = {'type': action_type, 'content': last_message}
        self.log_event(from_='GM', to=player_name, action=action)

        player = self.get_player(player_name)
        _prompt, _response, response_message = player(history, self.current_turn)

        # Player -> GM
        action = {'type': 'get message', 'content': response_message}
        self.log_event(from_=player_name, to="GM", action=action, call=(_prompt, _response))

        return response_message
