from typing import List, Dict

from backends import Model
from clemgame import get_logger
from clemgame.clemgame import GameBenchmark, GameMaster, GameScorer
from clemgame.clemgym import DialogueGymMaster
from games.taboo.master import Taboo, TabooScorer

GAME_NAME = "taboo-gym"

logger = get_logger(__name__)


class TabooGym(Taboo, DialogueGymMaster):

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(experiment, player_models)
        self.name = GAME_NAME


class TabooGymBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Taboo (Gym) game between two agents where one has to describe a word for the other to guess."

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return TabooGym(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return TabooScorer(experiment, game_instance)
