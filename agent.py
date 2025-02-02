import numpy as np
import torch
from mcts import MCTS
from model import ResNet
from alpha_zero import AlphaZero
from typing import final

class BaseAgent:
    obj_count = 1
    def __init__(self, game, args=None):
        self.object_id = BaseAgent.obj_count
        self.game = game
        self.wins = 0
        self.args = args
        BaseAgent.obj_count += 1

    def get_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    @final
    def __repr__(self):
        return f"{self.__class__.__name__}_{self.object_id}"


class RandomAgent(BaseAgent):
    def __init__(self, game, args):
        super().__init__(game)
    
    def get_action(self, state: np.ndarray) -> int:
        valid_actions = self.game.get_valid_actions(state)
        action = np.random.choice(np.where(valid_actions == 1)[0])
        return action
    

class HumanAgent(BaseAgent):
    def __init__(self, game, args):
        super().__init__(game)
    
    def get_action(self, state: np.ndarray) -> int:
        valid_actions = self.game.get_valid_actions(state)
        while True:
            try:
                readable_actions = [i for i in range(len(valid_actions)) if valid_actions[i] == 1]
                action = int(input(f"Valid actions: \n{readable_actions}, takes action: "))
                if valid_actions[action] == 1:
                    break
            except:
                print("Invalid input")
        return action


class MCTSAgent(BaseAgent):
    def __init__(self, game, args):
        super().__init__(game, args)
        self.mcts = MCTS(game, args)
        
    def get_action(self, state: np.ndarray) -> int:
        action_probs = self.mcts.search(state)
        action = np.argmax(action_probs)
        return action
    

class AlphaZeroAgent(BaseAgent):
    def __init__(self, game, args):
        super().__init__(game, args)
        
        self.model = ResNet(game, 9, 128, args["device"])
        self.model.load_state_dict(torch.load(args["model_path"], map_location=args["device"], weights_only=True))
        self.model.eval()
        self.alpha_zero = AlphaZero(game, self.model, optim=None, args=args)
        self.current_confidence = 0
        
    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> int:
        return self.alpha_zero.play(state)
    

AGENT_TYPES = {
    "random": RandomAgent,
    "human": HumanAgent,
    "mcts": MCTSAgent,
    "alpha_zero": AlphaZeroAgent
}