import numpy as np
import torch
from mcts import MCTS
from model import ResNet
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
                action = int(input(f"Valid actions: {valid_actions}, takes action: "))
                if valid_actions[action] == 1:
                    break
            except ValueError:
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
    

class AlphaMCTSAgent(BaseAgent):
    def __init__(self, game, args):
        super().__init__(game, args)
        
        self.model = ResNet(game, 4, 64, args["device"])
        self.model.load_state_dict(torch.load(args["model_path"], map_location=args["device"]))
        self.model.eval()
        self.temperature = args["temperature"] if "temperature" in args else 1
        self.current_confidence = 0
        
    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> int:
        encoded_state = torch.tensor(self.game.get_encoded_state(state), dtype=torch.float32, device=self.args["device"]).unsqueeze(0)
        action_probs, value = self.model(encoded_state)
        self.current_confidence = value.item()
        action_probs = action_probs.squeeze(0).cpu().numpy()
        valid_actions = self.game.get_valid_actions(state)
        action_probs = action_probs * valid_actions
        action_probs = action_probs / action_probs.sum()
        if self.temperature >= 5:
            action = np.random.choice([c for c in range(self.game.action_size) if action_probs[c] > 0])
        elif self.temperature <= 0:
            action = np.argmax(action_probs)
        else:
            action_probs = np.power(action_probs, 1/self.temperature)
            action_probs /= action_probs.sum()
            action = np.random.choice(self.game.action_size, p=action_probs)
        return action
    

AGENT_TYPES = {
    "random": RandomAgent,
    "human": HumanAgent,
    "mcts": MCTSAgent,
    "alphamcts": AlphaMCTSAgent
}