from alpha_zero import AlphaZero

from environment.TicTacToe import TicTacToe
from environment.ConnectFour import ConnectFour
from model import ResNet
from path import Path
import torch
import yaml
torch.set_printoptions(sci_mode=False, precision=1)


if __name__ == "__main__":
    config_path = Path("config/training_conf.yaml")
    config = yaml.safe_load(open(config_path, "r"))
    
    args = config["TicTacToe"]
    game = TicTacToe()
    # args = config["ConnectFour"]
    # game = ConnectFour()
    
    model = ResNet(game, 4, 64, args["device"])
    optim = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"],)
    
    alphaZero = AlphaZero(game, model, optim, args)
    alphaZero.learn()