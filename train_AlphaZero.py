from alpha_mcts import AlphaZero
from environment.TicTacToe import TicTacToe
from model import ResNet
import torch
torch.set_printoptions(sci_mode=False)

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

game = TicTacToe()

args = {
    "game_name": str(game),
    "C": 1.41,
    "temperature": 1.25,
    "learn_iters": 10,
    "selfplay_iters": 50, 
    "train_iters": 10, 
    "num_searches": 600, 
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "device": device,
    "model_path": "model.pt",
    "optimizer_path": "optimizer.pt"}

model = ResNet(game, 4, 64, device)
optim = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"],)
alphaZero = AlphaZero(game, model, optim, args)
alphaZero.learn()



# # Test trained model
# model = ResNet(game, 4, 64, device)
# model.load_state_dict(torch.load("model_0.pt", weights_only=True, map_location=device))
# model.eval()
# state = game.get_initial_state()
# state = game.get_next_state(state, 2, 1)
# state = game.get_next_state(state, 3, -1)
# state = game.get_next_state(state, 6, 1)
# state = game.get_next_state(state, 8, -1)

# neutral_state = game.get_perspective(state, -1)
# print(neutral_state)
# encoded_state = torch.tensor(game.get_encoded_state(neutral_state)).unsqueeze(0).to(device)
# policy, value = model(encoded_state)
# print(policy, value)
