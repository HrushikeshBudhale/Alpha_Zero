import argparse
from environment.ConnectFour import ConnectFour
from environment.TicTacToe import TicTacToe
from agent import AGENT_TYPES
import yaml

def create_player(player_type, game, args):
    return AGENT_TYPES[player_type](game, args)

def main(args):
    game = ConnectFour() if args.game else TicTacToe()
    config = yaml.safe_load(open(args.config, "r"))
    
    p1_args, p2_args = config["player_1"], config["player_2"]
    player1 = create_player(p1_args["player_type"], game, p1_args)
    player2 = create_player(p2_args["player_type"], game, p2_args)
    players = {1: player1, -1: player2}
    
    print(f"Starting game of {game}")
    print(f"Players: {player1} vs {player2}")
    print("============ Game Start ============")
    turn = 1
    state = game.get_initial_state()
    while True:
        print(state)
        action = players[turn].get_action(state)
        print(f"{players[turn]} takes action: {action}")
        state = game.get_next_state(state, action, turn)
        value, is_terminated = game.get_value_and_terminated(state, action)
        if is_terminated:
            break
        turn = game.get_opponent(turn)
    
    print(state)
    if value == 1:
        print(f"{players[turn]} wins the game!")
    else:
        print("Draw!")
    print("============ Game Over ============")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=bool, default=False, choices=[0,1], help="0: TicTacToe, 1: ConnectFour")
    parser.add_argument("--config", type=str, default="config/players.yaml", help="Path to config file containing player configurations")
    args = parser.parse_args()
    main(args)
        