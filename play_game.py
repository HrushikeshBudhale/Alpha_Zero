import argparse
from environment.ConnectFour import ConnectFour
from environment.TicTacToe import TicTacToe
from agent import AGENT_TYPES
from tqdm import tqdm
import yaml

class Tournament:
    def __init__(self, config_path: str):
        config = yaml.safe_load(open(config_path, "r"))
        self.N_rounds = config["tournament"]["num_rounds"]
        self.show_games = config["tournament"]["show_games"]
        self.game = self.create_game(config["game"])
        p1 = self.create_player(config["player_1"])
        p2 = self.create_player(config["player_2"])
        self.players = {1: p1, -1: p2}
    
    def create_player(self, player_settings: dict):
        player_type = player_settings["type"]
        self.show_games = player_type == "human"
        return AGENT_TYPES[player_type](self.game, player_settings)
    
    def create_game(self, game_settings: dict):
        game_type = game_settings["type"]
        if game_type == "TicTacToe":
            return TicTacToe(n=game_settings["rows"])
        elif game_type == "ConnectFour":
            rows = game_settings["rows"]
            cols = game_settings["cols"]
            in_a_row = game_settings["in_a_row"]
            return ConnectFour(rows, cols, in_a_row)
    
    def star_playing(self):
        print(f"Starting game of {self.game} with {self.N_rounds} rounds")
        for i in tqdm(range(self.N_rounds)):
            self.log(f"============ Round {i+1}/{self.N_rounds} ============")
            self.log(f"Players: {self.players[1]} vs {self.players[-1]}")
            if self.N_rounds > 1 and i > (self.N_rounds // 2):
                turn = -1
            else:
                turn = 1
            
            state = self.game.get_initial_state()
            while True:
                self.log(state)
                action = self.players[turn].get_action(state)
                self.log(f"{self.players[turn]} takes action: {action}")
                state = self.game.get_next_state(state, action, turn)
                value, is_terminated = self.game.get_value_and_terminated(state, action)
                if is_terminated:
                    break
                turn = self.game.get_opponent(turn)
                
            self.log(state)
            if value == 1:
                self.log(f"{self.players[turn]} wins the game!")
                self.players[turn].wins += 1
            else:
                self.log("Draw!")
            self.log("============ Game Over ============")
                
    def log(self, s: str):
        if self.show_games:
            print(s)
    
    def show_results(self):
        print("============ Results ============")
        print(f"{self.players[1]} \t wins: {self.players[1].wins}")
        print(f"{self.players[-1]} \t wins: {self.players[-1].wins}")
        print(f"Draw: {self.N_rounds - self.players[1].wins - self.players[-1].wins}")


def main(args):
    tournament = Tournament(args.config)
    tournament.star_playing()
    tournament.show_results()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/players.yaml", help="Path to config file containing player configurations")
    args = parser.parse_args()
    main(args)
        