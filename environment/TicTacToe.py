import numpy as np
from typing import Tuple

class TicTacToe:
    def __init__(self, n=3):
        self.rows = self.cols = n
        self.action_size = n * n
        
    def __repr__(self):
        return f"TicTacToe {self.rows}x{self.cols}"
    
    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.rows, self.cols), dtype=int)
    
    def get_valid_actions(self, state) -> np.ndarray:
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def get_next_state(self, state, action, player) -> np.ndarray:
        row = action // self.rows
        col = action % self.cols
        new_state = state.copy()
        new_state[row, col] = player
        return new_state
    
    def check_win(self, state, action) -> bool:
        if action == None:
            return False
        row = action // self.rows
        col = action % self.cols
        player = state[row, col]
        return (
            state[row, :] == player).all() \
        or (state[:, col] == player).all() \
        or (state.diagonal() == player).all() \
        or (state[::-1].diagonal() == player).all()
    
    def get_value_and_terminated(self, state, action) -> Tuple[float, bool]:
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_actions(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    # Functions for MCTS and Alpha-Zero
    def get_opponent_value(self, value):
        return -value
    
    def get_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(np.float32)
        if len(state.shape) == 3: # for batch processing
            encoded_state = encoded_state.transpose(1, 0, 2, 3) # (batch_size, channels, rows, cols)
        return encoded_state
    
    
def main():
    # Test playing a game with random actions
    game = TicTacToe()
    state = game.get_initial_state()
    player = 1
    while True:
        actions = game.get_valid_actions(state)
        action = np.random.choice(np.where(actions == 1)[0])
        print(f"Valid actions: {actions}, player: {player}, takes action: {action}")
        state = game.get_next_state(state, action, player)
        print(state)
        if game.check_win(state, action):
            print(f"Player {player} wins!")
            break
        player = game.get_opponent(player)
    print("Game Over")
    
    # Example output
    # Valid actions: [1 1 1 1 1 1 1 1 1], player: 1, takes action: 2
    # [[0 0 1]
    # [0 0 0]
    # [0 0 0]]
    # Valid actions: [1 1 0 1 1 1 1 1 1], player: -1, takes action: 3
    # [[ 0  0  1]
    # [-1  0  0]
    # [ 0  0  0]]
    # Valid actions: [1 1 0 0 1 1 1 1 1], player: 1, takes action: 4
    # [[ 0  0  1]
    # [-1  1  0]
    # [ 0  0  0]]
    # Valid actions: [1 1 0 0 0 1 1 1 1], player: -1, takes action: 5
    # [[ 0  0  1]
    # [-1  1 -1]
    # [ 0  0  0]]
    # Valid actions: [1 1 0 0 0 0 1 1 1], player: 1, takes action: 1
    # [[ 0  1  1]
    # [-1  1 -1]
    # [ 0  0  0]]
    # Valid actions: [1 0 0 0 0 0 1 1 1], player: -1, takes action: 8
    # [[ 0  1  1]
    # [-1  1 -1]
    # [ 0  0 -1]]
    # Valid actions: [1 0 0 0 0 0 1 1 0], player: 1, takes action: 0
    # [[ 1  1  1]
    # [-1  1 -1]
    # [ 0  0 -1]]
    # Player 1 wins!
    
if __name__ == "__main__":
    main()