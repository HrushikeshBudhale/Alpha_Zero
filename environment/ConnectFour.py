import numpy as np

class ConnectFour:
    def __init__(self, rows=6, cols=7, in_a_row=4):
        self.rows, self.cols = rows, cols
        self.in_a_row = in_a_row
        self.action_size = cols
        
    def __repr__(self):
        return f"ConnectFour(rows={self.rows}, cols={self.cols}, in_a_row={self.in_a_row})"
        
    def get_initial_state(self):
        return np.zeros((self.rows, self.cols), dtype=int)

    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        next_state = state.copy()
        next_state[row, action] = player
        return next_state
    
    def get_valid_actions(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        row = np.min(np.where(state[:, action] != 0))
        col = action
        player = state[row, col]
        
        def count(row_offset, col_offset):
            for i in range(1, self.in_a_row):
                r = row + i * row_offset
                c = col + i * col_offset
                if (
                    r < 0 or r >= self.rows
                    or c < 0 or c >= self.cols
                    or state[r, c] != player
                ):
                    return i - 1
            return self.in_a_row - 1
        
        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0,-1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1,-1)) >= self.in_a_row - 1 # diagonal
            or (count(-1, 1) + count(1,-1)) >= self.in_a_row - 1 # anti-diagonal
        )
        
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_actions(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    # useful for Alpha-Zero
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
    game = ConnectFour()
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
    # Valid actions: [1 1 1 1 1 1 1], player: -1, takes action: 6
    # [[ 0  0  0  0  0  0  0]
    # [ 0  0  1  0  0  0  0]
    # [ 0  0 -1  0  0  1 -1]
    # [ 0  0  1  0 -1  1 -1]
    # [ 1 -1  1 -1  1 -1 -1]
    # [ 1 -1  1 -1 -1  1  1]]
    # Valid actions: [1 1 1 1 1 1 1], player: 1, takes action: 1
    # [[ 0  0  0  0  0  0  0]
    # [ 0  0  1  0  0  0  0]
    # [ 0  0 -1  0  0  1 -1]
    # [ 0  1  1  0 -1  1 -1]
    # [ 1 -1  1 -1  1 -1 -1]
    # [ 1 -1  1 -1 -1  1  1]]
    # Valid actions: [1 1 1 1 1 1 1], player: -1, takes action: 0
    # [[ 0  0  0  0  0  0  0]
    # [ 0  0  1  0  0  0  0]
    # [ 0  0 -1  0  0  1 -1]
    # [-1  1  1  0 -1  1 -1]
    # [ 1 -1  1 -1  1 -1 -1]
    # [ 1 -1  1 -1 -1  1  1]]
    # Valid actions: [1 1 1 1 1 1 1], player: 1, takes action: 3
    # [[ 0  0  0  0  0  0  0]
    # [ 0  0  1  0  0  0  0]
    # [ 0  0 -1  0  0  1 -1]
    # [-1  1  1  1 -1  1 -1]
    # [ 1 -1  1 -1  1 -1 -1]
    # [ 1 -1  1 -1 -1  1  1]]
    # Valid actions: [1 1 1 1 1 1 1], player: -1, takes action: 6
    # [[ 0  0  0  0  0  0  0]
    # [ 0  0  1  0  0  0 -1]
    # [ 0  0 -1  0  0  1 -1]
    # [-1  1  1  1 -1  1 -1]
    # [ 1 -1  1 -1  1 -1 -1]
    # [ 1 -1  1 -1 -1  1  1]]
    # Player -1 wins!
    # Game Over
    
if __name__ == "__main__":
    main()