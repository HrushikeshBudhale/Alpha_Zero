import numpy as np

class MCTS_Node:
    def __init__(self, game, state, parent=None, action_taken=None, args=None):
        self.game = game
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.args = args
        
        self.children = []
        self.expandable_moves = game.get_valid_actions(state)
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0 and np.sum(self.expandable_moves) == 0
    
    def select_child(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child
    
    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args["C"] * np.sqrt(np.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        
        next_state = self.game.get_next_state(self.state, action, 1)
        child_state = self.game.get_perspective(next_state, player=-1)
        child = MCTS_Node(self.game, child_state, self, action, self.args)
        self.children.append(child)
        return child
    
    def simulate(self):
        state = self.state.copy()
        action = self.action_taken
        player = 1
        while True:
            value, is_terminated = self.game.get_value_and_terminated(state, action)
            if is_terminated:
                break
            valid_actions = self.game.get_valid_actions(state)
            action = np.random.choice(np.where(valid_actions == 1)[0])
            state = self.game.get_next_state(state, action, player)
            player = self.game.get_opponent(player)
        return self.game.get_opponent_value(value) if player == 1 else value
        
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args=None):
        self.game = game
        self.args = args
        
    def search(self, state):
        root = MCTS_Node(self.game, state, args=self.args)
        
        for _ in range(self.args["num_searches"]):
            node = root
            
            while node.is_fully_expanded():
                node = node.select_child()
            
            value, is_terminated = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            if not is_terminated:
                node = node.expand()
                value = node.simulate()
            node.backpropagate(value)
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs = action_probs / np.sum(action_probs)
        return action_probs
