import torch
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
from model import ResNet

class AlphaMCTS_Node:
    def __init__(self, game, state, parent=None, action_taken=None, prior=0, args=None):
        self.game = game
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior # probability of this node
        self.args = args
        
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args["C"] * (np.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                next_state = self.game.get_next_state(self.state, action, 1)
                child_state = self.game.get_perspective(next_state, player=-1)
                child = AlphaMCTS_Node(self.game, child_state, self, action, prior=prob, args=self.args)
                self.children.append(child)
        return child
    
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value)


class AlphaMCTS:
    def __init__(self, game, model, args=None):
        self.game = game
        self.model = model
        self.args = args
        self.device = args["device"]
    
    @torch.no_grad()
    def search(self, state):
        root = AlphaMCTS_Node(self.game, state, args=self.args)
        
        for _ in range(self.args["num_searches"]):
            node = root
            
            while node.is_fully_expanded():
                node = node.select_child()
            
            value, is_terminated = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            if not is_terminated:
                valid_actions = self.game.get_valid_actions(node.state)
                state = torch.tensor(self.game.get_encoded_state(node.state), device=self.device).unsqueeze(0)
                policy, value = self.model(state)
                policy = policy.squeeze(0).cpu().numpy()
                policy *= valid_actions
                policy /= np.sum(policy)
                value = value.item()
                node = node.expand(policy)
            
            node.backpropagate(value)
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs = action_probs / np.sum(action_probs)
        return action_probs

class AlphaZero:
    def __init__(self, game, model, optim, args=None):
        self.game = game
        self.model = model
        self.optimizer = optim
        self.args = args
        self.mcts = AlphaMCTS(game, model, args)
        self.device = args["device"]
        
    def selfPlay(self):
        memory = []
        state = self.game.get_initial_state()
        player = 1
        
        while True:
            neutral_state = self.game.get_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            memory.append((neutral_state, action_probs, player))
            temperature_action_probs = np.power(action_probs, 1/self.args["temperature"])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(len(action_probs), p=temperature_action_probs)
            state = self.game.get_next_state(neutral_state, action, player)
            value, is_terminated = self.game.get_value_and_terminated(state, action)
            if is_terminated:
                valueMemory = []
                for mem_neutral_state, mem_policy, mem_player in memory:
                    value_wrt_player = value if player == mem_player else self.game.get_opponent_value(value)
                    valueMemory.append((mem_neutral_state, mem_policy, value_wrt_player))
                return valueMemory
            player = self.game.get_opponent(player)
    
    def train(self, memory):
        random.shuffle(memory)
        batch_size = self.args["batch_size"]
        for i in range(0, len(memory), batch_size):
            batch = memory[i:i+batch_size]
            states, policy_targets, value_targets = zip(*batch)
            states, policy_targets, value_targets = np.array(states), np.array(policy_targets), np.array(value_targets).reshape(-1,1)
            encoded_states = self.game.get_encoded_state(states)
            encoded_states = torch.tensor(encoded_states, dtype=torch.float32, device=self.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.device)
            
            policy, value = self.model(encoded_states)
            value_loss = F.mse_loss(value, value_targets)
            policy_loss = F.cross_entropy(policy, policy_targets)
            loss = value_loss + policy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"val loss: {value_loss.item()}, policy loss: {policy_loss.item()}")
        return

    def learn(self):
        N = self.args["learn_iters"]
        for i in range(N):
            print(f"Learning Iteration {i+1}/{N}")
            memory = []
            
            self.model.eval()
            for _ in tqdm(range(self.args["selfplay_iters"]), desc=f"Self Play"):
                memory += self.selfPlay()
                
            self.model.train()
            for _ in tqdm(range(self.args["train_iters"]), desc=f"Training"):
                self.train(memory)
        
        print("Saving model...")
        torch.save(self.model.state_dict(), self.args["model_path"])
        torch.save(self.optimizer.state_dict(), self.args["optimizer_path"])    