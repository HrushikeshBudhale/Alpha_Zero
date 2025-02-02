import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, N_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(N_channels, N_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(N_channels)
        self.conv2 = nn.Conv2d(N_channels, N_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(N_channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, game, N_resBlocks, N_channels, device='cpu'):
        super(ResNet, self).__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, N_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(N_channels),
            nn.ReLU()
        )   
        
        self.backBone = nn.ModuleList(
            [ResBlock(N_channels) for _ in range(N_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(N_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rows * game.cols, game.action_size),
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(N_channels, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.rows * game.cols, 1),
            nn.Tanh()
        )
        self.to(device)
   
   
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value