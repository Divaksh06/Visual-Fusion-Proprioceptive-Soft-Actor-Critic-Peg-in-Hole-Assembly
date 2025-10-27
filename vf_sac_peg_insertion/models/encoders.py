"""Vision and Force Encoders"""

import torch
import torch.nn as nn
import torchvision.models as models

class VisionEncoder(nn.Module):
    """RGB-D Vision Encoder for single eye-in-hand camera"""
    def __init__(self, feature_dim=128, pretrained=True):
        super().__init__()
        
        resnet = models.resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data[:, :3] = resnet.conv1.weight.data
            self.conv1.weight.data[:, 3] = resnet.conv1.weight.data.mean(dim=1)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: [batch, 3, 256, 256]
            depth: [batch, 1, 256, 256]
        Returns:
            features: [batch, feature_dim]
        """
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        
        x = torch.cat([rgb, depth], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        features = self.fc(x)
        
        return features


class ForceEncoder(nn.Module):
    """Force-Torque Encoder with temporal sliding window"""
    def __init__(self, feature_dim=128, history_length=5):
        super().__init__()
        self.history_length = history_length
        input_dim = 6 * history_length
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, feature_dim)
        )
    
    def forward(self, force_torque_history):
        """
        Args:
            force_torque_history: [batch, 30]
        Returns:
            features: [batch, feature_dim]
        """
        normalized = torch.tanh(force_torque_history / 10.0)
        features = self.network(normalized)
        return features
