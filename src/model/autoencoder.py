import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PhysicsAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=2, hidden_layers=(32, 16), dropout=0.1):
        super().__init__()
        hidden_first, hidden_second = hidden_layers

        # Encoder
        self.conv1 = GCNConv(input_dim, hidden_first)
        self.conv2 = GCNConv(hidden_first, hidden_second)
        self.bottleneck = nn.Linear(hidden_second, latent_dim)
        self.dropout = dropout
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_second),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_second, hidden_first),
            nn.ReLU(),
            nn.Linear(hidden_first, input_dim)
        )

    def forward(self, data):
        h = F.relu(self.conv1(data.x, data.edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, data.edge_index))
        z = self.bottleneck(h)
        return self.dec(z)