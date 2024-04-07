import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp




class GCN(torch.nn.Module):
    def __init__(self, embedding_size=64, num_features=9):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index, edge_weight=None):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index, edge_weight)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index, edge_weight)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index, edge_weight)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index, edge_weight)
        hidden = F.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out