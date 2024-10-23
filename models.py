import torch
from torch_geometric.nn import GraphConv, SAGEConv, GATConv, FiLMConv, TransformerConv
from torch.nn import functional as F, Dropout, Linear


class GraphGCN(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GraphConv((-1, -1), dim_h)
        self.conv2 = GraphConv(dim_h, dim_h)
        self.linear = Linear(dim_h, 1)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict)
        x_dict = F.leaky_relu(x_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = F.leaky_relu(x_dict)
        x_dict = self.linear(x_dict)
        return x_dict


class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), dim_h, normalize=True)
        self.conv2 = SAGEConv(dim_h, dim_h)
        self.linear = Linear(dim_h, 1)
        self.dropout = Dropout(0.2)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.dropout(x_dict)
        x_dict = F.leaky_relu(x_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = F.leaky_relu(x_dict)
        x_dict = self.linear(x_dict)
        return x_dict


class GraphGAT(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GATConv((-1, -1), dim_h, heads=4, dropout=0.1, add_self_loops=False)
        self.conv2 = GATConv(dim_h * 4, dim_h, dropout=0.1, add_self_loops=False)
        self.linear = Linear(dim_h, 1)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        h = self.conv1(x_dict, edge_index_dict)
        h = F.leaky_relu(h)
        h = self.conv2(h, edge_index_dict)
        h = F.leaky_relu(h)
        h = self.linear(h)
        return h


class GraphFiLM(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = FiLMConv((-1, -1), dim_h)
        self.conv2 = FiLMConv(dim_h, dim_h)
        self.linear = Linear(dim_h, 1)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        h = self.conv1(x_dict, edge_index_dict)
        h = F.leaky_relu(h)
        h = self.conv2(h, edge_index_dict)
        h = F.leaky_relu(h)
        h = self.linear(h)
        return h


class GraphTransformer(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = TransformerConv((-1, -1), dim_h, heads=4, dropout=0.2)
        self.conv2 = TransformerConv(dim_h * 4, dim_h)
        self.linear = Linear(dim_h, 1)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        h = self.conv1(x_dict, edge_index_dict)
        h = F.leaky_relu(h)
        h = self.conv2(h, edge_index_dict)
        h = F.leaky_relu(h)
        h = self.linear(h)
        return h
