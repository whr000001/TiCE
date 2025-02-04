import torch
import torch.nn as nn
from torch_geometric.nn.conv import RGCNConv
# from torch_geometric.nn.pool import SAGPooling
# from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as func
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
# from torch_geometric.utils import batched_negative_sampling
# from torch_scatter import scatter
from torch_geometric.utils import degree


def split_graph(data, edge_score, ratio):
    def sparse_sort(src, index, dim=0, descending=False, eps=1e-12):
        f_min = torch.min(src)
        f_max = torch.max(src)
        norm = (src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
        perm = norm.argsort(dim=dim, descending=descending)
        return src[perm], perm

    def sparse_topk(src, index, dim=0, descending=False, eps=1e-12):
        rank, perm = sparse_sort(src, index, dim, descending, eps)
        num_nodes = degree(index, dtype=torch.long)
        # print(num_nodes)
        k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
        start_indices = torch.cat([torch.zeros((1,), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
        mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in
                range(len(num_nodes))]
        mask = torch.cat(mask, dim=0)
        mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
        topk_perm = perm[mask]
        exc_perm = perm[~mask]
        # print(topk_perm[:100])

        return topk_perm, exc_perm, rank, perm, mask
    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], descending=True)
    return new_idx_reserve, new_idx_drop


class Generator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.activation = func.leaky_relu
        self.edge_att = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 4),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim * 4, 1))

    def forward(self, data):
        h = self.activation(self.encoder(data.x))
        row, col = data.edge_index
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        edge_score = self.edge_att(edge_rep).view(-1)
        learned_mask = split_graph(data, edge_score, 0.25)

        return learned_mask


class RGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations):
        super().__init__()
        self.num_unit = 2
        self.linear_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.convs = nn.ModuleList()
        for i in range(self.num_unit):
            conv = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            self.convs.append(conv)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x = data.x
        x = self.linear_in(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index, data.edge_type)
            if i != len(self.convs) - 1:
                x = self.dropout(x)
        # x = global_mean_pool(x, data.batch)  # introduce too much noise in fact
        x = x[data.ptr[:-1]]
        return x


def graph_sample(data, mask):
    res = data.clone()
    edge_index = res.edge_index
    edge_type = res.edge_type
    edge_index = edge_index[:, mask]
    edge_type = edge_type[mask]
    col, row = edge_index
    edge_index = torch.cat([edge_index, torch.stack([row, col])], dim=-1)
    edge_type = torch.cat([edge_type, edge_type], dim=-1)
    edges = torch.cat([edge_index, edge_type.unsqueeze(0)], dim=0)
    edges = torch.unique(edges, dim=-1)
    edge_index, edge_type = torch.split(edges, [2, 1])
    edge_type = edge_type.squeeze(0)
    res.edge_index = edge_index
    res.edge_type = edge_type
    return res


class MyModel(nn.Module):
    def __init__(self, in_channels, hid_channels, num_cls, num_relations):
        super().__init__()
        self.gnn_encoder = RGCNEncoder(in_channels, hid_channels, num_relations)
        self.cls = nn.Sequential(
            nn.Linear(hid_channels * 2, hid_channels),
            nn.LeakyReLU(),
            nn.Linear(hid_channels, num_cls)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.graph_generator = Generator(input_dim=in_channels, hidden_dim=hid_channels)
        self.hidden_dim = hid_channels

    def forward(self, data):
        initial_rep = self.gnn_encoder(data)
        learned_mask, dropped_mask = self.graph_generator(data)

        learned_graph = graph_sample(data, learned_mask)
        # dropped_graph = graph_sample(data, dropped_mask)

        # dropped_graph.edge_index = dropped_graph.edge_index[:, learned_mask]
        # dropped_graph.edge_type = dropped_graph.edge_type[learned_mask]

        learned_rep = self.gnn_encoder(learned_graph)
        # dropped_rep = self.gnn_encoder(dropped_graph)

        # disentangled_loss = pearson_loss(graph_rep, dropped_rep)
        rep = torch.cat([initial_rep, learned_rep], dim=-1)

        y_pred = self.cls(rep)
        y = data.y
        loss = self.loss_fn(y_pred, y)

        # loss += disentangled_loss

        return y_pred, loss, y

    def graph_ablation(self, data, ablation):
        assert ablation in ['initial', 'learned']
        batch_size = data.batch_size
        if ablation == 'initial':
            initial_rep = torch.zeros(batch_size, self.hidden_dim).to(data.x.device)
        else:
            initial_rep = self.gnn_encoder(data)
        if ablation == 'learned':
            learned_rep = torch.zeros(batch_size, self.hidden_dim).to(data.x.device)
        else:
            learned_mask, dropped_mask = self.graph_generator(data)

            learned_graph = graph_sample(data, learned_mask)

            learned_rep = self.gnn_encoder(learned_graph)
        # dropped_rep = self.gnn_encoder(dropped_graph)

        # disentangled_loss = pearson_loss(graph_rep, dropped_rep)
        rep = torch.cat([initial_rep, learned_rep], dim=-1)

        y_pred = self.cls(rep)
        y = data.y
        loss = self.loss_fn(y_pred, y)

        # loss += disentangled_loss

        return y_pred, loss, y

    def representation(self, data):
        initial_rep = self.gnn_encoder(data)

        learned_mask, dropped_mask = self.graph_generator(data)
        learned_graph = graph_sample(data, learned_mask)
        learned_rep = self.gnn_encoder(learned_graph)

        return initial_rep, learned_rep


