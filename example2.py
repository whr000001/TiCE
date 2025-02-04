import torch
import torch.nn as nn
from GNN import GNN
from dataset import domain_shift_dataset
from torch_geometric.loader import DataLoader
# from utils.get_subgraph import relabel, split_batch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (negative_sampling, remove_self_loops, degree, add_self_loops,
                                   batched_negative_sampling)
import torch_geometric.data.batch as DataBatch


def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges


def relabel(x, edge_index, batch, pos=None):

    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=x.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            #PyG 2.0.4
            module._explain = True
            module._edge_mask = mask
            #PyG 1.7.2
            module.__explain__ = True
            module.__edge_mask__ = mask


def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            #PyG 2.0.4
            module._explain = False
            module._edge_mask = None
            #PyG 1.7.2
            module.__explain__ = False
            module.__edge_mask__ = None


class CIGA(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gnn_encoder = GNN(in_dim=in_dim, hidden_dim=hidden_dim, num_unit=3, gnn='GCN')
        self.edge_att = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 4),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim * 4, 1))
        c_in = "raw"
        c_rep = "rep"
        c_pool = "add"
        s_rep = "rep"
        self.ratio = 0.25
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = hidden_dim
        self.c_in = c_in
        self.c_input_dim = in_dim if c_in.lower() == "raw" else hidden_dim

    def split_graph(self, data, edge_score, ratio):
        # Adopt from GOOD benchmark to improve the efficiency
        from torch_geometric.utils import degree
        def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
            r'''
            Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
            '''
            f_src = src.float()
            f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
            norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
            perm = norm.argsort(dim=dim, descending=descending)

            return src[perm], perm

        def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
            rank, perm = sparse_sort(src, index, dim, descending, eps)
            num_nodes = degree(index, dtype=torch.long)
            k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
            start_indices = torch.cat([torch.zeros((1,), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
            mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in
                    range(len(num_nodes))]
            mask = torch.cat(mask, dim=0)
            mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
            topk_perm = perm[mask]
            exc_perm = perm[~mask]

            return topk_perm, exc_perm, rank, perm, mask

        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio,
                                                             descending=True)
        new_causal_edge_index = data.edge_index[:, new_idx_reserve]
        new_spu_edge_index = data.edge_index[:, new_idx_drop]

        new_causal_edge_weight = edge_score[new_idx_reserve]
        new_spu_edge_weight = -edge_score[new_idx_drop]

        if has_edge_attr:
            new_causal_edge_attr = data.edge_attr[new_idx_reserve]
            new_spu_edge_attr = data.edge_attr[new_idx_drop]
        else:
            new_causal_edge_attr = None
            new_spu_edge_attr = None

        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)

    def forward(self, batch):
        h = self.gnn_encoder(batch.x, batch.edge_index)
        print(h.shape)
        device = h.device
        row, col = batch.edge_index
        edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        print(batch.edge_index.shape)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)
        print(pred_edge_weight.shape)
        if self.ratio < 0:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = (
            batch.edge_index, batch.edge_attr, pred_edge_weight), \
                (batch.edge_index, batch.edge_attr, pred_edge_weight)
        else:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = self.split_graph(batch, pred_edge_weight, self.ratio)
        if self.c_in.lower() == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        else:
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)

            # obtain \hat{G_c}
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        print(causal_graph)
        # set_masks(causal_edge_weight, self.classifier)
        # # obtain predictions with the classifier based on \hat{G_c}
        # causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        # clear_masks(self.classifier)
        print('haha')




if __name__ == '__main__':
    model = CIGA(in_dim=1024, hidden_dim=512)
    m_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(m_device)
    train_data, val_data, test_data = domain_shift_dataset('Twitter15', 'Twitter16', 1.0)
    train_data = [item.to(m_device) for item in train_data]
    val_data = [item.to(m_device) for item in val_data]
    test_data = [item.to(m_device) for item in test_data]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    for x in train_loader:
        model(x)
        input()
    pass
