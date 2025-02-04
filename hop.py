import os
import json
import numpy as np
import torch
from model import MyModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Batch, Data
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import NeighborLoader
import torch_geometric


class MySampler(Sampler):
    def __init__(self, indices, shuffle):
        super().__init__(None)
        self.indices = indices
        if not torch.is_tensor(self.indices):
            self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = self.indices[torch.randperm(self.indices.shape[0])]
        else:
            indices = self.indices
        for item in indices:
            yield item

    def __len__(self):
        return len(self.indices)


class TrainDataset(Dataset):
    def __init__(self, data, labels, device):
        self.data = []
        for index, item in enumerate(data):
            item.y = labels[index]
            self.data.append(item)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].to(self.device)


def get_collate_fn():
    def collate_fn(batch):
        batch = Batch.from_data_list(batch)
        return batch

    return collate_fn


@torch.no_grad()
def inference(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    all_score = []
    # pbar = tqdm(loader, leave=False)
    # pbar.set_description_str(f'{train_community} {test_community}')
    for batch in loader:
        out, loss, truth = model(batch)

        score = torch.softmax(out, dim=-1).to('cpu')
        preds = out.argmax(-1).to('cpu')
        truth = truth.to('cpu')

        all_truth.append(truth)
        all_preds.append(preds)
        all_score.append(score)
    all_preds = torch.cat(all_preds, dim=0).clone()
    all_truth = torch.cat(all_truth, dim=0).clone()
    all_score = torch.cat(all_score, dim=0).clone()
    # print(f1_score(all_truth, all_preds, average='micro'))
    return [all_preds, all_truth, all_score]


def obtain(subgraph):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm(range(10), leave=False)
    pbar.set_description_str(subgraph)
    res = {}
    for community in pbar:
        initial = torch.load(f'../vanilla/subgraphs_large_new/{subgraph}/sub_community_{community}.pt',
                             weights_only=False)
        for hop in [1, 2, 3, 4]:
            if hop == 4:
                num_neighbors = [16, 8, 4, 2]
            elif hop == 3:
                num_neighbors = [16, 8, 4]
            elif hop == 2:
                num_neighbors = [16, 8]
            elif hop == 1:
                num_neighbors = [16]
            else:
                raise KeyError
            data = []
            torch_geometric.seed_everything(20241031)
            for item in initial:
                whole = Data(x=item[0], edge_index=item[1], edge_type=item[2], index=item[3])
                input_nodes = torch.zeros(1)
                neighbor = NeighborLoader(whole, input_nodes=input_nodes, num_neighbors=num_neighbors,
                                          batch_size=1, shuffle=False)
                for _ in neighbor:
                    data.append(_)
            for annotation in ['followers', 'following', 'tweet', 'verified']:
                labels = torch.load(f'../../example/{annotation}_cnt_processed.pt', weights_only=True)
                num_cls = labels.max().item() + 1
                num_relations = len(subgraph.split('_'))
                if 'replied_to' in subgraph:
                    num_relations -= 1
                indices = json.load(open(f'../../preprocessed/Twibot-22/sub-communities/indices_{community}.json'))
                indices = torch.tensor(indices, dtype=torch.long)
                labels = labels[indices]

                random.seed(20241031)
                dataset = TrainDataset(data, labels, device)
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                train_size = int(len(indices) * 0.6)
                val_size = int(len(indices) * 0.2)
                test_indices = indices[train_size + val_size:]
                test_sampler = MySampler(test_indices, shuffle=False)
                loader = DataLoader(dataset, batch_size=64, collate_fn=get_collate_fn(), sampler=test_sampler)

                checkpoint_path = f'checkpoints_large_{annotation}/{subgraph}_{community}'
                checkpoint_names = os.listdir(checkpoint_path)
                checkpoint_name = sorted(checkpoint_names, reverse=True)[0]
                # print(checkpoint_name)
                checkpoint = torch.load(f'{checkpoint_path}/{checkpoint_name}', weights_only=True)

                model = MyModel(in_channels=1024, hid_channels=256,
                                num_cls=num_cls, num_relations=num_relations).to(device)
                model.load_state_dict(checkpoint)
                out = inference(model, loader)
                res[f'{annotation}_{community}_{hop}'] = out
    torch.save(res, f'hop_res/{subgraph}.pt')


def main():
    # obtain('followers_following_post')
    subgraphs = os.listdir(f'../vanilla/subgraphs_large')
    for subgraph in subgraphs:
        obtain(subgraph)


if __name__ == '__main__':
    main()
