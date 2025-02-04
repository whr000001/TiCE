import time
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Batch
from tqdm import tqdm
import json
from torch_geometric.data import Data


class ImbalancedSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(None)
        self.indices = list(range(len(dataset)))
        self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.num_samples = len(self.indices)

        label_to_count = {}
        for idx in self.indices:
            label = dataset[idx].y.to('cpu').item()
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        weights = [1.0 / label_to_count[dataset[idx].y.to('cpu').item()]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        shuffle_perm = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_perm]
        self.weights = self.weights[shuffle_perm]
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


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
    def __init__(self, model_size, name, community_index, annotation, device):
        initial = torch.load(f'../vanilla/subgraphs_{model_size}_new/{name}/sub_community_{community_index}.pt',
                             weights_only=False)
        data = []
        for item in initial:
            data.append(Data(x=item[0],
                             edge_index=item[1],
                             edge_type=item[2],
                             index=item[3]))
        labels = torch.load(f'../../example/{annotation}_cnt_processed.pt', weights_only=True)
        self.num_cls = labels.max().item() + 1
        self.num_relations = 0
        indices = json.load(open(f'../../preprocessed/Twibot-22/sub-communities/indices_{community_index}.json'))
        indices = torch.tensor(indices, dtype=torch.long)
        labels = labels[indices]
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
