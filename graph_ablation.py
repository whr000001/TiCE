import time
import os
import numpy as np
import torch
from model import MyModel
from torch.utils.data import DataLoader
from dataset import TrainDataset, get_collate_fn, MySampler
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, f1_score
from argparse import ArgumentParser


def inference(annotation, subgraph, ablation):
    input_dim = 1024
    num_relations = len(subgraph.split('_'))
    if 'replied_to' in subgraph:
        num_relations -= 1
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm(range(10))
    pbar.set_description_str(f'{annotation}_{subgraph}_{ablation}')
    res = []
    for community in pbar:
        random.seed(20241031)
        dataset = TrainDataset('large', subgraph, community, annotation, device)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        train_size = int(len(indices) * 0.6)
        val_size = int(len(indices) * 0.2)
        test_indices = indices[train_size + val_size:]
        test_sampler = MySampler(test_indices, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=get_collate_fn(), sampler=test_sampler)

        model = MyModel(in_channels=input_dim, hid_channels=256,
                        num_cls=dataset.num_cls, num_relations=num_relations).to(device)

        checkpoint_path = f'checkpoints_large_{annotation}/{subgraph}_{community}'
        checkpoint_names = os.listdir(checkpoint_path)
        checkpoint_name = sorted(checkpoint_names, reverse=True)[0]
        # print(checkpoint_name)
        checkpoint = torch.load(f'{checkpoint_path}/{checkpoint_name}', weights_only=True)
        model.load_state_dict(checkpoint)
        with torch.no_grad():
            model.eval()
            all_truth = []
            all_preds = []
            all_score = []
            # pbar = tqdm(loader, leave=False)
            # pbar.set_description_str(f'{train_community} {test_community}')
            for batch in loader:
                out, loss, truth = model.graph_ablation(batch, ablation)

                score = torch.softmax(out, dim=-1).to('cpu')
                preds = out.argmax(-1).to('cpu')
                truth = truth.to('cpu')

                all_truth.append(truth)
                all_preds.append(preds)
                all_score.append(score)
            all_preds = torch.cat(all_preds, dim=0).clone()
            all_truth = torch.cat(all_truth, dim=0).clone()
            all_score = torch.cat(all_score, dim=0).clone()
        res.append([all_preds, all_truth, all_score])
    torch.save(res, f'graph_ablation_res/{annotation}_{subgraph}_{ablation}.pt')


def main():
    subgraphs = os.listdir(f'../vanilla/subgraphs_large')
    for subgraph in subgraphs:
        for annotation in ['followers', 'following', 'tweet', 'verified']:
            for ablation in ['initial', 'learned']:
                inference(annotation, subgraph, ablation)


if __name__ == '__main__':
    main()
