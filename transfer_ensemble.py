import glob
import os
import torch
import json
from sklearn.metrics import f1_score
import numpy as np


def ensemble(model_size, annotation, community):
    subgraphs = os.listdir('subgraph_edge_index')
    experts = []
    expert_path = 'expert_output'
    if annotation != 'followers':
        expert_path += f'_{annotation}'
    for subgraph in subgraphs:
        pattern = f'{expert_path}/{model_size}_{subgraph}_{community}_*.pt'
        candidates = glob.glob(pattern)
        candidates = sorted(candidates, reverse=True)
        # print(pattern)
        expert = torch.load(candidates[0], weights_only=True)
        experts.append(expert)
    all_truth, all_preds = [], []
    for index in range(experts[0][0].shape[0]):
        scores = []
        label = None
        for expert in experts:
            preds, truth, score = expert
            scores.append(score[index])
            if label is None:
                label = truth[index]
        scores = torch.stack(scores)
        scores = torch.sum(scores, dim=0)
        all_truth.append(label.item())
        all_preds.append(scores.argmax(-1).item())
    micro = f1_score(all_truth, all_preds, average='micro') * 100
    macro = f1_score(all_truth, all_preds, average='macro') * 100
    print(f'{micro:.2f} {macro:.2f}')


def main():
    annotation = 'verified'
    subgraphs = os.listdir('../vanilla/subgraph_edge_index')
    data = []
    for subgraph in subgraphs:
        res = torch.load(f'transfer_res/{annotation}_{subgraph}.pt', weights_only=True)
        data.append(res)
    res = {}
    for train_community in range(10):
        for test_community in range(10):
            experts = []
            for item in data:
                experts.append(item[f'{train_community}_{test_community}'])
            all_truth, all_preds = [], []
            for index in range(experts[0][0].shape[0]):
                scores = []
                label = None
                for expert in experts:
                    preds, truth, score = expert
                    scores.append(score[index])
                    if label is None:
                        label = truth[index]
                scores = torch.stack(scores)
                scores = torch.sum(scores, dim=0)
                all_truth.append(label.item())
                all_preds.append(scores.argmax(-1).item())
            micro = f1_score(all_truth, all_preds, average='micro') * 100
            macro = f1_score(all_truth, all_preds, average='macro') * 100
            res[f'{train_community}_{test_community}'] = (micro, macro)
    json.dump(res, open(f'transfer_res/ours_{annotation}.json', 'w'))


if __name__ == '__main__':
    main()
