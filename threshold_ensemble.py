import glob
import os
import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def calculate(experts):
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
        res = torch.load(f'critical_graph_res/{subgraph}.pt', weights_only=True)
        data.append(res)
    for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
        for community in range(10):
            experts = []
            for item in data:
                experts.append(item[f'{annotation}_{community}_{threshold}'])
            calculate(experts)
        print('-----------')
        input()


if __name__ == '__main__':
    main()
