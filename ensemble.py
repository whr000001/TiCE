import glob
import os
import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def inference(model_size, annotation, community):
    subgraphs = os.listdir('../vanilla/subgraph_edge_index')
    print(subgraphs)
    input()
    experts = []
    expert_path = f'expert_output_{annotation}'
    for subgraph in subgraphs:
        pattern = f'{expert_path}/{model_size}_{subgraph}_{community}_*.pt'
        candidates = glob.glob(pattern)
        candidates = sorted(candidates, reverse=True)
        # if len(candidates) == 0:
        #     print(pattern)
        #     return
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


def mean(model_size, annotation, community):
    subgraphs = os.listdir('../vanilla/subgraph_edge_index')
    experts = []
    expert_path = f'expert_output_{annotation}'
    for subgraph in subgraphs:
        pattern = f'{expert_path}/{model_size}_{subgraph}_{community}_*.pt'
        candidates = glob.glob(pattern)
        candidates = sorted(candidates, reverse=True)
        # if len(candidates) == 0:
        #     print(pattern)
        #     return
        expert = torch.load(candidates[0], weights_only=True)
        experts.append(expert)
    all_truth, all_preds = [], []
    for index in range(experts[0][0].shape[0]):
        ans = []
        label = None
        for expert in experts:
            preds, truth, score = expert
            ans.append(preds[index])
            if label is None:
                label = truth[index]
        ans = torch.stack(ans)
        pred = torch.mode(ans)[0]
        all_truth.append(label.item())
        all_preds.append(pred.item())
    micro = f1_score(all_truth, all_preds, average='micro') * 100
    macro = f1_score(all_truth, all_preds, average='macro') * 100
    print(f'{micro:.2f} {macro:.2f}')


def main():
    for community in range(10):
        inference('large', 'verified', community)


if __name__ == '__main__':
    main()
