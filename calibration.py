import glob
import json
import os
import torch
import random
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def inference(annotation, community):
    model_size = 'large'
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
    all_truth, all_preds, all_score = [], [], []
    for index in range(experts[0][0].shape[0]):
        scores = []
        label = None
        for expert in experts:
            preds, truth, score = expert
            scores.append(score[index])
            if label is None:
                label = truth[index]
        scores = torch.stack(scores)
        scores = torch.mean(scores, dim=0)
        all_truth.append(label.item())
        all_preds.append(scores.argmax(-1).item())
        score = torch.max(scores, dim=0)[0]
        all_score.append(score.item())
    return all_preds, all_truth, all_score


def select(annotation, community):
    model_size = 'large'
    # subgraphs = os.listdir('../vanilla/subgraph_edge_index')
    # random.seed(20241031)
    # subgraphs = random.sample(subgraphs, k=9)
    subgraphs = ['post']
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
    all_truth, all_preds, all_score = [], [], []
    for index in range(experts[0][0].shape[0]):
        scores = []
        label = None
        for expert in experts:
            preds, truth, score = expert
            scores.append(score[index])
            if label is None:
                label = truth[index]
        scores = torch.stack(scores)
        scores = torch.mean(scores, dim=0)
        all_truth.append(label.item())
        all_preds.append(scores.argmax(-1).item())
        score = torch.max(scores, dim=0)[0]
        all_score.append(score.item())
    return all_preds, all_truth, all_score


def main():
    # all_preds, all_truth, all_score = [], [], []
    # for community in range(10):
    #     preds, truth, score = inference('followers', community)
    #     all_preds += preds
    #     all_truth += truth
    #     all_score += score
    # json.dump([all_preds, all_truth, all_score], open('ours_calibration.json', 'w'))

    all_preds, all_truth, all_score = [], [], []
    for community in range(10):
        preds, truth, score = select('followers', community)
        all_preds += preds
        all_truth += truth
        all_score += score
    json.dump([all_preds, all_truth, all_score], open('post_calibration.json', 'w'))


if __name__ == '__main__':
    main()
