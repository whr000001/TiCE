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

    # '''for single'''
    # for community in range(10):
    #     expert_path = f'expert_output_{annotation}'
    #     max_micro = 0
    #     max_subgraph = None
    #     max_candidate = None
    #     for subgraph in subgraphs:
    #         pattern = f'{expert_path}/large_{subgraph}_{community}_*.pt'
    #         candidates = glob.glob(pattern)
    #         candidates = sorted(candidates, reverse=True)
    #         # if len(candidates) == 0:
    #         #     print(pattern)
    #         #     return
    #         micro = candidates[0].split('_')[-1].replace('.pt', '')
    #         micro = float(micro)
    #         if micro > max_micro:
    #             max_micro = micro
    #             max_subgraph = subgraph
    #             max_candidate = candidates[0]
    #     # print(max_micro, max_subgraph)
    #     experts = [torch.load(max_candidate, weights_only=True)]
    #     calculate(experts)

    '''for comprehensive'''
    subgraph = 'contain_discuss_followed_followers_following_like_membership_' \
               'mentioned_own_pinned_post_quoted_replied_to_retweeted'
    for community in range(10):
        expert_path = f'expert_output_{annotation}'

        pattern = f'{expert_path}/large_{subgraph}_{community}_*.pt'
        candidates = glob.glob(pattern)
        candidates = sorted(candidates, reverse=True)
        experts = [torch.load(candidates[0], weights_only=True)]
        calculate(experts)


if __name__ == '__main__':
    main()
