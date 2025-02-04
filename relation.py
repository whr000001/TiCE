import glob
import json
import os
import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from tqdm import tqdm


def inference(model_size, annotation, subgraphs):
    res = []
    for community in range(10):
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
        res.append(micro)
    return np.mean(res), np.std(res, ddof=1)


def main():
    for annotation in ['followers', 'following', 'tweet', 'verified']:
        size = 'large'
        relations = ['contain', 'discuss', 'followed', 'followers', 'following', 'like', 'membership',
                     'mentioned', 'own', 'pinned', 'post', 'quoted', 'replied_to', 'retweeted']
        subgraphs = os.listdir('../vanilla/subgraph_edge_index/')
        done_experts = {}
        for relation in relations:
            experts = []
            for subgraph in subgraphs:
                if relation in subgraph:
                    experts.append(subgraph)
            if tuple(experts) not in done_experts:
                done_experts[tuple(experts)] = []
            done_experts[tuple(experts)].append(relation)
        res = []
        for key, value in done_experts.items():
            with_relation = list(key)
            without = set(subgraphs) - set(key)
            without_relation = list(without)
            with_res = inference(size, annotation, with_relation)
            without_res = inference(size, annotation, without_relation)
            res.append({
                'with': with_res,
                'without': without_res,
                'with_expert': with_relation,
                'without_expert': without_relation,
                'relations': value
            })
        json.dump(res, open(f'relation_res/{annotation}.json', 'w'))


if __name__ == '__main__':
    main()
