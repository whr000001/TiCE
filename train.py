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


parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--subgraph', type=str, default='followers_following')
parser.add_argument('--size', type=str, default='xsmall')
parser.add_argument('--community', type=int,  default=0)
parser.add_argument('--annotation', type=str,  default='followers')
args = parser.parse_args()
batch_size = args.batch_size
lr = args.lr
subgraph = args.subgraph
model_size = args.size
community = args.community
annotation = args.annotation
num_relations = len(subgraph.split('_'))
if 'replied_to' in subgraph:
    num_relations -= 1

if model_size == 'xsmall':
    input_dim = 384
elif model_size == 'small':
    input_dim = 768
elif model_size == 'base':
    input_dim = 768
elif model_size == 'large':
    input_dim = 1024
else:
    raise KeyError


def train_one_epoch(model, optimizer, loader, epoch):
    model.train()
    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    # pbar = tqdm(loader, leave=False)
    # pbar.set_description_str(f'Training {epoch} epoch')
    for batch in loader:
        optimizer.zero_grad()
        out, loss, truth = model(batch)
        loss.backward()
        optimizer.step()

        preds = out.argmax(-1).to('cpu')
        truth = truth.to('cpu')

        ave_loss += loss.item() * len(batch)
        cnt += len(batch)
        all_truth.append(truth)
        all_preds.append(preds)

    ave_loss /= cnt
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return ave_loss, accuracy_score(all_truth, all_preds), \
        f1_score(all_truth, all_preds, average='micro'), f1_score(all_truth, all_preds, average='macro')


@torch.no_grad()
def validate(model, loader, split, epoch):
    model.eval()

    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    # pbar = tqdm(loader, leave=False)
    # pbar.set_description_str(f'{split} {epoch} epoch')
    for batch in loader:
        out, loss, truth = model(batch)

        preds = out.argmax(-1).to('cpu')
        truth = truth.to('cpu')

        ave_loss += loss.item() * len(batch)
        cnt += len(batch)
        all_truth.append(truth)
        all_preds.append(preds)

    ave_loss /= cnt
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return ave_loss, accuracy_score(all_truth, all_preds), \
        f1_score(all_truth, all_preds, average='micro'), f1_score(all_truth, all_preds, average='macro')


@torch.no_grad()
def inference(model, loader):
    model.eval()
    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    all_score = []
    # pbar = tqdm(loader, leave=False)
    # pbar.set_description_str(f'{split} {epoch} epoch')
    for batch in loader:
        out, loss, truth = model(batch)

        score = torch.softmax(out, dim=-1).to('cpu')
        preds = out.argmax(-1).to('cpu')
        truth = truth.to('cpu')

        ave_loss += loss.item() * len(batch)
        cnt += len(batch)
        all_truth.append(truth)
        all_preds.append(preds)
        all_score.append(score)
    ave_loss /= cnt
    all_preds = torch.cat(all_preds, dim=0)
    all_truth = torch.cat(all_truth, dim=0)
    all_score = torch.cat(all_score, dim=0)
    return all_preds, all_truth, all_score


def train(train_loader, val_loader, test_loader, device, save_path):
    model = MyModel(in_channels=input_dim, hid_channels=256,
                    num_cls=train_loader.dataset.num_cls, num_relations=num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_metrics = 0
    best_state = model.state_dict()
    for key, value in best_state.items():
        best_state[key] = value.clone()
    no_up_limits = 10
    no_up = 0
    pbar = range(100)
    # pbar = tqdm(range(100), leave=False)
    for _ in pbar:
        train_loss, train_acc, train_micro, train_macro = train_one_epoch(model, optimizer, train_loader, _)
        # print('train', _, train_loss, train_acc, train_f1)
        val_loss, val_acc, val_micro, val_macro = validate(model, val_loader, 'val', _)
        # print('val', _, val_acc, val_f1)
        # pbar.set_postfix({
        #     'train_loss': train_loss,
        #     'train_micro': train_micro,
        #     'train_macro': train_macro,
        #     'val_micro': val_micro,
        #     'val_macro': val_macro
        # })
        if val_micro > best_metrics:
            best_metrics = val_micro
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
            no_up = 0
        else:
            no_up += 1
        if no_up >= no_up_limits:
            break
    model.load_state_dict(best_state)
    test_loss, test_acc, test_micro, test_macro = validate(model, test_loader, 'test', 0)
    test_micro *= 100
    # print(test_micro, test_macro)
    torch.save(best_state, f'{save_path}/{test_micro:.2f}.pt')
    all_preds, all_truth, all_score = inference(model, test_loader)
    if not os.path.exists(f'expert_output_{annotation}'):
        os.mkdir(f'expert_output_{annotation}')
    inference_path = f'expert_output_{annotation}/{model_size}_{subgraph}_{community}_{test_micro:.2f}.pt'
    torch.save([all_preds, all_truth, all_score], inference_path)


def main():
    name = f'{subgraph}_{community}'
    save_dir = f'checkpoints_{model_size}_{annotation}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = f'{save_dir}/{name}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    done_cnt = len(os.listdir(save_path))
    if done_cnt >= 5:
        exit(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(20241031)
    dataset = TrainDataset(model_size, subgraph, community, annotation, device)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_size = int(len(indices) * 0.6)
    val_size = int(len(indices) * 0.2)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    train_sampler = MySampler(train_indices, shuffle=True)
    val_sampler = MySampler(val_indices, shuffle=False)
    test_sampler = MySampler(test_indices, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=get_collate_fn(), sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=get_collate_fn(), sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=get_collate_fn(), sampler=test_sampler)
    for _ in range(5 - done_cnt):
        train(train_loader, val_loader, test_loader, device, save_path)


if __name__ == '__main__':
    main()
