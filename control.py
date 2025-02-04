import os
from argparse import ArgumentParser
import time


parser = ArgumentParser()
parser.add_argument('--community', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--annotation', type=str)
parser.add_argument('--size', type=str)
args = parser.parse_args()
community = args.community
gpu = args.gpu
annotation = args.annotation
model_size = args.size

assert annotation in ['followers', 'following', 'tweet', 'verified']
assert model_size in ['large', 'base', 'small', 'xsmall']


def main():
    # cmd = 'python obtain_subgraphs.py --size xsmall'
    # os.system(cmd)
    #
    # cmd = 'python obtain_subgraphs.py --size small'
    # os.system(cmd)
    #
    # cmd = 'python obtain_subgraphs.py --size base'
    # os.system(cmd)
    #
    # cmd = 'python obtain_subgraphs.py --size large'
    # os.system(cmd)
    # time.sleep(5 * 60 * 60)
    subgraphs = os.listdir(f'../vanilla/subgraphs_{model_size}')
    for i, subgraph in enumerate(subgraphs):
        cmd = f'CUDA_VISIBLE_DEVICES={gpu} python train.py'
        cmd += f' --size {model_size} --subgraph {subgraph} --community {community} --annotation {annotation}'
        os.system(cmd)


if __name__ == '__main__':
    main()
