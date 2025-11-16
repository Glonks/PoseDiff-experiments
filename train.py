import os
import yaml

from argparse import ArgumentParser

from pose_diff import PoseDiffModel


# TODO: Remove this
import torch
torch.set_printoptions(precision=10)


def main(config):
    model = PoseDiffModel(config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
