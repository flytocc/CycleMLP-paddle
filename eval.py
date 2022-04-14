# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse

import paddle
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

import util.misc as misc
from util.datasets import build_dataset

from engine import evaluate

import cycle_mlp


RABatchSampler = DistributedBatchSampler


def get_args_parser():
    parser = argparse.ArgumentParser('CycleMLP training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='CycleMLP_B1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--cls_label_path', default=None, type=str,
                        help='dataset label path')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("{}".format(args).replace(', ', ',\n'))

    dataset_val = build_dataset(is_train=False, args=args)

    if args.dist_eval:
        num_tasks = misc.get_world_size()
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = DistributedBatchSampler(
            dataset_val, args.batch_size, shuffle=True, drop_last=False)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = BatchSampler(dataset=dataset_val, batch_size=args.batch_size)

    data_loader_val = DataLoader(dataset_val, batch_sampler=sampler_val, num_workers=args.num_workers)

    model = cycle_mlp.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path
    )

    model = paddle.DataParallel(model)
    model_without_ddp = model._layers
    n_parameters = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
    print('number of params:', n_parameters)

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    model_ema=None, optimizer=None, loss_scaler=None)

    test_stats = evaluate(data_loader_val, model)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CycleMLP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.eval = True
    main(args)
