# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

import paddle

import util.misc as misc

import cycle_mlp


def get_args_parser():
    parser = argparse.ArgumentParser('CycleMLP training and evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    return parser


def main(args):
    # model
    model = cycle_mlp.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path
    )

    misc.load_model(args=args, model_without_ddp=model,
                    model_ema=None, optimizer=None, loss_scaler=None)

    shape = [-1, 3, args.input_size, args.input_size]

    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.output_dir, 'model')
    paddle.jit.save(model, save_path)
    print(f'Model is saved in {args.output_dir}.') # model.pdiparams|info|model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CycleMLP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.eval = True
    main(args)
