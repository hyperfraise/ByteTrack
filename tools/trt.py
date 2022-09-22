import numpy as np
from loguru import logger

import tensorrt as trt
import torch

from yolox.exp import get_exp

import argparse
import os
import shutil


def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    return parser


class TypeCaster(torch.nn.Module):
    is_half = False

    def __init__(self, input_type):
        super(TypeCaster, self).__init__()
        self.input_type = input_type

    def half(self):
        self.is_half = True
        return super(TypeCaster, self).half()

    def forward(self, x):
        if self.is_half:
            x = x.half()
        else:
            x = x.float()
        if self.input_type == "int8":
            x = x.where(x >= 0., x+256.)
        self.normalization_layer(x)
        x /= 255.0
        x -= (0.485, 0.456, 0.406)
        return x / (0.229, 0.224, 0.225)


@logger.catch
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.eval()

    input_tensor = torch.randint(0, 256, (1, 3, exp.test_size[0], exp.test_size[1])).char()

    typecaster_layer = TypeCaster(input_type="int8")
    model = torch.nn.Sequential(
        *[typecaster_layer, model]
    )

    model.cuda()
    model.head.decode_in_inference = False
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda().float()
    dynamic_axes = {"input": {0: 'batch_size'}, "output": {0: 'batch_size'}}
    with torch.no_grad():
        torch.onnx.export(
            model,  # model being run
            x,  # model input (or a tuple for multiple inputs)
            "model.onnx",
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            export_params=True,
            #dynamic_axes=dynamic_axes,
            verbose=True,
            opset_version=13
        )
    return


if __name__ == "__main__":
    main()
