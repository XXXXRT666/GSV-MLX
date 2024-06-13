import torch
import numpy


torchmodel = "/Users/XXXXRT/GSV-mlx/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
mlxmodel = "/Users/XXXXRT/GSV-mlx/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.npz"


def convert1(torch_model: str, mlx_model: str) -> None:
    model = torch.load(torch_model, map_location="cpu")

    for key, tensor in model["weight"].items():
        model["weight"][key] = tensor.numpy()
    numpy.savez(mlx_model, **model)

convert1(torchmodel, mlxmodel)

