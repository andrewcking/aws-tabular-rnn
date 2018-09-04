from functools import reduce

import numpy as np
import torch


def num_params(rnn):
    """
    Get the number of parameters in the network
    :param rnn: The RNN
    :return: number of trainable parameters in the RNN
    """
    model_parameters = filter(lambda p: p.requires_grad, rnn.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return str("Number of Trainable Parameters:" + str(num_params))


def flatten(in_torch_tensor):
    """
    PyTorch currently lacks a flatten helper function but may add this in a future release
    :param in_torch_tensor: Tensor to be flattened
    :return: Flattened Tensor
    """
    return torch.Tensor(reduce(lambda x, y: x + y, in_torch_tensor.cpu().numpy().tolist()))


def denorm(data, norms):
    """
    Denormalize a normalized output from a network
    :param data: data to be denormalized
    :param norms: the normalize metrics from the normalize dataframe
    :return: The denormalized data
    """
    # we are denorming the 3 funnel pieces
    if data.shape[2] == 3:
        d_sq = np.squeeze(data)
        for idx, col in enumerate(norms.columns.values):
            d_sq[..., idx:idx + 1] = (d_sq[..., idx:idx + 1] * (norms[col][1] - norms[col][0])) + norms[col][0]
        return d_sq
    else:
        raise ValueError('Denorm Failed, not written to work with this Tensor Output Shape')
