import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sagemaker_containers.beta.framework import worker, encoders
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from torch.utils.data import DataLoader

from TRnn import TRnn
from helper import num_params, denorm
from tabular_data_loader import TabularDataset

logger = logging.getLogger(__name__)

"""
Tabular RNN for AWS Sagemaker
Requires PyTorch >= 0.4.0

Training Script
"""

model_type = "GRU"  # Choices are RNN, GRU and LSTM

# RNN TENSOR DIMENSION SETUP
num_layers = 32
batch_size = 16

hidden_size = 100  # number of hidden features passed between layers
input_size = 200  # number of inputs at each step
output_size = 1  # number of outputs expected at each time step


def sequence_collate(batch):
    """
    This custom collate function is necessary to support batches that may contain variable length sequences (different amounts of data available)
    The function pads the sequences to be the same length and then packs them specially so torch will know to not actually run the full padded length
    This is a requirement because of CUDNN
    """
    x_out = [item['X'] for item in batch]
    y_out = [item['Y'] for item in batch]

    x_out = pack_sequence(x_out)
    y_out = pack_sequence(y_out)

    return {'X': x_out, 'Y': y_out}


def set_optim(rnn_optimizer):
    """
    Return specified optimizer
    """
    # Set loss and optimizer function
    if rnn_optimizer == "rmsprop":
        return torch.optim.RMSprop(rnn.parameters(), lr=lr)
    elif rnn_optimizer == "sgd":
        return torch.optim.SGD(rnn.parameters(), lr=lr, momentum=momentum, nesterov=True)
    elif rnn_optimizer == "adam":
        return torch.optim.Adam(rnn.parameters(), lr=lr)
    elif rnn_optimizer == "adadelta":
        return torch.optim.Adadelta(rnn.parameters(), lr=lr)
    else:
        raise ValueError('Unspecified optimizer provided')


def train(epochs, best_loss=sys.maxsize):
    """
    Trains the RNN, assumes hyperparameters have already been determined and set in research (for deployment only, no validation occurs here)
    :param epochs: the number of epochs to train
    :param best_loss: the best loss seen at training
    """
    print("-" * 9, "BEGIN TRAINING", "-" * 9)
    print(configs)
    print(num_params(rnn))
    print("Optimizing with:", rnn_optimizer)
    for epoch in range(epochs):
        rnn.train()
        print("-" * 9, "BEGIN EPOCH", epoch, "-" * 9)
        rolling_loss = 0
        num_inst = 0
        for iterat, batch in enumerate(loader):
            inputs = batch['X']
            labels = batch['Y']

            # run through network and get outputs
            outputs = rnn(inputs)

            # We pack each batch within the network, now we unpack/pad it to return it to normal shape
            # this allows us to provide different sequence lengths within the same batch
            # BUT batches must be sorted by length in decreasing order (most to fewest)
            outputs, unpacked_len = pad_packed_sequence(outputs, batch_first=True)
            labels, unpacked_lab_len = pad_packed_sequence(labels, batch_first=True)

            # zero our gradients
            optimizer.zero_grad()

            # calculate loss
            loss = criterion(outputs, labels)
            rolling_loss += loss.data
            num_inst += 1

            # backpropagate loss through the network and step our optimizer
            loss.backward()
            optimizer.step()
            if iterat % 200 == 0:
                print("Iteration:", iterat, "Loss:", (rolling_loss / num_inst).item())

        if rolling_loss <= best_loss:  # set greater or equal since equal may still have better weights moving in the right direction
            best_loss = rolling_loss

            filename = os.path.join(args['model_dir'], 'model.torch')
            dirname = os.path.dirname(filename)
            if dirname != '':
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
            print("Saving Model")
            with open(filename, 'wb') as f:
                torch.save(rnn.state_dict(), f)
        print("*" * 4, "Metrics", "*" * 4)
        print("LOSS:", (rolling_loss.data / num_inst).item())


##########
# AWS SAGEMAKER FUNCTIONS
##########
def model_fn(model_dir):
    """
    Load the model and return it
    :param model_dir: this is passed automatically by sagemaker
    :return: The RNN model after loading
    """
    the_model = TRnn(model_type=model_type, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_size=batch_size)
    with open(os.path.join(model_dir, 'model.torch'), 'rb') as f:
        the_model.load_state_dict(torch.load(f))
    return the_model


def predict_fn(input_data, rnn):
    """
    Make a prediction for next time step given input data and a model
    :param input_data: This is given as a torch tensor to us from sagemaker
    :param rnn: This is the loaded model given by model_fun
    :return: prediction for next time step
    """
    # Load data from inference call
    tabular_norms = pd.DataFrame(norm_name)
    data = input_data.numpy().reshape(-1, input_size)
    data = torch.Tensor(data)  # convert to Torch Tensor

    # normalize the data and prep it for forward pass
    for idx, col in enumerate(tabular_norms.columns.values):
        data[:, idx:idx + 1] = (data[:, idx:idx + 1] - tabular_norms[col][0]) / (tabular_norms[col][1] - tabular_norms[col][0])
    data = [data]
    data = pack_sequence(data)

    # ready RNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rnn.to(device)
    rnn.eval()

    # Run forward pass for prediction and return
    with torch.no_grad():
        prediction = rnn(data)
        prediction, unpacked_lab_len = pad_packed_sequence(prediction, batch_first=True)
        prediction = denorm(prediction, tabular_norms)
        prediction = np.abs(prediction)
    return prediction[-1]  # return torch tensor which sagemaker deserializes to numpy


def output_fn(prediction, accept):
    """
    Convert the torch tensor to numpy and return it
    """
    return worker.Response(encoders.encode(prediction, accept), accept)


##########
# MAIN GUARD
##########
if __name__ == '__main__':

    ###
    # ACCEPTED ARGS
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--learningrate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-cuda', '--cuda', type=bool, default=False)
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = vars(parser.parse_args())

    # Sets if we are running on a GPU or CPU
    if args['cuda']:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Hyperparameters (defaults are expected to be set in research, this script is not for validation)
    train_epochs = args['epochs']
    lr = args['learningrate']
    momentum = args['momentum']
    rnn_optimizer = args['optimizer']  # Choices are rmsprop, sgd, adam and adadelta
    batch_size = args['batchsize']
    train_path = args['train']

    # Models Save Dir and Setup Model Name
    configs = "AP_RNN-EXTERN-Neuron_{}-Layers_{}-Optim_{}-LR_{}-BS_{}-epochs_{}.torch".format(model_type, num_layers, rnn_optimizer, lr, batch_size, train_epochs)


    # PyTorch Dataset Setup
    df_name = 'tabular_pytorch.csv'
    norm_name = 'tabular_norms_pytorch.csv'

    data_location = train_path + "/" + df_name
    norm_location = train_path + "/" + norm_name

    dataset = TabularDataset(data_path=data_location, data_path_norms_csv=norm_location, num_inputs=input_size, out_size=output_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=sequence_collate)

    # RNN Setup
    rnn = TRnn(model_type=model_type, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_size=batch_size).to(device)  # Instantiate RNN model

    criterion = torch.nn.L1Loss()
    optimizer = set_optim(rnn_optimizer)  # called after initializing rnn

    train(train_epochs)
