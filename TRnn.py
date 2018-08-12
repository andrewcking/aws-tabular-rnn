import torch.nn as nn


class TRnn(nn.Module):
    """
    Simple Tabular Dataset Recurrent Neural Network Architecture
    Can use LSTM, RNN, and GRU neural units
    """

    def __init__(self, model_type, input_size, hidden_size, num_layers, batch_size):
        super(TRnn, self).__init__()

        self.input_size = input_size  # number of input features
        self.hidden_size = hidden_size  # number of hidden features
        self.num_layers = num_layers  # number of layers (like in a DNN - the deeper the more complex function but vanishing gradient)
        self.model_type = model_type
        self.batch_size = batch_size
        # expects (sequence, batch, input_size) but we are providing batches first so specify batch_first=True
        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, batch_first=True, bias=True)
        elif model_type == "RNN":
            self.rnn = nn.RNN(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, batch_first=True, bias=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, batch_first=True, bias=True)
        else:
            raise ValueError('Unspecified model type provided')

    def forward(self, x):

        output, _ = self.rnn(x)

        return output
