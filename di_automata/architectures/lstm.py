from typing import Callable
import torch
from torch import nn
import functools


class LSTMModule(nn.Module):
    """Basic LSTM model.
    For binary strings, sequence length = string length. Batch size arbitrary. Input size 1.

    Args:
        input_dim: The number of expected features in the input `x`
        hidden_dim: The number of features in the hidden state `h`
        output_dim: The number of expected features in the output
        num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together
        activation_constructor: Auto set to sigmoid for binary classification.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int,
        activation_constructor: Callable[[], nn.Module] = functools.partial(nn.Sigmoid, inplace=True)
        ):
        super(LSTMModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.activation = activation_constructor()

    def init_hidden(self, batch_size):
        """Set both h and c to zero at the start of each input sequence (binary string).
        This should be done in the main training loop.
        """
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                       torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        
    def initialise_params(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, input):
        """Assume input dimensions are (batch_size, sequence_length, input_size).
        - deviation from typical (sequence_length, batch_size, input_size) format that PyTorch LSTMs usually expect. 
        ."""
        batch_size = input.size(0)
        # Forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(input.view(len(input), batch_size, -1), self.hidden)
        y_pred = self.linear(lstm_out[-1].view(batch_size, -1))
        y_pred = self.activation(y_pred)
        return y_pred
