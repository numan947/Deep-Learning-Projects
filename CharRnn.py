import torch
import torch.nn as nn



class RNNModel(nn.Module):
    """RNN model."""

    def __init__(self, input_size, hidden_size, output_size, recurrent_type="rnn", n_layers=1, batch_first=True, bidirectional=False):
        super(RNNModel, self).__init__()
        
        self.recurrent_type = recurrent_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
     
        self.recurrent_layer = None        
        if recurrent_type == "rnn":
            self.recurrent_layer = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=batch_first,
                bidirectional=bidirectional
                )
        elif recurrent_type == "gru":
            self.recurrent_layer = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=batch_first,
                bidirectional=bidirectional
                )
        elif recurrent_type == "lstm":
            self.recurrent_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=batch_first,
                bidirectional=bidirectional
                )

        if self.bidirectional:
            self.dense = nn.Linear(2*self.hidden_size, self.output_size)
        else:
            self.dense = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):
        # """Forward function"""
        if x is not None:
            x = x.float()
        if hidden is not None:
            hidden=hidden.float()
        Y, hidden = self.recurrent_layer(x, hidden)

        if self.bidirectional:
            Y = Y.contiguous().view(-1, 2*self.hidden_size)
        else:
            Y = Y.contiguous().view(-1, self.hidden_size)

        output = self.dense(Y)
        # output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, hidden

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            stacks=2*self.n_layers
        else:
            stacks=self.n_layers
        if self.recurrent_type == "lstm":
            return (
                torch.zeros(stacks, batch_size, self.hidden_size,requires_grad=True, device=device),
                torch.zeros(stacks, batch_size, self.hidden_size,requires_grad=True, device=device)
                )
        else:
            return torch.zeros(stacks, batch_size, self.hidden_size,requires_grad=True, device=device)