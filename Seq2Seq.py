import myd2l as d2l
import torch
import torch.nn as nn


class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, bidirectional=False, recurrent_type="lstm", batch_first=False, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        recurrent_type = recurrent_type.lower()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.recurrent_type = recurrent_type
        self.batch_first = batch_first


        self.embedding = nn.Embedding(vocab_size, embed_size)
        if recurrent_type == "rnn":
            self.recurrent_layer = nn.RNN(
                input_size=embed_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout
            )
        elif recurrent_type == "lstm":
            self.recurrent_layer = nn.LSTM(
                input_size=embed_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout
            )
        elif recurrent_type == "gru":
            self.recurrent_layer = nn.GRU(
                input_size=embed_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout
            )
        else:
            raise Exception("Unknown recurrent type")
    
    def init_hidden(self, batch_size, device, **kwargs):
        stacks = self.num_layers
        if self.bidirectional:
            stacks*=2
        
        if self.recurrent_type=="lstm":
            return (
                torch.zeros(stacks, batch_size, self.num_hiddens, requires_grad=True, device=device),
                torch.zeros(stacks, batch_size, self.num_hiddens, requires_grad=True, device=device)
            )
        else:
            return torch.zeros(stacks, batch_size, self.num_hiddens, requires_grad=True, device=device)
    
    def forward(self, X, **kwargs):
        X = self.embedding(X.long())
        batch_size = X.shape[0]
        
        if not self.batch_first:
            X = X.transpose(0,1) ## make it sequence first
        
        state = self.init_hidden(batch_size, X.device) ## currently input shape is (seq_len, batch_size, embedding_dim)
        out, state = self.recurrent_layer(X, state)
        
        
        if not self.batch_first:
            out = out.transpose(0,1) ## make it batch first again, or don't it doesn't matter, it's not used in the later phases
        return out, state


class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, bidirectional=False, recurrent_type="lstm", batch_first=False, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        recurrent_type = recurrent_type.lower()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.recurrent_type = recurrent_type
        self.batch_first = batch_first


        self.embedding = nn.Embedding(vocab_size, embed_size)
        if recurrent_type == "rnn":
            self.recurrent_layer = nn.RNN(
                input_size=embed_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout
            )
        elif recurrent_type == "lstm":
            self.recurrent_layer = nn.LSTM(
                input_size=embed_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout
            )
        elif recurrent_type == "gru":
            self.recurrent_layer = nn.GRU(
                input_size=embed_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout
            )
        else:
            raise Exception("Unknown recurrent type")
        
        if self.bidirectional:
            self.dense = nn.Linear(2*self.num_hiddens, self.vocab_size)
        else:
            self.dense = nn.Linear(self.num_hiddens, self.vocab_size)
    

    def init_state(self, enc_outputs, **kwargs):
        return enc_outputs[1]
    
    def forward(self, X, state, **kwargs):
        X = self.embedding(X.long())
        batch_size = X.shape[0]
        
        if not self.batch_first:
            X = X.transpose(0,1) ## make it sequence first
        
        out, state = self.recurrent_layer(X, state)
        out = self.dense(out)

        if not self.batch_first:
            out = out.transpose(0,1) ## make it batch first again, or don't it doesn't matter, it's not used in the later phases
        return out, state