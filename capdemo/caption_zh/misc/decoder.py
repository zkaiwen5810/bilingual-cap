import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple

class Decoder(nn.Module):
    """解码器，包括词嵌入、LSTM、输出层
    """
    def __init__(self, 
                 output_dim: int, 
                 emb_dim: int, 
                 hid_dim: int, 
                 n_layers: int, 
                 dropout: float):
        """output_dim: 词表大小
        emb_dim: 词的稠密向量的维度
        hid_dim: LSTM隐藏层和cell向量的维度
        n_layers: LSTM的层数
        dropout: 0-1.0
        """
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0.0)
        
        self.out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                input: Tensor, 
                hidden: Tensor, 
                cell: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

if __name__ == '__main__':
    batch_size = 4
    output_dim = 16
    emb_dim = 32
    hid_dim = 64 
    n_layers = 2
    dropout = 0.5
    decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
    input_ts = torch.randint(low=0, high=output_dim, size=(batch_size,))
    hidden = torch.rand((n_layers, batch_size, hid_dim))
    cell = torch.rand((n_layers, batch_size, hid_dim))
    print(decoder(input_ts, hidden, cell)[0].size())