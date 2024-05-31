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
                 dropout: float,
                 kdim: int,
                 vdim: int):
        """output_dim: 词表大小
        emb_dim: 词的稠密向量的维度
        hid_dim: LSTM隐藏层和cell向量的维度
        dropout: 0-1.0
        """
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.rnn0 = LSTM_Layer(emb_dim, hid_dim, is_attended=True, kdim=kdim, vdim=vdim)
        self.dropout0 = nn.Dropout(dropout)
        self.rnn1 = LSTM_Layer(hid_dim, hid_dim, is_attended=False)

        self.out = nn.Linear(hid_dim, output_dim)
        
    def forward(self, 
                input: Tensor, 
                hidden: Tensor, 
                cell: Tensor,
                k: Tensor,
                v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ 因为是解码器，input不包含代表句子长度的维度，只是一个minibatch里单词的索引
        hidden和cell的shape是(2, bs, hid_dim)
        k,v的shape是(49, bs, kdim或vdim)"""
        #input = [bs]
        #hidden = [n layers * n directions, bs, hid dim]
        #cell = [n layers * n directions, bs, hid dim]
        
        #n directions in the decoder will both always be 1, n layers be 2, therefore:
        #hidden = [2, bs, hid dim]
        #context = [2, bs, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, bs]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, bs, emb dim]
                
        output, attn_weights, (hidden0, cell0) = self.rnn0(embedded, (hidden[0].unsqueeze(0), cell[0].unsqueeze(0)), k, v)
        
        output = self.dropout0(output)

        output, _, (hidden1, cell1) = self.rnn1(output, (hidden[1].unsqueeze(0), cell[1].unsqueeze(0)))
        
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, attn_weights, torch.cat((hidden0, hidden1)), torch.cat((cell0, cell1))

class LSTM_Layer(nn.Module):
    """ 作为解码器的lstm层 """
    def __init__(self, input_dim:int, hidden_dim:int, is_attended:bool=False, kdim:int=None, vdim:int=None):
        """ 是否有注意力层 """
        assert (not is_attended) or (is_attended and (kdim is not None) and (vdim is not None))
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.is_attended = is_attended
        self.kdim = kdim
        self.vdim = vdim

        self.rnn = nn.LSTM(input_dim, hidden_dim, 1)
        if is_attended:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, kdim=kdim, vdim=vdim)

    def forward(self, y:Tensor, prev:Tuple[Tensor, Tensor], k:Tensor=None, v:Tensor=None):
        """ y (1, bs, input_dim) 模拟LSTM且作为解码器使用
        prev (hidden, cell) 上一时刻的输出，shape都是(1, bs, hidden_dim)，因为num_layers和num_directions
        都是1
        k (49, bs, 2048)
        v (49, bs, 2048)
        """

        output, (h, c) = self.rnn(y, prev)

        output_weights = None
        if self.is_attended:
            # output (1, bs, hidden_dim) output_weights (bs, 1, 49)
            output, output_weights = self.attention(output, k, v)

        return output, output_weights, (h, c)

if __name__ == '__main__':
    bs = 2
    output_dim = 10
    emb_dim = 8
    hid_dim = 4
    dropout = 0.2
    
    kdim = 32
    vdim = kdim

    y = torch.randint(low=0, high=output_dim, size=(2,))
    x = torch.randn((16, bs, kdim))
    hidden = torch.randn((2, bs, hid_dim))
    cell = torch.randn((2, bs, hid_dim))

    model = Decoder(output_dim, emb_dim, hid_dim, dropout, kdim, vdim)
    model(y, hidden, cell, x, x)
