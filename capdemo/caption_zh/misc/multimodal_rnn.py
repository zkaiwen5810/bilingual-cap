import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch import Tensor
import random
import logging

from .encoder import Vgg16Feats

class MultimodalRNN(nn.Module):
    """multimodal recurrent neural network Deep Visual-Semantic Alignments for Generating Image Descriptions
    """
    def __init__(self, decoder:nn.Module):
        super().__init__()
        # 特征图 (bs, 512, 7, 7) 特征向量 (bs, 4096)
        self.encoder = Vgg16Feats()
        self.encoder.requires_grad_(False)
        self.decoder = decoder
        # self.ncaps_per_img = ncaps_per_img

        self.hid_dim = decoder.hid_dim
        self.dropout = decoder.dropout

        # 一半用作隐状态，另一半用作cell
        self.hidden_projection = nn.Linear(4096, self.hid_dim, self.dropout)
        self.cell_projection = nn.Linear(4096, self.hid_dim, self.dropout)
    
    def forward(self, x:Tensor, y:Tensor, teacher_forcing_ratio:float):
        """输入是batch后的图像(bs, 3, 224, 224)，数字化后的句子(max_len, bs * ncaps_per_img)
        teacher_forcing_ratio设置为 负数 就是预测行为，设置为1.0或大于1.0就永远是teacher forcing
        """
        # batch_size = x.size()[0]
        max_len, total_bs = y.size()

        # vocab_size = self.decoder.output_dim

        outputs = []

        feature_map, feature_vector = self.encoder(x)
        # length是图像特征向量的维数
        bs, depth, height, width = feature_map.size()
        assert total_bs % bs == 0
        ncaps_per_img = total_bs // bs
        length = feature_vector.size()[1]

        feature_map = feature_map.unsqueeze(0).expand(ncaps_per_img, bs, depth, height, width).reshape(-1, depth, height, width)
        feature_vector = feature_vector.unsqueeze(0).expand(ncaps_per_img, bs, length).reshape(-1, length)

        hidden = F.relu(self.hidden_projection(feature_vector))
        cell = F.relu(self.cell_projection(feature_vector))

        hidden, cell = hidden.unsqueeze(0), cell.unsqueeze(0)
        
        # hidden, cell = init_vector[:, : self.hid_dim], init_vector[:, self.hid_dim :]
        # hidden, cell = hidden.unsqueeze(0), cell.unsqueeze(0)
        y_t = y[0, :]

        for ts in range(1, max_len):
            output, hidden, cell = self.decoder(y_t, hidden, cell)
            outputs.append(output)
            # random.random()产生的随机数 [0.0, 1.0)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            y_t = (y[ts, :] if teacher_force else top1)
        
        return torch.stack(outputs)
    
    def finetune(self, optim:Optimizer, optim_algo:str, lr:float, alpha:float, beta:float, eps:float, weight_decay:float):
        """精调编码器，指定优化算法及其参数
        """
        self.encoder.requires_grad_(True)
        if optim_algo == 'adam':
            param_grp = {
                'params': self.encoder.parameters(),
                'initial_lr': lr,
                'lr': lr,
                'betas': (alpha, beta),
                'eps': eps,
                'weight_decay': weight_decay
            }
        elif optim_algo == 'rmsprop':
            param_grp = {
                'params': self.encoder.parameters(),
                'initial_lr': lr,
                'lr': lr,
                'alpha': alpha,
                'eps': eps,
                'weight_decay': weight_decay
            }
        elif optim_algo == 'sgd':
            param_grp = {
                'params': self.encoder.parameters(),
                'initial_lr': lr,
                'lr': lr,
                'weight_decay': weight_decay
            }
        else:
            logging.error('{} not suported for cnn'.format(optim_algo))
            exit()
        optim.add_param_group(param_grp)
# if __name__ == '__main__':
#     from .decoder import Decoder
#     output_dim = 16
#     emb_dim = 8
#     hid_dim = 8
#     n_layers = 1
#     dropout = 0.5

#     ncaps_per_img = 2
#     decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
#     model = MultimodalRNN(decoder, ncaps_per_img)

#     bs = 4
#     max_len = 8
#     x = torch.rand((bs, 3, 224, 224))
#     y = torch.randint(low=0, high=output_dim, size=(max_len, bs))

