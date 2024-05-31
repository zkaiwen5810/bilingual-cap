import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch import Tensor
import random
import logging

from .encoder import Encoder
from .decoder_to_feature import Decoder

class MultimodalRNN(nn.Module):
    """resnet101编码器，生成特征图和特征向量。特征图作为注意力的key和value，特征向量作为lstm初始隐向量
    双层lstm，第0层的输出作为注意力的query
    """
    def __init__(self, output_dim:int, emb_dim:int, hid_dim:int, dropout:float):
        super().__init__()
        # 特征图 (batch_size, 2048, 7, 7) 特征向量 (batch_size, 2048)
        self.encoder = Encoder()
        self.encoder.requires_grad_(False)
        self.decoder = Decoder(output_dim, emb_dim, hid_dim, dropout, 2048, 2048)

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout

        # 一半用作隐状态，另一半用作cell
        self.hidden_projection0 = nn.Linear(2048, hid_dim, dropout)
        self.hidden_projection1 = nn.Linear(2048, hid_dim, dropout)
        self.cell_projection0 = nn.Linear(2048, hid_dim, dropout)
        self.cell_projection1 = nn.Linear(2048, hid_dim, dropout)
    
    def forward(self, x:Tensor, y:Tensor, teacher_forcing_ratio:float):
        """输入是batch后的图像(bs, 3, 224, 224)，数字化后的句子(max_len, total_bs)
        teacher_forcing_ratio设置为 负数 就是预测行为，设置为1.0或大于1.0就永远是teacher forcing
        """
        max_len, total_bs = y.size()

        outputs = []
        attns = []
        out_feas = []

        feature_map, feature_vector = self.encoder(x)
        # length是图像特征向量的维数
        bs, depth, height, width = feature_map.size()
        assert total_bs % bs == 0
        ncaps_per_img = total_bs // bs
        # 图片特征向量的维度
        length = feature_vector.size()[1]

        # 把feature map处理成适合注意力机制的shape，首先先扩展，再permute
        feature_map = feature_map.unsqueeze(0).expand(ncaps_per_img, bs, depth, height, width).reshape(-1, depth, height, width)
        feature_map = feature_map.reshape((total_bs, depth, -1)).permute((2, 0, 1))
        feature_vector = feature_vector.unsqueeze(0).expand(ncaps_per_img, bs, length).reshape(-1, length)

        hidden0 = F.relu(self.hidden_projection0(feature_vector))
        hidden1 = F.relu(self.hidden_projection1(feature_vector))
        cell0 = F.relu(self.cell_projection0(feature_vector))
        cell1 = F.relu(self.cell_projection1(feature_vector))

        hidden = torch.stack((hidden0, hidden1), dim=0)
        cell = torch.stack((cell0, cell1), dim=0)

        y_t = y[0, :]

        for ts in range(1, max_len):
            output, attn_weights, hidden, cell ,out_fea = self.decoder(y_t, hidden, cell, feature_map, feature_map)
            outputs.append(output)
            out_feas.append(out_fea) 
            attns.append(attn_weights)
            # random.random()产生的随机数 [0.0, 1.0)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            y_t = (y[ts, :] if teacher_force else top1)
        
        return torch.stack(outputs), torch.cat(attns, dim=1),torch.stack(out_feas)
    
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
        elif optim_algo in ('sgd', 'adagrad'):
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
