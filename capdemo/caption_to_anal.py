import torch
import os.path as oph

#from .caption_zh.misc.decoder import Decoder


device = torch.device('cpu')
rel_dir = oph.dirname(__file__)


'''def get_model_zh():
    from .caption_zh.misc.multimodal_rnn import MultimodalRNN
    """ 从checkpoint读取参数，构建模型，中文的， 返回模型和字典的元组 """
    path = oph.join(rel_dir, 'caption_zh/model/best_model.pth')
    saved = torch.load(open(path, 'rb'), map_location=device)
    data_source = saved['data_source']
    vocab = data_source['vocab']
    decoder_cfg = saved['decoder_cfg']

    decoder = Decoder(len(vocab), decoder_cfg['emb_dim'], decoder_cfg['hid_dim'], decoder_cfg['n_layers'], decoder_cfg['dropout'])
    model = MultimodalRNN(decoder)
    model.load_state_dict(saved['state_dict'])
    model.to(device)

    return model, vocab'''

def get_model_en():
    from caption_en.misc.multimodal_rnn_to_text import MultimodalRNN
    """ 从checkpoint读取参数，构建模型，英文的， 返回模型和字典的元组 """
    path = oph.join(rel_dir, 'caption_en/model/best_model.pth')
    saved = torch.load(open(path, 'rb'), map_location=device)
    data_source = saved['data_source']
    vocab = data_source['vocab']
    decoder_cfg = saved['decoder_cfg']

    model = MultimodalRNN(len(vocab), decoder_cfg['emb_dim'], decoder_cfg['hid_dim'], decoder_cfg['dropout'])
    
    model.load_state_dict(saved['state_dict'])
    model.to(device)

    return model, vocab

#model_zh, vocab_zh = get_model_zh()
model_en, vocab_en = get_model_en()

'''def cap_gen_zh(x, model, vocab, max_iters:int=15):
    """  """
    model.eval()

    eos_idx = vocab.stoi['<eos>']
    y = torch.empty((max_iters + 2, 1), dtype=torch.long, device=device)
    y[0, 0] = eos_idx

    with torch.no_grad():
        # outputs (max_iters + 1, bs, output_dim)
        outputs = model(x, y, -1.0)
        outputs = outputs.max(dim=-1)[1]

        tokens = []
        for k in range(max_iters + 1):
            if outputs[k][0] == eos_idx:
                break
            tokens.append(vocab.itos[outputs[k][0]])
    return ''.join(tokens)'''

def cap_gen_en(x, model, vocab, max_iters:int=15):
    """  """
    model.eval()

    eos_idx = vocab.stoi['<eos>']
    y = torch.empty((max_iters + 2, 1), dtype=torch.long, device=device)
    y[0, 0] = eos_idx

    with torch.no_grad():
        # outputs (max_iters + 1, bs, output_dim)
        outputs, _ , out_feas = model(x, y, -1.0)
        outputs = outputs.max(dim=-1)[1]

        tokens = []
        
        for k in range(max_iters + 1):
           if outputs[k][0] == eos_idx:
               break
           tokens.append(vocab.itos[outputs[k][0]])
    return ' '.join(tokens), out_feas


#l(17)*b(1)*w(512)

def cap_example(x):
    """ 读入图片张量，返回成对的中英文描述 """
    x = x.to(device).unsqueeze(0)
    #caption_zh = cap_gen_zh(x, model_zh, vocab_zh, 15)
    caption_en , out_att = cap_gen_en(x, model_en, vocab_en, 15)
    rst = {
        #'caption_zh': caption_zh,
        'caption_en': caption_en
    }
    return caption_en,out_att


