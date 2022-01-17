import torch.nn as nn
import sys


def print_network(model):
    parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(model)
    print(f'The model has {parameters_num:,} trainable parameters.')


def get_init_emb_layers(dataset):
    init_emb_layers = []
    if dataset in ['Multi30k']:
        init_emb_layers = ['decoder.embedding.weight']
    elif dataset in ['DailyDialog']:
        init_emb_layers = ['encoder.embedding.weight', 'decoder.embedding.weight']
    else:
        print('Unknow Dataset, train aborted!')
        sys.exit()
    return init_emb_layers


def init_parameters_LSTM(model, args):
    weights = {'encoder.embedding.weight': args.SRC.vocab.vectors,
               'decoder.embedding.weight': args.TRG.vocab.vectors}
    init_emb_layers = get_init_emb_layers(args.dataset)

    for name, param in model.named_parameters():
        if name in init_emb_layers:
            param.data.copy_(weights[name])
        else:
            nn.init.uniform_(param.data, -0.08, 0.08)


def init_parameters_GRU(model, args):
    weights = {'encoder.embedding.weight': args.SRC.vocab.vectors,
               'decoder.embedding.weight': args.TRG.vocab.vectors}
    init_emb_layers = get_init_emb_layers(args.dataset)

    for name, param in model.named_parameters():
        if name in init_emb_layers:
            param.data.copy_(weights[name])
        else:
            nn.init.normal_(param.data, mean=0, std=0.01)


def init_parameters_GRU_Attention(model, args):
    weights = {'encoder.embedding.weight': args.SRC.vocab.vectors,
               'decoder.embedding.weight': args.TRG.vocab.vectors}
    init_emb_layers = get_init_emb_layers(args.dataset)

    for name, param in model.named_parameters():
        if name in init_emb_layers:
            param.data.copy_(weights[name])
        elif 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
