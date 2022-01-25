import argparse
import torch
import importlib
import os


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='LSTM')
    parser.add_argument('--dataset', type=str, default='DailyDialog')
    args, _ = parser.parse_known_args()

    ckpt_path = f'checkpoints/{args.dataset}/{args.model_name}.pth'  # checkpoints保存路径
    args_path = f'checkpoints/{args.dataset}/arguments/{args.model_name}.yaml'  # 参数保存路径
    logs_dir = f'logs/{args.dataset}/{args.model_name}'  # logs保存路径
    parser = globals()[f'get_parser_{args.model_name}'](parser, ckpt_path, args_path, logs_dir)
    args = parser.parse_args()

    # 创建ckpt/args/logs目录
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 配置运行环境
    args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')

    # 导入数据集
    import_data = importlib.import_module(f'datasets.{args.dataset}')
    dataset = getattr(import_data, args.dataset)(args.batch_size, args.device)
    args.SRC, args.TRG = dataset.get_Fields()
    args.get_data_iterator = dataset.get_data_iterator
    args.encode_src_sentence = dataset.encode_src_sentence  # seq2index
    args.get_sos_eos_indexes = dataset.get_sos_eos_indexes  # 获取<sos>和<eos>的index
    args.decode_trg_sentence = dataset.decode_trg_sentence  # index2seq

    # 导入模型
    import_model = importlib.import_module(f'models.{args.model_name}')
    args.network = getattr(import_model, args.model_name)
    args.network_inference = getattr(import_model, f'{args.model_name}_inference')

    return args


def get_parser_common(parser, ckpt_path, args_path, logs_dir):
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--train_from_zero', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=f'{ckpt_path}')
    parser.add_argument('--args_path', type=str, default=f'{args_path}')
    parser.add_argument('--logs_dir', type=str, default=f'{logs_dir}')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    return parser


def get_parser_LSTM(parser, ckpt_path, args_path, logs_dir):
    get_parser_common(parser, ckpt_path, args_path, logs_dir)
    parser.add_argument('--enc_emb_dim', type=int, default=300)
    parser.add_argument('--dec_emb_dim', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=512)
    parser.add_argument('--enc_dropout', type=float, default=0.5)
    parser.add_argument('--dec_dropout', type=float, default=0.5)
    return parser


def get_parser_GRU(parser, ckpt_path, args_path, logs_dir):
    get_parser_common(parser, ckpt_path, args_path, logs_dir)
    parser.add_argument('--enc_emb_dim', type=int, default=300)
    parser.add_argument('--dec_emb_dim', type=int, default=300)
    parser.add_argument('--hid_dim', type=int, default=512)
    parser.add_argument('--enc_dropout', type=float, default=0.5)
    parser.add_argument('--dec_dropout', type=float, default=0.5)
    return parser


def get_parser_GRU_Attention(parser, ckpt_path, args_path, logs_dir):
    get_parser_common(parser, ckpt_path, args_path, logs_dir)
    parser.add_argument('--enc_emb_dim', type=int, default=300)
    parser.add_argument('--dec_emb_dim', type=int, default=300)
    parser.add_argument('--enc_hid_dim', type=int, default=512)
    parser.add_argument('--dec_hid_dim', type=int, default=512)
    parser.add_argument('--enc_dropout', type=float, default=0.5)
    parser.add_argument('--dec_dropout', type=float, default=0.5)
    return parser
