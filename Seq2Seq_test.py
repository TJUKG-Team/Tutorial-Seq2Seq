import torch
import torch.nn as nn

from Seq2Seq_config import get_arguments
from utils.network import *


def test(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # Forward pass
            output = model(src, trg, teacher_forcing_ratio=0)  # turn off teacher_forcing
            output = output[1:].view(-1, output.shape[-1])  # [trg_len, batch_size, output_dim] → [(trg_len-1)*batch_size, output_dim]
            trg = trg[1:].view(-1)  # [trg_len, batch_size] → [(trg_len-1)*batch_size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def start_test(args, test_iterator, model, criterion):
    try:
        checkpoint = torch.load(args.ckpt_path)
    except FileNotFoundError:
        print(f'[*] The checkpoint file {args.ckpt_path} not exist, test aborted!')
    else:
        print(f'[*] Load model from {args.ckpt_path}, test start ...')
        model.load_state_dict(checkpoint['model'])
        test_loss = test(model, test_iterator, criterion)
        best_epoch, best_valid_loss = checkpoint['best_epoch'], checkpoint['best_valid_loss']
        print(f'Trained epochs: {best_epoch:02} | Valid Loss: {best_valid_loss:.3f} | Test Loss: {test_loss:.3f}', end='\n\n')


if __name__ == '__main__':
    # 参数解析
    args = get_arguments()

    # 读入数据
    _, _, test_iterator = args.get_data_iterator()

    # 构建模型
    model = args.network(args).to(args.device)
    print_network(model)

    # 损失函数或其他评价指标
    criterion = nn.CrossEntropyLoss(ignore_index=args.TRG.vocab.stoi[args.TRG.pad_token])

    # 测试模型
    start_test(args, test_iterator, model, criterion)
