import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
from Seq2Seq_config import get_arguments
from utils.logger import Logger
from utils.network import *


def train(model, iterator, optimizer, criterion, logger, epoch):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        src = batch.src
        trg = batch.trg

        # Forward pass
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])  # [trg_len, batch_size, output_dim] → [(trg_len-1)*batch_size, output_dim]
        trg = trg[1:].view(-1)  # [trg_len, batch_size] → [(trg_len-1)*batch_size]
        loss = criterion(output, trg)
        epoch_loss += loss.item()
        logger.add_scalar('train_loss', loss.item(), epoch * len(iterator) + i)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(get_trainable_parameters(model), max_norm=1)
        optimizer.step()

    return epoch_loss / len(iterator)


def valid(model, iterator, criterion, logger, epoch):
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
            logger.add_scalar('valid_loss', loss.item(), epoch * len(iterator) + i)

    return epoch_loss / len(iterator)


def start_train(args, train_iterator, valid_iterator, model, optimizer, criterion):
    # 是否使用checkpoint
    if not args.train_from_zero:
        try:
            checkpoint = torch.load(args.ckpt_path)
        except FileNotFoundError:
            print(f'[*] The checkpoint file {args.ckpt_path} not exist, train start ...')
            current_epoch = 0
            best_valid_loss = float('inf')
        else:
            print(f'[*] Load model from {args.ckpt_path}, train start ...')
            model.load_state_dict(checkpoint['model'])
            best_epoch = checkpoint['best_epoch']
            current_epoch = checkpoint['last_epoch']
            best_valid_loss = checkpoint['best_valid_loss']
            if current_epoch == args.epochs or (current_epoch - best_epoch >= args.early_stopping_patience):
                print(f'[*] The model has been trained successfully! Stop from {current_epoch}th epoch!', end='\n\n')
                return
    else:
        print(f'[*] Train model from zero, train start ...')
        current_epoch = 0
        best_valid_loss = float('inf')

    # TensorboardX
    logger = Logger(args.logs_dir, flush_secs=30)
    input_example = (torch.ones(2, args.batch_size, dtype=int).to(args.device),
                     torch.ones(2, args.batch_size, dtype=int).to(args.device))
    logger.add_graph(model, input_example)
    logger.save_arguments(args)

    # 开始训练
    early_stopping_counter = 0
    for epoch in range(current_epoch, args.epochs):
        print(f'Epoch: {epoch+1:02}/{args.epochs:02} ...')

        train_loss = train(model, train_iterator, optimizer, criterion, logger, epoch)
        valid_loss = valid(model, valid_iterator, criterion, logger, epoch)
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        logger.add_scalar('train_epoch_loss', train_loss, epoch)
        logger.add_scalar('valid_epoch_loss', valid_loss, epoch)
        logger.save_trainable_parameters(model, epoch)

        if valid_loss < best_valid_loss:
            early_stopping_counter = 0
            best_valid_loss = valid_loss
            checkpoint = {'model': model.state_dict(), 'best_epoch': epoch+1, 'best_valid_loss': valid_loss, 'last_epoch': epoch+1}
            torch.save(checkpoint, args.ckpt_path)
            print(f'  Save model trained in {epoch+1}th epoch!')
        else:
            early_stopping_counter += 1
            print(f'  EarlyStopping counter: {early_stopping_counter} out of {args.early_stopping_patience}!')
            if early_stopping_counter >= args.early_stopping_patience or epoch+1 == args.epochs:
                checkpoint = torch.load(args.ckpt_path)
                checkpoint['last_epoch'] = epoch+1
                torch.save(checkpoint, args.ckpt_path)
                break

    print(f'[*] The model has been trained successfully! Stop from {epoch+1}th epoch!', end='\n\n')


if __name__ == '__main__':
    # 参数解析
    args = get_arguments()

    # 读入数据
    train_iterator, valid_iterator, _ = args.get_data_iterator()

    # 构建模型
    model = args.network(args).to(args.device)

    # 参数初始化
    globals()[f'init_parameters_{args.model_name}'](model, args)
    print_network(model)

    # 优化器与损失函数
    optimizer = optim.Adam(get_trainable_parameters(model))
    criterion = nn.CrossEntropyLoss(ignore_index=args.TRG.vocab.stoi[args.TRG.pad_token])

    # 训练模型
    start_train(args, train_iterator, valid_iterator, model, optimizer, criterion)
