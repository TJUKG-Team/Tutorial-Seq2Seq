import torch
from Seq2Seq_config import get_arguments
from utils.network import *


def inference(model, input_sentence):
    model.eval()
    with torch.no_grad():
        input_indexes = args.encode_src_sentence(input_sentence).to(args.device)  # [src_len, batch_size=1]
        trg_indexes = args.get_sos_eos_indexes().to(args.device)  # [[stoi('<sos>')], [stoi('<eos>')]] = [[2], [3]]
        output_indexes = model(input_indexes, trg_indexes)
        output_indexes = output_indexes.squeeze(1).argmax(1)  # [trg_len, batch_size=1, output_dim] → [trg_len]
        output_sentence = args.decode_trg_sentence(output_indexes)
    return output_sentence


def start_inference(args, model):
    try:
        checkpoint = torch.load(args.ckpt_path)
    except FileNotFoundError:
        print(f'[*] The checkpoint file {args.ckpt_path} not exist, inference aborted!')
    else:
        print(f'[*] Load model from {args.ckpt_path}, inference start ...')
        model.load_state_dict(checkpoint['model'])
        while True:
            # ein schwarzer hund und ein gefleckter hund kämpfen. --> a black dog and a spotted dog are fighting
            # Ein Hund rennt im Schnee. --> A dog is running in the snow
            # eine frau spielt ein lied auf ihrer geige --> a female playing a song on her violin
            # die person im gestreiften shirt klettert auf einen berg --> the person in the striped shirt is mountain climbing
            input_sentence = input('source sentence: ')
            if input_sentence == 'exit':
                break
            output_sentence = inference(model, input_sentence)
            print(f'output sentence: {output_sentence}\n', end='-'*80+'\n')


if __name__ == '__main__':
    # 参数解析
    args = get_arguments()

    # 构建模型
    model = args.network_inference(args).to(args.device)
    print_network(model)

    # 推理模型
    start_inference(args, model)
