from tensorboardX import SummaryWriter


class Logger(SummaryWriter):
    def save_arguments(self, args):
        arguments = ""
        for key, value in dict(sorted(args.__dict__.items())).items():
            if str(value).startswith('<'):
                continue
            else:
                arguments += f'{key}: {value}\n'
        with open(args.args_path, 'w') as f:
            f.write(arguments.strip())

    def save_trainable_parameters(self, model, epoch):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.add_histogram(name.replace('.', '/', 1), param.data, epoch)
