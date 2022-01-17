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
