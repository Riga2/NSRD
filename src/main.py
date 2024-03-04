import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)


def main():
    global model
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
