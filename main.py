# main.py

import sys
import traceback
import torch
import random
import config
import utils
from model import Model

from train_tgt_adv import Trainer
from train_encr import train_e
from test import Tester

from dataloader import Dataloader
from checkpoints import Checkpoints


def main():
    # import pdb
    # pdb.set_trace()
    # parse the arguments
    args = config.parse_args()
    if (args.ngpu > 0 and torch.cuda.is_available()):
        device = "cuda:0"
    else:
        device = "cpu"
    args.device = torch.device(device)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.save_results:
        utils.saveargs(args)

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    for lam in args.lambd:
        models = Model(args)

        model, criterion, evaluation = models.setup(checkpoints)

        # Data Loading
        dataloader = Dataloader(args)
        loaders_train_e = dataloader.create("Train_E")
        loaders_train = dataloader.create("Train")
        loaders_test = dataloader.create("Test")

        trainer_train = train_e ()
        trainer_test = Trainer (args, model, criterion, evaluation, lam)
        tester_test = Tester (args, model, criterion, evaluation, lam)

        # start training !!!

        theta, inputs = trainer_train(args, loaders_train_e, lam)

        for epoch in range(int(args.nepochs)):
            print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

            loss_train_E, input_train = trainer_test.train(epoch, loaders_train, theta, inputs)
            with torch.no_grad():
                loss_test_E = tester_test.test(epoch, loaders_test, theta, inputs)

if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        utils.cleanup()
