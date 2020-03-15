# model.py

import math
import models
import losses
import evaluate
from torch import nn
import config

def weights_init(m):
    args = config.parse_args()
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, args.variance)
        m.bias.data.zero_()


class Model:
    def __init__(self, args):
        self.ngpu = args.ngpu
        self.device = args.device

        # targ-adv
        self.model_type_A = args.model_type_a
        self.model_type_T = args.model_type_t
        ###########################################
        # targ-adv
        self.model_options_A = args.model_options_A
        self.model_options_T = args.model_options_T
        ############################################

        self.loss_type_A = args.loss_type_a
        self.loss_type_T = args.loss_type_t
        #############################################

        self.loss_options_A = args.loss_options_A
        self.loss_options_T = args.loss_options_T
        ################################################

        self.evaluation_type_A = args.evaluation_type_a
        self.evaluation_type_T = args.evaluation_type_t
        #######################################################

        self.evaluation_options_A = args.evaluation_options_A
        self.evaluation_options_T = args.evaluation_options_T
        # import pdb
        # pdb.set_trace()

    def setup(self, checkpoints):


        model_A = getattr(models, self.model_type_A)(**self.model_options_A)
        model_T = getattr(models, self.model_type_T)(**self.model_options_T)
        ######################################################################

        criterion_A = getattr(losses, self.loss_type_A)(**self.loss_options_A)
        criterion_T = getattr(losses, self.loss_type_T)(**self.loss_options_T)
        #######################################################################

        evaluation_A = getattr(evaluate, self.evaluation_type_A)(
            **self.evaluation_options_A)
        evaluation_T = getattr(evaluate, self.evaluation_type_T)(
            **self.evaluation_options_T)

        if self.ngpu > 1:
            #####################################################################

            model_A = nn.DataParallel(model_A, device_ids=list(range(self.ngpu)))
            model_T = nn.DataParallel(model_T, device_ids=list(range(self.ngpu)))


        model_A = model_A.to(self.device)
        model_T = model_T.to(self.device)

        criterion_A = criterion_A.to(self.device)
        criterion_T = criterion_T.to(self.device)

        if checkpoints.latest('resume') is None:

            model_A.apply(weights_init)
            model_T.apply(weights_init)

            # pass

        else:

            model_A = checkpoints.load(model_A, checkpoints.latest('resume'))
            model_T = checkpoints.load(model_T, checkpoints.latest('resume'))

        model ={}

        model['Adversary'] = model_A
        model['Target'] = model_T

        criterion = {}

        criterion['Adversary'] = criterion_A
        criterion['Target'] = criterion_T

        evaluation = {}
        evaluation['Adversary'] = evaluation_A
        evaluation['Target'] = evaluation_T

        return model, criterion, evaluation
