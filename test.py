# test.py

import time
import torch
import plugins
from train_encr import kernel_Gaussian
from train_encr import kernel_poly


class Tester:
    def __init__(self, args, model, criterion, evaluation, lam):
        self.args = args
        self.kernel = args.kernel
        self.lam = lam


        if args.kernel == 'Gaussian':
            self.kernel = kernel_Gaussian()

        if args.kernel == 'Polynomial':
            self.kernel = kernel_poly()

        self.model_A = model['Adversary']
        self.model_T = model['Target']

        self.criterion_A = criterion['Adversary']
        self.criterion_T = criterion['Target']


        self.evaluation_A = evaluation['Adversary']
        self.evaluation_T = evaluation['Target']
        self.save_results = args.save_results

        self.env = args.env
        self.port = args.port
        self.dir_save = args.save_dir
        self.log_type = args.log_type

        self.device = args.device
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size_test


        # for classification
        self.labels = torch.zeros(
            self.batch_size,
            dtype=torch.long,
            device=self.device
        )
        self.inputs = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )
        self.sensitives = torch.zeros(
            self.batch_size,
            dtype=torch.long,
            device=self.device
        )

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger_%4f.txt' %self.lam,
            self.save_results
        )

        self.params_loss = ['Loss_A', 'Loss_T', 'Accuracy_A', 'Accuracy_T']
        self.log_loss.register(self.params_loss)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Loss_A': {'dtype': 'running_mean'},
            'Loss_T': {'dtype': 'running_mean'},
            'Accuracy_A': {'dtype': 'running_mean'},
            'Accuracy_T': {'dtype': 'running_mean'}
        }

        self.visualizer = plugins.Visualizer(self.port, self.env, 'Test')
        self.params_visualizer = {
            'Loss_A': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_A',
                            'layout': {'windows': ['train', 'test'], 'id': 1}},
            'Accuracy_A': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'accuracy_A',
                                'layout': {'windows': ['train', 'test'], 'id': 1}},
            'Loss_T': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_T',
                            'layout': {'windows': ['train', 'test'], 'id': 1}},
            'Accuracy_T': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'accuracy_T',
                                'layout': {'windows': ['train', 'test'], 'id': 1}}
        }

        self.monitor.register(self.params_monitor)
        self.visualizer.register(self.params_visualizer)

        if self.log_type == 'traditional':
            # display training progress
            self.print_formatter = 'Test [%d/%d][%d/%d] '
            for item in self.params_loss:
                self.print_formatter += item + " %.4f "
        elif self.log_type == 'progressbar':
            # progress bar message formatter
            self.print_formatter = '({}/{})'
            for item in self.params_loss:
                self.print_formatter += ' |' + item + ' {:.4f}'
            self.print_formatter += ' | lam: {:.4f}'

        self.evalmodules = []
        self.losses = {}

    def model_eval(self):

        self.model_A.eval()
        self.model_T.eval()

    def test(self, epoch, dataloader, theta, X_old):
        # import pdb
        # pdb.set_trace()
        dataloader = dataloader['test']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()

        if self.log_type == 'progressbar':
            # progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<2}'.format('Test'), max=len(dataloader))

        end = time.time()

        if self.args.kernel != 'Linear':
            n = X_old.shape[0]
            D = torch.eye(n) - torch.ones(n) / n

        for i, (inputs, labels, sensitives) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end

            # import pdb
            # pdb.set_trace()
            ############################
            # Evaluate Network
            ############################

            batch_size = inputs.size(0)
            # self.inputs.resize_(inputs.size()).copy_(inputs)
            self.labels.resize_(labels.size()).copy_(labels)
            self.sensitives.resize_(labels.size()).copy_(sensitives)
            # inputs = inputs.float()

            if self.args.kernel == 'Linear':
                self.output_E = torch.mm(inputs, torch.t(theta)).to(self.device)

            elif self.args.kernel == 'Gaussian':
                K = self.kernel(X_old, inputs, self.args.sigma)
                self.output_E = torch.mm(torch.mm(torch.t(K), D), torch.t(theta)).to(self.device)

            elif self.args.kernel == 'Polynomial':
                K = self.kernel(X_old, inputs, self.args.c, self.args.d)
                self.output_E = torch.mm(torch.mm(torch.t(K), D), torch.t(theta)).to(self.device)


            output_A = self.model_A(self.output_E.float())
            output_T = self.model_T(self.output_E.float())

            if self.args.dataset_train == 'YaleB':
                Y = torch.zeros(batch_size, self.args.nclasses_t).scatter_(1, self.labels.long().cpu(), 1).to(self.device)
                S = torch.zeros(batch_size, self.args.nclasses_a).scatter_(1, self.sensitives.long().cpu(), 1).to(self.device)

                loss_A = self.criterion_A(output_A.squeeze(), S)
                loss_T = self.criterion_T(output_T.squeeze(), Y)

            else:
                loss_A = self.criterion_A(output_A.squeeze(), self.sensitives.squeeze())
                loss_T = self.criterion_T(output_T.squeeze(), self.labels.squeeze())


            self.model_A.zero_grad()
            self.model_T.zero_grad()


            acc_A = self.evaluation_A(output_A, self.sensitives)
            acc_T = self.evaluation_T(output_T, self.labels)

            acc_A = acc_A[0].item()
            acc_T = acc_T[0].item()
            loss_A = loss_A.item()
            loss_T = loss_T.item()

            self.losses['Accuracy_A'] = acc_A
            self.losses['Accuracy_T'] = acc_T
            self.losses['Loss_A'] = loss_A
            self.losses['Loss_T'] = loss_T
            self.monitor.update(self.losses, batch_size)

            if self.log_type == 'traditional':
                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i+1, len(dataloader)] +
                    [self.losses[key] for key in self.params_monitor]))
            elif self.log_type == 'progressbar':
                # update progress bar
                batch_time = time.time() - end
                processed_data_len += len(inputs)
                bar.suffix = self.print_formatter.format(
                    # *[processed_data_len, len(dataloader.sampler), data_time,
                    #   batch_time, bar.elapsed_td, bar.eta_td] +
                    *[processed_data_len, len(dataloader.sampler)]+
                     [self.losses[key] for key in self.params_monitor]
                    +[self.lam]
                )
                bar.next()
                end = time.time()

        if self.log_type == 'progressbar':
            bar.finish()

        loss = self.monitor.getvalues()
        self.log_loss.update(loss)


        self.visualizer.update(loss)
        # return self.monitor.getvalues('Loss_A'), self.monitor.getvalues('Loss_T'), self.monitor.getvalues('Accuracy_A'), self.monitor.getvalues('Accuracy_T')
