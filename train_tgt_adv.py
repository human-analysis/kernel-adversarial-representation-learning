# train.py

import time
import torch
import torch.optim as optim
import plugins
import losses
from train_encr import kernel_Gaussian
from train_encr import kernel_poly


class Trainer:
    def __init__(self, args, model, criterion, evaluation, lam):
        # self.E_Gaussian = losses.encoder_Gaussian()
        self.args = args
        self.lam = lam
        self.r = args.r
        self.model_A = model['Adversary']
        self.model_T = model['Target']
        self.criterion_A = criterion['Adversary']
        self.criterion_T = criterion['Target']

        self.evaluation_A = evaluation['Adversary']
        self.evaluation_T = evaluation['Target']

        self.save_results = args.save_results

        if args.kernel == 'Gaussian':
            self.kernel = kernel_Gaussian()

        elif args.kernel == 'Polynomial':
            self.kernel = kernel_poly()


        self.env = args.env
        self.port = args.port
        self.dir_save = args.save_dir
        self.log_type = args.log_type

        self.device = args.device
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size_train

        self.lr_a = args.learning_rate_a
        self.lr_t = args.learning_rate_t

        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method
        self.scheduler_options = args.scheduler_options

        # import pdb
        # pdb.set_trace()


        self.optimizer_A = getattr(optim, self.optim_method)(
            filter(lambda p: p.requires_grad, self.model_A.parameters()),
            lr=self.lr_a, **self.optim_options)
        self.optimizer_T = getattr(optim, self.optim_method)(
            filter(lambda p: p.requires_grad, self.model_T.parameters()),
            lr=self.lr_t, **self.optim_options)
        if self.scheduler_method is not None:
            self.scheduler_A = getattr(optim.lr_scheduler, self.scheduler_method)(
                self.optimizer_A, **self.scheduler_options
            )
            self.scheduler_T = getattr(optim.lr_scheduler, self.scheduler_method)(
                self.optimizer_T, **self.scheduler_options
            )



        # for classification
        self.labels = torch.zeros(
            self.batch_size,
            dtype=torch.long,
            device=self.device
        )
        self.sensitives = torch.zeros(
            self.batch_size,
            dtype=torch.long,
            device=self.device
        )
        self.inputs = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )

        # import pdb; pdb.set_trace()
        # logging training
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TrainLogger_%.4f.txt' %self.lam,
            self.save_results
        )


        self.params_loss = ['Loss_A', 'Loss_T', 'Accuracy_A', 'Accuracy_T']
        self.log_loss.register(self.params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Loss_A': {'dtype': 'running_mean'},
            'Loss_T': {'dtype': 'running_mean'},
            'Accuracy_A': {'dtype': 'running_mean'},
            'Accuracy_T': {'dtype': 'running_mean'},
           }
        self.visualizer = plugins.Visualizer(self.port, self.env, 'Train')
        self.params_visualizer = {
            'Loss_A': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_A',
                            'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Loss_T': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_T',
                            'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Accuracy_A': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'accuracy_A',
                                'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Accuracy_T': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'accuracy_T',
                                'layout': {'windows': ['train', 'test'], 'id': 0}},
        }

        self.monitor.register(self.params_monitor)
        self.visualizer.register(self.params_visualizer)

        if self.log_type == 'traditional':
            # display training progress
            self.print_formatter = 'Train [%d/%d][%d/%d] '
            for item in self.params_loss:
                self.print_formatter += item + " %.4f "
        elif self.log_type == 'progressbar':
            # progress bar message formatter
            self.print_formatter = '({}/{})'

            for item in self.params_loss:
                self.print_formatter += '| ' + item + ' {:.4f}'
            self.print_formatter += '| lr: {:.2e}'
            self.print_formatter += '| lam: {:.4f}'
            self.print_formatter += '| Kernel: {:7s}'

        self.evalmodules = []
        self.losses = {}




        self.model_A.train()
        self.model_T.train()


    def train(self, epoch, dataloader, theta, X_old):
        dataloader = dataloader['train']

        self.monitor.reset()
        # switch to train mode


        if self.log_type == 'progressbar':
            # Progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<2}'.format('Train'), max=len(dataloader))
        end = time.time()

        if self.args.kernel != 'Linear':
            n = X_old.shape[0]
            D = torch.eye(n) - torch.ones(n) / n

        for i, (inputs, labels, sensitives) in enumerate(dataloader):

            # keeps track of data loading time
            data_time = time.time() - end

            ############################
            # Update network
            ############################
            # inputs= inputs.float()
            batch_size = inputs.size(0)

            # self.inputs.resize_(inputs.size()).copy_(inputs)
            self.labels.resize_(labels.size()).copy_(labels)
            self.sensitives.resize_(sensitives.size()).copy_(sensitives)
            # self.labels = self.labels.float()



            if self.args.kernel == 'Linear':
                self.outputs_E = torch.mm(inputs, torch.t(theta)).to(self.device)

            elif self.args.kernel == 'Gaussian':
                K = self.kernel(X_old, inputs, self.args.sigma)
                self.outputs_E = torch.mm(torch.mm(torch.t(K), D), torch.t(theta)).to(self.device)

            elif self.args.kernel == 'Polynomial':
                K = self.kernel(X_old, inputs, self.args.c, self.args.d)
                # import pdb; pdb.set_trace()
                self.outputs_E = torch.mm(torch.mm(torch.t(K), D), torch.t(theta)).to(self.device)
                # self.outputs_E = torch.mm(torch.t(K)-torch.mean(torch.t(K), dim=1), torch.t(theta)).to(self.device)


            outputs_A = self.model_A(self.outputs_E.float())
            outputs_T = self.model_T(self.outputs_E.float())

            if self.args.dataset_train == 'YaleB':
                Y = torch.zeros(batch_size, self.args.nclasses_t).scatter_(1, self.labels.long().cpu(), 1).to(self.device)
                S = torch.zeros(batch_size, self.args.nclasses_a).scatter_(1, self.sensitives.long().cpu(), 1).to(self.device)

                # import pdb; pdb.set_trace()

                loss_A = self.criterion_A(outputs_A.squeeze(), S)
                loss_T = self.criterion_T(outputs_T.squeeze(), Y)

            else:
                loss_A = self.criterion_A(outputs_A.squeeze(), self.sensitives.squeeze())
                loss_T = self.criterion_T(outputs_T.squeeze(), self.labels.squeeze())





            self.optimizer_A.zero_grad()
            self.optimizer_T.zero_grad()
            loss_A.backward(retain_graph=True)
            self.optimizer_A.step()

            self.optimizer_A.zero_grad()
            self.optimizer_T.zero_grad()
            loss_T.backward()
            self.optimizer_T.step()
            # loss = (1 - self.lam) * loss_T - self.lam * loss_A




            acc_A = self.evaluation_A(outputs_A, self.sensitives)
            acc_T = self.evaluation_T(outputs_T, self.labels)
            # import pdb; pdb.set_trace()

            acc_A = acc_A[0].item()
            acc_T = acc_T[0].item()

            # loss = loss.item()


            loss_A = loss_A.item()
            loss_T = loss_T.item()


            # self.losses['Loss_test'] = loss
            self.losses['Loss_A'] = loss_A
            self.losses['Loss_T'] = loss_T
            self.losses['Accuracy_A'] = acc_A
            self.losses['Accuracy_T'] = acc_T
            # import pdb
                # pdb.set_trace()

                # self.losses['embedding'] = acc_A


            self.monitor.update(self.losses, batch_size)


            if self.log_type == 'traditional':
                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i+1, len(dataloader)] +
                    +[self.losses[key] for key in self.params_monitor]))
            elif self.log_type == 'progressbar':
                # update progress bar
                batch_time = time.time() - end
                processed_data_len += len(inputs)

                bar.suffix = self.print_formatter.format(
                    # *[processed_data_len, len(dataloader.sampler), data_time,
                    #   batch_time, bar.elapsed_td, bar.eta_td]
                    *[processed_data_len, len(dataloader.sampler)]
                     +[self.losses[key] for key in self.params_monitor]
                     +[self.optimizer_A.param_groups[-1]['lr']]
                    +[self.lam]
                    +[self.args.kernel]
                )
                bar.next()
                end = time.time()

        #
            if self.log_type == 'progressbar':
                bar.finish()


        loss = self.monitor.getvalues()
        # import pdb; pdb.set_trace()
        self.log_loss.update(loss)


        # if self.encoder ==True:
        #     if self.args.adverserial_type == 'closed-form':
        #         if  epoch%20==0:
        #             loss['embedding_A'] = [self.outputs_E.detach().cpu().numpy(), self.sensitives.detach().cpu().numpy(), self.labels.cpu().detach().cpu().numpy()]
        #             loss['embedding_T'] = [self.outputs_E.detach().cpu().numpy(), self.sensitives.detach().cpu().numpy(), self.labels.cpu().detach().cpu().numpy()]


        self.visualizer.update(loss)

        if self.scheduler_method is not None:
            if self.scheduler_method == 'ReduceLROnPlateau':
                self.scheduler_A.step(loss['Loss'])
                self.scheduler_T.step(loss['Loss'])
            else:
                self.scheduler_A.step()
                self.scheduler_T.step()

        return loss,  self.inputs
