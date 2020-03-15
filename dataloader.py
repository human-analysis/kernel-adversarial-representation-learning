# dataloader.py

import torch
import datasets
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


class Dataloader:
    def __init__(self, args):
        self.args = args

        self.loader_input = args.loader_input
        self.loader_label = args.loader_label

        self.dataset_options = args.dataset_options

        self.split_test = args.split_test
        self.split_train = args.split_train
        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train

        self.train_dev_percent = args.train_dev_percent
        self.test_dev_percent = args.test_dev_percent

        self.resolution = (args.resolution_wide, args.resolution_high)

        self.input_filename_test = args.input_filename_test
        self.label_filename_test = args.label_filename_test
        self.input_filename_train = args.input_filename_train
        self.label_filename_train = args.label_filename_train


        if self.dataset_train_name == 'Gaussian':
            self.dataset_train = datasets.Gaussian(
                self.args.dataroot, train=True)

        elif self.dataset_train_name == 'YaleB':
            self.dataset_train = datasets.YaleB(
                self.args.dataroot, train=True)

        elif self.dataset_train_name == 'Adult':
            self.dataset_train = datasets.Adult(
                self.args.dataroot, train=True)

        elif self.dataset_train_name == 'German':
            self.dataset_train = datasets.German(
                self.args.dataroot, train=True)

        elif self.dataset_train_name == 'Cifar100':
            self.dataset_train = datasets.Cifar100(
                self.args.dataroot, train=True)

        else:
            raise(Exception("Unknown Dataset"))


        if self.dataset_test_name == 'Gaussian':
            self.dataset_test = datasets.Gaussian(
                self.args.dataroot, train=False)

        elif self.dataset_test_name == 'YaleB':
            self.dataset_test = datasets.YaleB(
                self.args.dataroot, train=False)

        elif self.dataset_test_name == 'Adult':
            self.dataset_test = datasets.Adult(
                self.args.dataroot, train=False)

        elif self.dataset_test_name == 'German':
            self.dataset_test = datasets.German(
            self.args.dataroot, train=False)


        elif self.dataset_test_name == 'Cifar100':
            self.dataset_test = datasets.Cifar100(
                self.args.dataroot, train=False)

        else:
            raise(Exception("Unknown Dataset"))

    def create(self, flag=None):
        dataloader = {}
        if flag == "Train_E":
            dataloader['train_e'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size_e,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )
            return dataloader

        elif flag == "Train":
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size_train,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )
            return dataloader

        elif flag == "Test":
            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size_test,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True
            )
            return dataloader

        elif flag is None:
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size_train,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )

            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size_test,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True)
            return dataloader

    def create_devsets(self, flag=None):
        dataloader = {}
        if flag is None or flag == "train":
            train_len = len(self.dataset_train)
            train_cut_index = int(train_len * self.train_dev_percent)
            train_random_indices = list(torch.randperm(train_len))
            train_indices = train_random_indices[train_cut_index:]
            train_dev_indices = train_random_indices[:train_cut_index]

            train_sampler = SubsetRandomSampler(train_indices)
            train_dev_sampler = SubsetRandomSampler(train_dev_indices)

            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=self.args.batch_size_train,
                sampler=train_sampler, num_workers=self.args.nthreads,
                pin_memory=True
            )

            dataloader['train_dev'] = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=self.args.batch_size_train,
                sampler=train_dev_sampler, num_workers=self.args.nthreads,
                pin_memory=True
            )

        if flag is None or flag == "test":
            test_len = len(self.dataset_test)
            test_cut_index = int(test_len * self.test_dev_percent)
            test_random_indices = list(torch.randperm(test_len))
            test_indices = test_random_indices[test_cut_index:]
            test_dev_indices = test_random_indices[:test_cut_index]

            test_sampler = SubsetRandomSampler(test_indices)
            test_dev_sampler = SubsetRandomSampler(test_dev_indices)

            dataloader['test_dev'] = torch.utils.data.DataLoader(
                self.dataset_test, batch_size=self.args.batch_size_test,
                sampler=test_dev_sampler, num_workers=self.args.nthreads,
                pin_memory=True
            )

            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test, batch_size=self.args.batch_size_test,
                sampler=test_sampler, num_workers=self.args.nthreads,
                pin_memory=True
            )

        return dataloader
