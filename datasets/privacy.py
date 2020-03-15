import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np

import config
import losses
import utils



class Gaussian(Dataset):

    def __init__(self, root, train=True,
                 transform=None):
        if train:
            D_train = sio.loadmat(root+'D_train.mat')
            D = torch.from_numpy(D_train['Data_train_n']).float()

            self.X = torch.t(D[0:3,:])
            self.X = self.X / torch.max(self.X)
            self.Y = torch.t(D[3:4, :])
            self.S = torch.t(D[4:5, :])


        else:
            D_test = sio.loadmat(root + 'D_test.mat')
            D = torch.from_numpy(D_test['Data_test_n']).float()

            self.X = torch.t(D[0:3,:])
            self.X = self.X / torch.max(self.X)
            self.Y = torch.t(D[3:4, :])
            self.S = torch.t(D[4:5, :])


    def __len__(self):
            return len(self.X)

    def __getitem__(self, index):
        X, Y, S = self.X[index], self.Y[index], self.S[index]
        return X, Y, S


class YaleB(Dataset):

    def __init__(self, root, train=True,
                 transform=None):
        self.data = sio.loadmat(root + 'data.mat')

        if train:

            # data = self.data['D_train']
            data = self.data['D_test']

            X = torch.from_numpy(data[0:504, :]).float()
            Y = torch.from_numpy(data[504:505, :])
            S = torch.from_numpy(data[505:506, :])

            self.X = torch.t(X)
            self.X = self.X / torch.max(self.X)
            # self.X = self.X / torch.max(self.X, dim=0)
            self.Y = torch.t(Y)
            self.S = torch.t(S)
            # import pdb
            # pdb.set_trace()

        else:
            # data = self.data['D_test']
            data = self.data['D_train']
            # import pdb;
            # pdb.set_trace()

            X = torch.from_numpy(data[0:504, :]).float()
            Y = torch.from_numpy(data[504:505, :])
            S = torch.from_numpy(data[505:506, :])

            self.X = torch.t(X)
            self.X = self.X / torch.max(self.X)
            # self.X = self.X / torch.norm(self.X, dim=0)
            self.Y = torch.t(Y)
            self.S = torch.t(S)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, Y, S = self.X[index], self.Y[index], self.S[index]
        return X, Y, S

class Adult(Dataset):

    def __init__(self, root, train=True,
                 transform=None):
        self.data = sio.loadmat(root + 'data.mat')
        if train:
            X = torch.from_numpy(self.data['X']).float()
            Y = torch.from_numpy(self.data['Y'])
            S = torch.from_numpy(self.data['S'])

            self.X = torch.t(X)
            self.Y = torch.t(Y)
            self.S = torch.t(S)
            # import pdb
            # pdb.set_trace()

        else:
            X = torch.from_numpy(self.data['X_test']).float()
            Y = torch.from_numpy(self.data['Y_test'])
            S = torch.from_numpy(self.data['S_test'])

            self.X = torch.t(X)
            self.Y = torch.t(Y)
            self.S = torch.t(S)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, Y, S = self.X[index], self.Y[index], self.S[index]
        return X, Y, S

class German(Dataset):

    def __init__(self, root, train=True,
                 transform=None):
        args = config.parse_args()
        self.data = sio.loadmat(root + 'data.mat')
        self.data = np.transpose(self.data['Data'])
        # import pdb
        # pdb.set_trace()
        np.random.seed(args.german_seed)
        np.random.shuffle((self.data))

        X = torch.from_numpy(self.data[:, 0:24]).float()
        X = X / torch.max(X)
        Y = torch.from_numpy(self.data[:, 24:25])
        S = torch.from_numpy(self.data[:, 25:26])


        if train:

            self.X = X[0:700, :]
            self.Y = Y[0:700, :]
            self.S = S[0:700, :]


        else:
            self.X = X[700:, :]
            self.Y = Y[700:, :]
            self.S = S[700:, :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, Y, S = self.X[index], self.Y[index], self.S[index]
        return X, Y, S

class Cifar100(Dataset):

    def __init__(self, root, train=True,
                 transform=None):

        if train:
            self.data = sio.loadmat(root + 'cifar-100-train.mat')

            X = torch.from_numpy(self.data['features']).float()
            # X = X[1:-1:4, :]


            X = X / torch.max(X)
            S = torch.from_numpy(self.data['labels'])[0]
            # S = S[1:-1:4]

            Y = torch.zeros(X.shape[0])

            Y [np.isin(S.numpy(), [4, 30, 55, 72, 95])]  = 0
            Y [np.isin(S.numpy(), [1, 32, 67, 73, 91])]  = 1
            Y [np.isin(S.numpy(), [54, 62, 70, 82, 92])] = 2
            Y [np.isin(S.numpy(), [9, 10, 16, 28, 61])]  = 3
            Y [np.isin(S.numpy(), [0, 51, 53, 57, 83])]  = 4
            Y [np.isin(S.numpy(), [22, 39, 40, 86, 87])] = 5
            Y [np.isin(S.numpy(), [5, 20, 25, 84, 94])]  = 6
            Y [np.isin(S.numpy(), [6, 7, 14, 18, 24])]   = 7
            Y [np.isin(S.numpy(), [3, 42, 43, 88, 97])]  = 8
            Y [np.isin(S.numpy(), [12, 17, 37, 68, 76])] = 9
            Y [np.isin(S.numpy(), [23, 33, 49, 60, 71])] = 10
            Y [np.isin(S.numpy(), [15, 19, 21, 31, 38])] = 11
            Y [np.isin(S.numpy(), [34, 63, 64, 66, 75])] = 12
            Y [np.isin(S.numpy(), [26, 45, 77, 79, 99])] = 13
            Y [np.isin(S.numpy(), [2, 11, 35, 46, 98])]  = 14
            Y [np.isin(S.numpy(), [27, 29, 44, 78, 93])] = 15
            Y [np.isin(S.numpy(), [36, 50, 65, 74, 80])] = 16
            Y [np.isin(S.numpy(), [47, 52, 56, 59, 96])] = 17
            Y [np.isin(S.numpy(), [8, 13, 48, 58, 90])]  = 18
            Y [np.isin(S.numpy(), [41, 69, 81, 85, 89])] = 19



            self.X = X
            self.Y = Y.unsqueeze(1)
            self.S = S.unsqueeze(1)
            # import pdb
            # pdb.set_trace()

        else:
            self.data = sio.loadmat(root + 'cifar-100-test.mat')
            X = torch.from_numpy(self.data['features']).float()
            X = X / torch.max(X)
            S = torch.from_numpy(self.data['labels'])[0]
            Y = torch.zeros(X.shape[0])

            Y [np.isin(S.numpy(), [4, 30, 55, 72, 95])]  = 0
            Y [np.isin(S.numpy(), [1, 32, 67, 73, 91])]  = 1
            Y [np.isin(S.numpy(), [54, 62, 70, 82, 92])] = 2
            Y [np.isin(S.numpy(), [9, 10, 16, 28, 61])]  = 3
            Y [np.isin(S.numpy(), [0, 51, 53, 57, 83])]  = 4
            Y [np.isin(S.numpy(), [22, 39, 40, 86, 87])] = 5
            Y [np.isin(S.numpy(), [5, 20, 25, 84, 94])]  = 6
            Y [np.isin(S.numpy(), [6, 7, 14, 18, 24])]   = 7
            Y [np.isin(S.numpy(), [3, 42, 43, 88, 97])]  = 8
            Y [np.isin(S.numpy(), [12, 17, 37, 68, 76])] = 9
            Y [np.isin(S.numpy(), [23, 33, 49, 60, 71])] = 10
            Y [np.isin(S.numpy(), [15, 19, 21, 31, 38])] = 11
            Y [np.isin(S.numpy(), [34, 63, 64, 66, 75])] = 12
            Y [np.isin(S.numpy(), [26, 45, 77, 79, 99])] = 13
            Y [np.isin(S.numpy(), [2, 11, 35, 46, 98])]  = 14
            Y [np.isin(S.numpy(), [27, 29, 44, 78, 93])] = 15
            Y [np.isin(S.numpy(), [36, 50, 65, 74, 80])] = 16
            Y [np.isin(S.numpy(), [47, 52, 56, 59, 96])] = 17
            Y [np.isin(S.numpy(), [8, 13, 48, 58, 90])]  = 18
            Y [np.isin(S.numpy(), [41, 69, 81, 85, 89])] = 19


            self.X = X
            self.Y = Y.unsqueeze(1)
            self.S = S.unsqueeze(1)

            # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, Y, S = self.X[index], self.Y[index], self.S[index]
        return X, Y, S