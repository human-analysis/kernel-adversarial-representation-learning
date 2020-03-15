
import torch

__all__ = ['train_e', 'kernel_Gaussian', 'kernel_poly']



def encoder(K, Y, S, r, lam):
    Y_bar = (Y - torch.mean(Y, dim=0))
    S_bar = (S - torch.mean(S, dim=0))


# finding orthonormal basis for K
#     import pdb; pdb.set_trace()
    U, Sigma, V = torch.svd(K)
    d = torch.matrix_rank(K).item()
    L = U[:, 0:d]
    V_d = V[:, 0:d]
    Sigma_d = torch.diag(1/Sigma[0:d])


    B1 = lam * torch.mm(torch.t(L), S_bar)
    B2 = torch.mm(B1, torch.t(S_bar))

    B3 = (lam - 1) * torch.mm(torch.t(L), Y_bar)
    B4 = torch.mm(B3, torch.t(Y_bar))

    B = torch.mm(B2 + B4, L)

    beta, U = torch.eig(B, eigenvectors=True)
    sorted, indices = torch.sort(beta[:, 0])

    G = U[:, indices[0:r]]


    theta = torch.mm(L, G)
    # theta = torch.mm(torch.pinverse(K), theta)
    # import pdb;
    # pdb.set_trace()
    theta = torch.mm(torch.mm(torch.mm(V_d, Sigma_d), torch.t(L)), theta)

    return torch.t(theta)


class kernel_Gaussian:
    def __init__(self):
        pass

    def __call__(self, X, Y, sigma=1):

        n_x = X.shape[0]
        n_y = Y.shape[0]

        X_norm = torch.pow(torch.norm(X, dim=1).reshape([1, n_x]), 2)
        Y_norm = torch.pow(torch.norm(Y, dim=1).reshape([1, n_y]), 2)

        ONES_x = torch.ones([1, n_x])
        ONES_y = torch.ones([1, n_y])

        K = torch.exp(
        (-torch.mm(torch.t(X_norm), ONES_y) - torch.mm(torch.t(ONES_x), Y_norm) + 2 * torch.mm(X, torch.t(Y)))
        / sigma)

        return K

class kernel_poly:
    def __init__(self):
        pass

    def __call__(self, X, Y, c=0, d=1):
        # import pdb; pdb.set_trace()

        K = torch.pow(torch.mm(X, torch.t(Y))+c, d)
        return K


class train_e:

    def __init__(self):
        self.theta_overal = []
        self.theta_overal_weights = []
        self.X = []
        pass

    def __call__(self, args, dataloader, lam):
        dataloader = dataloader['train_e']

        # import pdb; pdb.set_trace()
        for i, (inputs, labels, sensitives) in enumerate(dataloader):
        # if 1<2:

            n = inputs.shape[0]
            D = torch.eye(n) - torch.ones(n) / n


            if args.dataset_train == "YaleB":
                Y = torch.zeros(n, args.total_classes).scatter_(1, labels.long(), 1)
                S = torch.zeros(n, args.total_classes).scatter_(1, sensitives.long(), 1)

            else:
                # import pdb; pdb.set_trace()
                Y = torch.zeros(n, args.nclasses_t).scatter_(1, labels.long(), 1)
                S = torch.zeros(n, args.nclasses_a).scatter_(1, sensitives.long(), 1)
                # S = torch.zeros(n, args.total_classes).scatter_(1, sensitives.long() +args.total_classes - args.nclasses_a, 1)


            # Y = labels
            # S = sensitives


            if args.kernel == 'Linear':
                K = torch.mm(D, inputs)

            elif args.kernel == 'Gaussian':
                kernel = kernel_Gaussian()
                K = kernel(inputs, inputs, args.sigma)
                # import pdb; pdb.set_trace()
                K = torch.mm(torch.mm(D, K), D)

            elif args.kernel == 'Polynomial':
                kernel = kernel_poly()
                K = kernel(inputs, inputs, args.c, args.d)
                # import pdb;
                # pdb.set_trace()
                K = torch.mm(torch.mm(D, K), D)

            else:
                print('#########################'
                      ' Unknown Kernel! ######################\n\n', 'Choose "Linear", "Polynomial" or "Gaussian"\n')
                exit()


            theta = encoder(K, Y, S, args.r, lam)

            self.theta_overal.append(theta)
            self.theta_overal_weights.append(n)
            self.X.append(inputs)

            if args.kernel != 'Linear':  #for Kernelization, one random sampling is enough
                break

        # import pdb; pdb.set_trace()
        if len(self.theta_overal_weights)>1:
            if self.theta_overal_weights[-1] < self.theta_overal_weights[-2]:
                del self.theta_overal[-1]
                del self.theta_overal_weights[-1]
        # import pdb; pdb.set_trace()
        ###################### Averaging over all encoders
        theta_overal = torch.zeros(self.theta_overal[0].shape)
        inputs_overal = torch.zeros(self.X[0].shape)
        for i in range(len(self.theta_overal)):

            theta_overal += self.theta_overal_weights[i]*self.theta_overal[i] / sum(self.theta_overal_weights)
            inputs_overal += self.theta_overal_weights[i]*self.X[i] / sum(self.theta_overal_weights)
        ###################################
        # inputs_overal = self.X[0]
        return theta_overal, inputs_overal

