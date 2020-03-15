import torch.nn as nn
import config


args = config.parse_args()

__all__ = ['Adversary', 'Target']


class Adversary(nn.Module):
    def __init__(self, r=args.r, nclasses=args.nclasses_a):
        super(Adversary, self).__init__()
        self.fc1 = nn.Linear(r, nclasses,bias=True)

    def forward(self, x):
        x = self.fc1(x)
        return x

# class Adversary(nn.Module):
#     def __init__(self, embed_length=args.r, num_classes=args.nclasses_a):
#         super().__init__()
#
#         self.model1 = nn.Sequential(
#             nn.Linear(embed_length, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#         )
#         self.classlayer = nn.Linear(64, num_classes)
#         self.softmaxlayer = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         z = self.model1(x)
#         out = self.classlayer(z)
#         return out
#######################################

class Target(nn.Module):
    def __init__(self, r=args.r, nclasses=args.nclasses_t):
        super(Target, self).__init__()
        self.fc1 = nn.Linear(r, nclasses, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        return x
#
# class Target(nn.Module):
#     def __init__(self, embed_length=args.r, num_classes=args.nclasses_t):
#         super().__init__()
#
#         self.model1 = nn.Sequential(
#             nn.Linear(embed_length, 37, bias=True),
#             # nn.ReLU(),
#         )
#         self.classlayer = nn.Linear(37, num_classes)
#
#     def forward(self, x):
#         z = self.model1(x)
#         out = self.classlayer(z)
#         return out