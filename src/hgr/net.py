from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.autograd import Variable

from src.hgr.hgr import HGR


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class AdversarialHGR(HGR):
    @property
    def name(self) -> str:
        return 'nn'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


# The following code is obtained from the official repository of
# "Fairness-Aware Neural Renyi Minimization for Continuous Features"
# (https://github.com/fairml-research/HGR_NN/)

H = 16
H2 = 8


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(82, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.8)
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.8)
        x = self.fc3(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.8)
        x = self.fc4(x)
        return x


# noinspection PyRedeclaration
H = 15
# noinspection PyRedeclaration
H2 = 15


# noinspection PyPep8Naming
class Net_HGR(nn.Module):
    def __init__(self):
        super(Net_HGR, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = torch.tanh(self.fc4(h3))
        return h4


# noinspection PyPep8Naming
class Net2_HGR(nn.Module):
    def __init__(self):
        super(Net2_HGR, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = torch.tanh(self.fc4(h3))
        return h4


model_Net_F = Net_HGR()
model_Net_G = Net2_HGR()


# noinspection PyPep8Naming,PyUnusedLocal
class HGR_NN(nn.Module):

    def __init__(self, model_F, model_G, device, display):
        super(HGR_NN, self).__init__()
        self.mF = model_Net_F
        self.mG = model_Net_G
        self.device = device
        self.optimizer_F = torch.optim.Adam(self.mF.parameters(), lr=0.0005)
        self.optimizer_G = torch.optim.Adam(self.mG.parameters(), lr=0.0005)
        self.display = display

    def forward(self, yhat, s_var, nb):

        svar = Variable(torch.FloatTensor(np.expand_dims(s_var, axis=1))).to(self.device)
        yhatvar = Variable(torch.FloatTensor(np.expand_dims(yhat, axis=1))).to(self.device)

        self.mF.to(self.device)
        self.mG.to(self.device)

        for j in range(nb):

            pred_F = self.mF(yhatvar)
            pred_G = self.mG(svar)

            epsilon = 0.000000001

            pred_F_norm = (pred_F - torch.mean(pred_F)) / torch.sqrt((torch.std(pred_F).pow(2) + epsilon))
            pred_G_norm = (pred_G - torch.mean(pred_G)) / torch.sqrt((torch.std(pred_G).pow(2) + epsilon))

            ret = torch.mean(pred_F_norm * pred_G_norm)
            loss = - ret  # maximize
            self.optimizer_F.zero_grad()
            self.optimizer_G.zero_grad()
            loss.backward()

            if (j % 100 == 0) and (self.display is True):
                print(j, ' ', loss)

            self.optimizer_F.step()
            self.optimizer_G.step()

        # noinspection PyUnboundLocalVariable
        return ret.cpu().detach().numpy()


# noinspection PyPep8Naming,PyUnusedLocal
def FairQuant(s_test, y_test, y_predt_np):
    d = {'sensitivet': s_test, 'y_testt': y_test, 'y_pred3t': y_predt_np}
    df = pd.DataFrame(data=d)
    vec = []
    for i in np.arange(0.02, 1.02, 0.02):
        tableq = df[df.sensitivet <= df.quantile(i)['sensitivet']]
        av_BIN = tableq.y_pred3t.mean()
        av_Glob = df.y_pred3t.mean()
        vec = np.append(vec, (av_BIN - av_Glob))
    FairQuantabs50 = np.mean(np.abs(vec))
    FairQuantsquare50 = np.mean(vec ** 2)
    # print(FairQuantabs50)
    return FairQuantabs50
