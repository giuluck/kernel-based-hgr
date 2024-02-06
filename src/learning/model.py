from typing import Optional, Union, Iterable, Tuple, Dict, Any

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.optim import Optimizer, Adam

from src.hgr import DoubleKernelHGR, AdversarialHGR, HGR
from src.hgr.adv import Net_HGR, Net2_HGR, EPOCHS


class MultiLayerPerceptron(pl.LightningModule):
    """Template class for a Multi-layer Perceptron in Pytorch Lightning."""

    THRESHOLD: float = 0.2
    """The threshold to be imposed in the penalty."""

    def __init__(self,
                 units: Iterable[int],
                 classification: bool,
                 feature: int,
                 metric: Optional[HGR] = None,
                 alpha: Optional[float] = None):
        """
        :param units:
            The neural network hidden units.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param feature:
            The index of the excluded feature.

        :param metric:
            The kind of metric to be imposed as penalty, or None for unconstrained model.

        :param alpha:
            The weight of the penalizer, or None for automatic weight regularization via lagrangian dual technique.
            If the penalty is None, must be None as well and it is ignored.
        """
        super(MultiLayerPerceptron, self).__init__()

        # disable lightning manual optimization to potentially deal with two optimizers
        self.automatic_optimization = False

        # build the layers by appending the final unit
        layers = []
        units = list(units)
        for inp, out in zip(units[:-1], units[1:]):
            layers.append(nn.Linear(inp, out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(units[-1], 1))

        # if there is a penalty and alpha is None, then build a variable for alpha
        if alpha is None and metric is not None:
            alpha = Variable(torch.zeros(1), requires_grad=True, name='alpha')
        # otherwise, check that either there is a penalty or alpha is None (since there is no penalty)
        else:
            assert metric is not None or alpha is None, "If metric=None, alpha must be None as well."

        # build a dictionary of arguments for the penalizer and check input consistency
        if isinstance(metric, DoubleKernelHGR):
            penalty_arguments = dict(x0=None)
        elif isinstance(metric, AdversarialHGR):
            penalty_arguments = dict(epochs=EPOCHS, net_1=Net_HGR(), net_2=Net2_HGR())
        else:
            penalty_arguments = dict()

        self.model: nn.Sequential = nn.Sequential(*layers)
        """The neural network."""

        self.loss: nn.Module = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
        """The loss function."""

        self.metric: Optional[HGR] = metric
        """The metric to be used as penalty, or None for unconstrained model."""

        self.alpha: Union[None, float, Variable] = alpha
        """The alpha value for balancing compiled and regularized loss."""

        self.feature: int = feature
        """The index of the excluded feature."""

        self._penalty_arguments: Dict[str, Any] = penalty_arguments
        """The arguments passed to the penalizer."""

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass on the model given the input (x)."""
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Performs a training step on the given batch."""
        # retrieve the data and the optimizers
        inp, out = batch
        optimizers = self.optimizers()
        def_opt, reg_opt = optimizers if isinstance(optimizers, list) else (optimizers, None)
        # perform the standard loss minimization step
        def_opt.zero_grad()
        pred = self.model(inp)
        def_loss = self.loss(pred, out)
        # if there is a penalty, compute it for the minimization step
        if self.metric is None:
            reg = torch.tensor(0.0)
            alpha = torch.tensor(0.0)
            reg_loss = torch.tensor(0.0)
        else:
            reg = self.metric(a=inp[:, self.feature], b=pred.squeeze(), **self._penalty_arguments)
            reg_loss = self.alpha * reg
            alpha = self.alpha
        # build the total minimization loss and perform the backward pass
        tot_loss = def_loss + reg_loss
        self.manual_backward(tot_loss)
        def_opt.step()
        # if there is a variable alpha, perform the maximization step (loss + penalty with switched sign)
        if isinstance(self.alpha, Variable):
            reg_opt.zero_grad()
            pred = self.model(inp)
            def_loss = self.loss(pred, out)
            reg = self.metric(a=inp[:, self.feature], b=pred.squeeze(), **self._penalty_arguments)
            reg_loss = self.alpha * reg
            tot_loss = def_loss + reg_loss
            self.manual_backward(-tot_loss)
            reg_opt.step()
        # return the information about the training
        return {
            'loss': tot_loss,
            'def_loss': def_loss,
            'reg_loss': reg_loss,
            'alpha': alpha,
            'reg': reg
        }

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return dict()

    def configure_optimizers(self) -> Union[Optimizer, Tuple[Optimizer, Optimizer]]:
        """Configures the optimizer for the MLP depending on whether there is a variable alpha or not."""
        if isinstance(self.alpha, Variable):
            # noinspection PyTypeChecker
            return Adam(params=self.model.parameters()), Adam(params=[self.alpha])
        else:
            return Adam(params=self.model.parameters())
