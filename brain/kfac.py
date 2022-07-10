# K-FAC Optimizer as a pre-conditioner for SGD
import torch
import torch.nn.functional as F  # noqa
from torch import optim


class KFAC(optim.Optimizer):
    def __init__(self,
                 model: torch.nn.Module,
                 lr,
                 reg=0,
                 damping=1e-3,
                 momentum=0.9,
                 eps=0.99,
                 Ts=1,  # noqa
                 Tf=10  # noqa
                 ):

        super(KFAC, self).__init__(model.parameters(), {})
        self.acceptable_layer_types = [torch.nn.Linear, torch.nn.Conv2d]
        self.model = model
        self.eps = eps
        self.Ts = Ts
        self.Tf = Tf
        self._k = 1
        self._aa_hat = {}
        self._keep_track_aa_gg()

    def _keep_track_aa_gg(self):
        for m in self.model.modules():
            if type(m) in self.acceptable_layer_types:
                m.register_forward_pre_hook(self._save_aa)
                m.register_backward_hook(self._save_gg)

    def _save_aa(self, layer, layer_input):
        a = layer_input[0]
        batch_size = a.size(0)
        if torch.is_grad_enabled() and self._k % self.Ts == 0:
            if isinstance(layer, torch.nn.Conv2d):
                a = img2col(a, layer.kernel_size, layer.stride, layer.padding)

            if layer.bias is not None:
                a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
            aa = a.T @ a / batch_size

            if self._k == 1: # TODo
                self._aa_hat[layer] = aa.clone()

            polyak_avg(aa, self._aa_hat[layer], self.eps)

    def _save_gg(self, layer, grad_forwardprop, grad_backprop):  # noqa
        g = grad_backprop[0]
        batch_size = g.size(0)
        # if torch.is_grad_enabled() and self._k % self.Tf == 0:
        #     if isinstance(layer, torch.nn.Conv2d):
        #         a = img2col(a, layer.kernel_size, layer.stride, layer.padding)
        #
        # if layer.bias is not None:
        #     a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        # aa = a.T @ a / batch_size
        #
        # if self._k == 0:
        #     self._aa_hat[layer] = aa.clone()
        #
        # polyak_avg(aa, self._aa_hat, self.eps)


def img2col(tensor,
            kernel_size: tuple,
            stride: tuple,
            padding: tuple
            ):
    x = tensor
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1],
                      padding[0], padding[0]
                      )
                  ).data
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(-1, x.size(3) * x.size(4) * x.size(5))
    return x


def polyak_avg(new, old, tau):  # noqa
    old *= tau / (1 - tau)
    old += new
    old *= (1 - new)
