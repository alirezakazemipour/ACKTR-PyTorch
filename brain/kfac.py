# K-FAC Optimizer as a pre-conditioner for SGD
import torch
import torch.nn.functional as F  # noqa
from torch import optim
import math


class KFAC(optim.Optimizer):
    def __init__(self,
                 model: torch.nn.Module,
                 lr,
                 weight_decay=0,
                 damping=1e-3,
                 momentum=0.9,
                 eps=0.99,
                 Ts=1,  # noqa
                 Tf=10,  # noqa
                 max_lr=1,
                 trust_region=0.001
                 ):

        super(KFAC, self).__init__(model.parameters(), {})
        self.acceptable_layer_types = [torch.nn.Linear, torch.nn.Conv2d]
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.damping = damping
        self.eps = eps
        self.Ts = Ts
        self.Tf = Tf
        self.max_lr = max_lr
        self.trust_region = trust_region
        self._k = 0
        self._aa_hat, self._gg_hat = {}, {}
        self._eig_a, self._Q_a = {}, {}
        self._eig_g, self._Q_g = {}, {}
        self._trainable_layers = []
        self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.fisher_backprop = False
        self._keep_track_aa_gg()

    def _keep_track_aa_gg(self):
        for m in self.model.modules():
            if type(m) in self.acceptable_layer_types:
                m.register_forward_pre_hook(self._save_aa)
                m.register_backward_hook(self._save_gg)
                self._trainable_layers.append(m)

    def _save_aa(self, layer, layer_input):
        if torch.is_grad_enabled() and self._k % self.Ts == 0:
            a = layer_input[0]
            batch_size = a.size(0)
            if isinstance(layer, torch.nn.Conv2d):
                a, spatial_size = img2col(a, layer.kernel_size, layer.stride, layer.padding)

            if layer.bias is not None:
                a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

            # if isinstance(layer, torch.nn.Conv2d):
            #     a /= spatial_size

            aa = a.t() @ a / batch_size

            if self._k == 0:
                self._aa_hat[layer] = aa.clone()

            polyak_avg(aa, self._aa_hat[layer], self.eps)

    def _save_gg(self, layer, grad_forwardprop, grad_backprop):  # noqa
        if self.fisher_backprop:
            g = grad_backprop[0]
            batch_size = g.size(0)
            if self._k % self.Ts == 0:
                if isinstance(layer, torch.nn.Conv2d):
                    ow, oh = g.shape[-2:]
                    g = g.transpose_(1, 2).transpose_(2, 3).contiguous()
                    g = g.view(-1, g.size(-1)) * ow * oh

                g *= batch_size
                gg = g.t() @ (g / batch_size)

                if self._k == 0:
                    self._gg_hat[layer] = gg.clone()

                polyak_avg(gg, self._gg_hat[layer], self.eps)

    def _update_inverses(self, l):
        self._eig_a[l], self._Q_a[l] = torch.linalg.eigh(self._aa_hat[l], UPLO='U')
        self._eig_g[l], self._Q_g[l] = torch.linalg.eigh(self._gg_hat[l], UPLO='U')
        self._eig_a[l] *= (self._eig_a[l] > 1e-6).float()
        self._eig_g[l] *= (self._eig_g[l] > 1e-6).float()

    def step(self, closure=None):
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(p.data, self.weight_decay)

        updates = {}
        for layer in self._trainable_layers:
            p = next(layer.parameters())

            if self._k % self.Tf == 0:
                self._update_inverses(layer)
            grad = p.grad.data
            if isinstance(layer, torch.nn.Conv2d):
                grad = grad.view(grad.size(0), -1)

            if layer.bias is not None:
                grad = torch.cat([grad, layer.bias.grad.data.view(-1, 1)], 1)

            V1 = self._Q_g[layer].t() @ grad @ self._Q_a[layer]  # noqa
            V2 = V1 / (self._eig_g[layer].unsqueeze(-1) @ self._eig_a[layer].unsqueeze(0) + (
                    self.damping + self.weight_decay))  # noqa
            delta_h_hat = self._Q_g[layer] @ V2 @ self._Q_a[layer].t()

            if layer.bias is not None:
                delta_h_hat = [delta_h_hat[:, :-1], delta_h_hat[:, -1:]]
                delta_h_hat[0] = delta_h_hat[0].view(layer.weight.grad.data.size())
                delta_h_hat[1] = delta_h_hat[1].view(layer.bias.grad.data.size())
            else:
                delta_h_hat = [delta_h_hat.view(layer.weight.grad.data.size())]

            updates[layer] = delta_h_hat

        second_taylor_expand_term = 0
        for layer in self._trainable_layers:
            v = updates[layer]
            second_taylor_expand_term += (v[0] * layer.weight.grad.data * self.lr ** 2).sum()
            if layer.bias is not None:
                second_taylor_expand_term += (v[1] * layer.bias.grad.data * self.lr ** 2).sum()

        nu = min(self.max_lr, math.sqrt(2 * self.trust_region / (second_taylor_expand_term + 1e-6)))

        for layer in self._trainable_layers:
            v = updates[layer][0]
            layer.weight.grad.data.copy_(v)
            layer.weight.grad.data.mul_(nu)
            if layer.bias is not None:
                v = updates[layer][1]
                layer.bias.grad.data.copy_(v)
                layer.bias.grad.data.mul_(nu)

        self.optim.step()
        self._k += 1


def img2col(tensor,
            kernel_size: tuple,
            stride: tuple,
            padding: tuple
            ):
    x = tensor.data
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1],
                      padding[0], padding[0]
                      )
                  )
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    spatial_size = x.size(1) * x.size(2)
    x = x.view(-1, x.size(3) * x.size(4) * x.size(5))
    return x, spatial_size


def polyak_avg(new, old, tau):  # noqa
    old *= tau / (1 - tau)
    old += new
    old *= (1 - tau)
