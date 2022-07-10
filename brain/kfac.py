# K-FAC Optimizer as a pre-conditioner for SGD
import torch

from torch import optim


class KFAC(optim.Optimizer):
    def __init__(self,
                 model: torch.nn.Module,
                 lr,
                 damping,
                 reg,
                 momentum,
                 eps,
                 ):

        super(KFAC, self).__init__(model.parameters(), {})
        self.acceptable_layer_types = [torch.nn.Linear, torch.nn.Conv2d]
        self.model = model
        self._k = 0
        self._aa_hat = {}
        self._keep_track_aa_gg()

    def _keep_track_aa_gg(self):
        for m in self.model.modules():
            if type(m) in self.acceptable_layer_types:
                m.register_forward_pre_hook(self._save_aa)
                # m.register_backward_hook(self._save_gg)

    def _save_aa(self, layer, layer_input):
        if torch.is_grad_enabled() and self._k == 0:
            aa = get_cov_a(layer_input)

        if self._k == 0:
            self._aa_hat[layer] = layer_input.clone()


def get_cov_a(tensor):
    return 0
