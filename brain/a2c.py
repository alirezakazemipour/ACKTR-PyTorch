from .model import CNNModel
import torch
from torch import from_numpy
import numpy as np
from torch.optim.adam import Adam
from common import explained_variance


class Brain:
    def __init__(self, **config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = CNNModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config["lr"])
        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).to(self.device)
        with torch.no_grad():
            dist, value = self.model(state)
            action = dist.sample()
        return action.cpu().numpy(), value.cpu().numpy().squeeze()

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_returns(rewards, next_values, dones, n=self.config["n_workers"])
        target_values = np.hstack(values)
        advs = returns - values

        dist, value = self.model(states)
        ent = dist.entropy().mean()
        log_prob = dist.log_prob(actions)
        a_loss = (log_prob * advs).mean()
        c_loss = self.mse_loss(target_values, value.squeeza(-1))
        total_loss = a_loss + self.config["critic_coeff"] * c_loss - self.config["ent_coeff"] * ent  # noqa
        self.optimize(total_loss)
        return a_loss.item(), c_loss.numpy(), ent.numpy(), explained_variance(target_values, returns)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_returns(self, rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, n: int) -> np.ndarray:
        if next_values.shape == ():
            next_values = next_values[None]

        returns = [[] for _ in range(n)]
        for worker in range(n):
            R = next_values[worker]  # noqa
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + self.config["gamma"] * R * (1 - dones[worker][step])  # noqa
                returns[worker].insert(0, R)

        return np.hstack(returns).astype("float32")

    def set_from_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def set_to_eval_mode(self):
        self.model.eval()
