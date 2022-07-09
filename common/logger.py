import time
import numpy as np
import os
import datetime
import glob
from collections import deque
import json
import psutil
import sys
import math
import wandb


class Logger:
    def __init__(self, brain, **config):
        self.config = config
        self.brain = brain
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.episode = 0
        self.episode_reward = 0
        self.running_reward = 0
        self.max_episode_rewards = -np.inf
        self.episode_length = 0
        self.moving_avg_window = 10
        self.running_training_logs = 0
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        self.running_last_10_r = 0  # It is not correct but does not matter.
        self.to_gb = lambda x: x / 1024 / 1024 / 1024

        sys.stdout.write("\033[;1m")  # bold code
        print("params:", self.config)
        sys.stdout.write("\033[0;0m")  # Reset code

        wandb.init(project=self.config["agent_name"],
                   config=config,
                   job_type="train",
                   name=self.log_dir
                   )
        # wandb.watch(agent.online_model)
        if not self.config["do_test"]:
            self.create_wights_folder(self.log_dir)

        self.exp_avg = lambda x, y: 0.99 * x + 0.01 * y if (y != 0).all() else y

    @staticmethod
    def create_wights_folder(dir):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        os.mkdir("weights/" + dir)

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log_iteration(self, *args):
        iteration, beta, training_logs = args

        if np.isnan(np.mean(training_logs[:-1])):
            raise RuntimeError(f"NN has output NaNs! {training_logs}")
        if math.isnan(training_logs[-1]):
            training_logs = list(training_logs[:-1])
            training_logs.append(self.running_training_logs[-1])

        self.running_training_logs = self.exp_avg(self.running_training_logs, np.array(training_logs))

        if iteration % (self.config["interval"] // 3) == 0:
            self.save_params(self.episode, iteration)

        metrics = {"Running Episode Reward": self.running_reward,
                   "Running last 10 Reward": self.running_last_10_r,
                   "Max Episode Reward": self.max_episode_rewards,
                   "Episode Length": self.episode_length,
                   "Running PG Loss": self.running_training_logs[0],
                   "Running Value Loss": self.running_training_logs[1],
                   "Running Entropy": self.running_training_logs[2],
                   "Running Grad norm": self.running_training_logs[3],
                   "Running Explained variance": self.running_training_logs[4],
                   "episode": self.episode,
                   "iteration": iteration
                   }
        wandb.log(metrics)

        self.off()
        if iteration % self.config["interval"] == 0:
            ram = psutil.virtual_memory()
            print("\nIter: {}| "
                  "E: {}| "
                  "E_Reward: {:.1f}| "
                  "E_Running_Reward: {:.1f}| "
                  "Iter_Duration: {:.3f}| "
                  "Mem_size: {}| "
                  "Beta: {:.1f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time: {} "
                  .format(iteration,
                          self.episode,
                          self.episode_reward,
                          self.running_reward,
                          self.duration,
                          len(self.brain.memory),
                          beta,
                          self.to_gb(ram.used),
                          self.to_gb(ram.total),
                          datetime.datetime.now().strftime("%H:%M:%S"),
                          )
                  )
        self.on()

    def log_episode(self, *args):
        self.episode, self.episode_reward, episode_length = args

        self.max_episode_rewards = max(self.max_episode_rewards, self.episode_reward)

        if self.episode == 1:
            self.running_reward = self.episode_reward
            self.episode_length = episode_length
        else:
            self.running_reward = self.exp_avg(self.running_reward, self.episode_reward)
            self.episode_length = 0.99 * self.episode_length + 0.01 * episode_length

        self.last_10_ep_rewards.append(self.episode_reward)
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            self.running_last_10_r = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')

    # region save_params
    def save_params(self, episode, iteration):
        stats_to_write = {"iteration": iteration,
                          "episode": episode,
                          "running_reward": self.running_reward,
                          "running_last_10_r": self.running_last_10_r
                          if not isinstance(self.running_last_10_r, np.ndarray) else self.running_last_10_r[0],
                          "running_training_logs": list(self.running_training_logs)
                          }
        self.brain.policy.save_weights("Models/" + self.weight_dir + "/weights.h5", save_format="h5")
        with open("Models/" + self.weight_dir + "/stats.json", "w") as f:
            f.write(json.dumps(stats_to_write))
            f.flush()

    # endregion

    # region load_weights
    def load_weights(self):
        model_dir = glob.glob("Models/*")
        model_dir.sort()
        self.weight_dir = model_dir[-1].split(os.sep)[-1]

        #         self.brain.policy.build([(None, *self.config["state_shape"]), (None, 256), (None, 256)])
        self.brain.policy.load_weights(model_dir[-1] + "/weights.h5")
        with open(model_dir[-1] + "/stats.json", "r") as f:
            stats = json.load(f)
        self.running_last_10_r = stats["running_last_10_r"]
        self.running_training_logs = np.asarray(stats["running_training_logs"])
        self.running_reward = stats["running_reward"]

        return stats["iteration"], stats["episode"]
    # endregion
