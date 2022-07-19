[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

# ACKTR-PyTorch

## Demo
<p align="center">
  <img src="results/seaquest.gif" height=250>
</p>  

## Results
> Environment: SeaquestNoFrameskip-v4

<p align="center">
  <img src="results/merged1.png" >
</p>  
<p align="center">
  <img src="results/merged2.png" >
</p>  
<p align="center">
  <img src="results/merged3.png" >
</p>  
<p align="center">
  <img src="results/merged4.png" >
</p>  


## Dependencies
- gym == 0.24.1
- numpy == 1.23.1
- opencv_python == 4.6.0.66
- psutil == 5.9.1
- torch == 1.12.0
- tqdm == 4.64.0
- wandb == 0.12.21

## Usage
```bash
python main.py --interval=500 --train_from_scratch --online_wandb --env_name="SeaquestNoFrameskip-v4"
```
```bash
usage: main.py [-h] [--env_name ENV_NAME] [--num_worker NUM_WORKER]
               [--total_iterations TOTAL_ITERATIONS] [--interval INTERVAL]
               [--online_wandb] [--do_test] [--render] [--train_from_scratch]
               [--seed SEED]

Variable parameters based on the configuration of the machine or user's choice

options:
  -h, --help            show this help message and exit
  --env_name ENV_NAME   Name of the environment.
  --num_worker NUM_WORKER
                        Number of parallel workers. (-1) to use as many as cpu
                        cores.
  --total_iterations TOTAL_ITERATIONS
                        The total number of iterations.
  --interval INTERVAL   The interval specifies how often different parameters
                        should be saved and printed, counted by iterations.
  --online_wandb        Run wandb in online mode.
  --do_test             The flag determines whether to train the agent or play
                        with it.
  --train_from_scratch  The flag determines whether to train from scratch or
                        continue the last try.
  --seed SEED           The random seed.
```
###  Considerations
- You can put your _wandb API key_ in a file named `api_key.wandb` at the root directory of the project and the code will automatically read the key and as a result, there will be no need to insert your wandb credentials each time:
> common/utils.py:
```python
def init_wandb(online_mode=False):
    if os.path.exists("api_key.wandb"):
        with open("api_key.wandb", 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
            if not online_mode:
                os.environ["WANDB_MODE"] = "offline"
```
- At the time of testing, the code by default uses the weights of the latest run available in _`weights`_ folder because each subdirectory is named by the time and the date (e.g. 2022-07-13-06-51-32 indicating 7/13/2022, 6:51:32) that the code was executed correspondingly so, please bear in mind to put your desired `*.pth` file in the appropriate subdirectory inside the _`weights`_ directory! 👇
> common/logger.py:
```python
def load_weights(self):
    model_dir = glob.glob("weights/*")
    model_dir.sort()
    # model_dir[-1] -> means the latest run!
    self.log_dir = model_dir[-1].split(os.sep)[-1]
    checkpoint = torch.load("weights/" + self.log_dir + "/params.pth")

    self.brain.model.load_state_dict(checkpoint["model_state_dict"])
    self.brain.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    self.running_last_10_r = checkpoint["running_last_10_r"]
    self.running_training_logs = 	np.asarray(checkpoint["running_training_logs"])
    self.running_reward = checkpoint["running_reward"]

    return checkpoint["iteration"], checkpoint["episode"]
```

## References
1. [_Optimizing Neural Networks with Kronecker-factored Approximate Curvature_, Martens, et al., 2015](https://arxiv.org/abs/1503.05671)
2. [_A Kronecker-factored approximate Fisher matrix for convolution layers_, Martens et al., 2016](https://arxiv.org/abs/1602.01407)
3. [_Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation_, Wu et al., 2017](https://arxiv.org/abs/1708.05144)

## Acknowledgement
Following repositories were great guides to build up the current repository. Big thanks to them for their works and you can find them very handy if you're interested in more advanced implementations of KFAC or ACKTR:
1. [KFAC-Pytorch](https://github.com/alecwangcq/KFAC-Pytorch) by [@alecwangcq](https://github.com/alecwangcq)
2. [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) by [@ikostrikov](https://github.com/ikostrikov)
3. [baselines](https://github.com/openai/baselines) by [@OpenAI](https://github.com/openai)