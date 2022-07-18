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


## References
1. [_Optimizing Neural Networks with Kronecker-factored Approximate Curvature_, Martens, et al., 2015](https://arxiv.org/abs/1503.05671)
2. [_A Kronecker-factored approximate Fisher matrix for convolution layers_, Martens et al., 2016](https://arxiv.org/abs/1602.01407)
3. [_Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation_, Wu et al., 2017](https://arxiv.org/abs/1708.05144)

## Acknowledgement
Following repositories were great guides to build up the current repository. Big thanks to them for their works and you can find them very handy if you're interested in more advanced implementations of KFAC or ACKTR:
1. [KFAC-Pytorch](https://github.com/alecwangcq/KFAC-Pytorch) by [@alecwangcq](https://github.com/alecwangcq)
2. [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) by [@ikostrikov](https://github.com/ikostrikov)
3. [baselines](https://github.com/openai/baselines) by [@OpenAI](https://github.com/openai)n