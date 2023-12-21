# DQN - Playing Atari with Deep Reinforcement Learning

This repo aims to recreate the results from the [DeepMind](https://deepmind.google/) [paper](https://arxiv.org/abs/1312.5602):

```
@misc{mnih2013playing,
      title={Playing Atari with Deep Reinforcement Learning}, 
      author={Volodymyr Mnih and Koray Kavukcuoglu and David Silver and Alex Graves and Ioannis Antonoglou and Daan Wierstra and Martin Riedmiller},
      year={2013},
      eprint={1312.5602},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
Additionally helpful [paper](https://doi.org/10.1038/nature14236).

# Training

Run the following command to train ($\epsilon$-greedy with linear $\epsilon$ decay from 1 to 0.1 over 1000000 explortation steps; see the paper for details):

`python dqn.py --seed 0 --env ENVNAME --savepath PATH/TO/SAVE --device cuda`
- Use the argument `--N` to specify the replay buffer size. 1000000 is true to the paper, but your hardware (i.e., GPU) might be limiting.
- If you wish to modify to the number of training epoch, use the argumet `--epochs`. Otherwise, it defaults to the value 5000000 which is true to the paper.
- To enable learning rate scheduling use `--lr_scheduler`. It is configured to [`ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html).
- To enable gradient clipping use `--clipgrads`. It is configured to [-1,1].

# Evaluating

## DQN Evaluations
Run the following command to evaluate ($\epsilon$-greedy with $\epsilon$=0.05 for 10000 steps; see the paper for details):

`python eval.py --seed 0 --env ENVNAME --savepath PATH/TO/SAVE --weights PATH/TO/ENVNAME/ENVNAME_0_XXX.pth --device cuda`
- To change the number of evaluation steps, use `--steps`. It defaults to 10000 which is true to the paper.
- To change $\epsilon$ for $\epsilon$-greedy, use `--epsilon`. It defaults to 0.05 which is true to the paper.
- To enable cropping of environment, use `--cropenv`.

## Random Policy Evaluations
Use something like below (similar to [DQN Evaluations](dqn-evaluations) above):

`python eval_random.py --seed 0 --env ENVNAME --savepath PATH/TO/SAVE/UNCROPPED`

`python eval_random.py --seed 0 --env ENVNAME --savepath PATH/TO/SAVE/CROPPED --cropenv`

