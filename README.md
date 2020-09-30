### Code

`train_full_rl.py`: Main file. Call training.py -> rl.py -> model/rl.py. 

`training.py`: training loop

`rl.py`: training epoch (rewards, official evaluation)

`model/rl.py`: core functions, where MMR is injected

`ConfManager.py`: Parameters in addition to argparse

`data_info.py`: store all data paths, data follows the format of [fast_abs_rl](https://github.com/ChenRocks/fast_abs_rl)

`ScoreAgent.py`: MMR



### Acknowledgments
Part of the code is adapted from [fast_abs_rl](https://github.com/ChenRocks/fast_abs_rl).
