# LDMnet implementation in pytorch
This repo implements LDMNet presented in the following work:
>[
LDMNet: Low Dimensional Manifold Regularized Neural Networks](https://arxiv.org/abs/1711.06246)
Wei Zhu, Qiang Qiu, Jiaji Huang, Robert Calderbank, Guillermo Sapiro, Ingrid Daubechies

 This implementation uses:
- [pytorch > 0.4.1](https://pytorch.org/) for network models
- [skorch >= 0.4.0](https://github.com/dnouri/skorch) as a high level model training API
- [sacred > 0.7.4]() for argument and experiment management

## To run
To run use:
```
python main.py with mnist
python main.py with cifar10
python main.py with svhn
```

You can change parameters as follows:
```
python main.py with mnist train_size=1000 device=cuda dropout=0.5 alphaupdate.lambda_bar=0.01
```

Call `python main.py print_config` to see all parameters available.

Results, arguments, run info and network weights for each run will be stored in `ldmnet_runs`, under a directory corresponding to the run's id number.

## To be implemented
- `test` command to load previous run and evaluate on test set
