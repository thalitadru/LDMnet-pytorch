This repo implements LDMnet using:
- pytorch for network models
- skorch as a high level model training API
- sacred for argument and experiment management

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

