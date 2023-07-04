# Image regression experiments

To evaluate the image regression model, run the following commands for each dataset.

MNIST model
```
python main.py --config.restore=trained_models/May28_184005_jldx/
```

Celeb-a model
```
python main.py --config.restore=trained_models/May29_131249_jafi/ --config.dataset=celeba32
```

Configurations for each model can be found in next to the checkpoint file.

To train a model from scratch, run the following command:
```
python main.py --config.dataset={mnist,celeba32}
```
Other options can be found in `config.py`.