# Regression experiment

## Data
We assume the data is stored in the `data` folder, which is a child directory of `project_root/experiments/regression/`. The data files are named as `<dataset>_<input-dim>_<split>.npz`, where `dataset` is the name of the dataset, `input-dim` is the input dimension and `split` is "training" or "interpolation" (i.e. test split). For example, the data file for the SE dataset with input dimension 1 for training is `se_1_training.npz`. The data files can be downloaded from [here](https://drive.google.com/drive/folders/1PgeXKvNRnz13FJNF2MZhCsH61EI8OdB7?usp=share_link).

The data can also be generated by running the following command:
```
python generate_data.py;
```
all the necessary logic to sample from each dataset is in `data.py`.


## Model evaluation and training

To *evaluate* the NDP regression models, run the following commands (in parallel if you have multiple GPUs) for each dataset and input dimension.
```
python main.py --config.dataset=se --config.input_dim=1 --config.restore=trained_models/May26_014440_rcva/;
python main.py --config.dataset=se --config.input_dim=2 --config.restore=trained_models/May26_023039_zrox/;
python main.py --config.dataset=se --config.input_dim=3 --config.restore=trained_models/May26_042713_hjds/;
python main.py --config.dataset=matern --config.input_dim=1 --config.restore=trained_models/May27_145742_pjjl/;
python main.py --config.dataset=matern --config.input_dim=2 --config.restore=trained_models/May27_145742_zjii/;
python main.py --config.dataset=matern --config.input_dim=3 --config.restore=trained_models/May27_154336_nter/;
```

The models can be *trained* by running the following commands (in parallel if you have multiple GPUs) for each dataset and input dimension.
```
python main.py  --config.dataset=se --config.input_dim=1;
python main.py  --config.dataset=se --config.input_dim=2;
python main.py  --config.dataset=se --config.input_dim=3;
python main.py  --config.dataset=matern --config.input_dim=1;
python main.py  --config.dataset=matern --config.input_dim=2;
python main.py  --config.dataset=matern --config.input_dim=3;
```
The complete list of hyperparameters can be found in `config.py`.


## Other files

- `data.py`: contains the logic to sample the data.
- `eval_gp.py`: contains the logic to evaluate the GP baseline models, and `evaluate_gp.sh` is a bash script to loop over all the datasets and input dimensions.
- `create_commands.py`: Utility script to create the bash commands to run the experiments.