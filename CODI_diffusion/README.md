# Efficient Multi-agent Offline Coordination via Diffusion-based Trajectory Stitching

This repository contains implementation for **Efficient Multi-agent Offline Coordination via Diffusion-based Trajectory Stitching** (MADiTS).

## Environment Installation

Build the environment by running:

```
pip install -r requirements.txt
```

Install the MPE environment by running:

```
pip install -e third_party/multiagent-particle-envs
pip install -e third_party/ddpg-agent
```

Install the StarCraft Multi-Agent Challenge (SMAC) environment by running:

```
pip install -e third_party/smac
```

## Run an experiment

### Training Stage

Before training the models required for trajectory stitching, you should include your dataset under the path `diffuser/datasets/data/<env_name>/<task_name>/<dataset_name>`.

One example here for start training by run the following command:

```bash
python run_experiment.py -e exp_specs/<env_name>/<task>/<dataset>.yaml
```

You can modify the content in the yaml file to modify the specific experiment settings.

### Stitching Stage

The models after training are placed under the directory `logs`. By specifying the path of the model and other hyperparameters. Start trajectory stitching by running:

```bash
python run_experiment.py -e exp_specs/syn.yaml
```

And the generated dataset for augmentation is under the path `<model_path/syn_datasets>` for subsequent policy training.








