# Boosting Offline MARL under Imbalanced Datasets via Compositional Diffusion Models

This repository contains implementation for **Boosting Offline MARL under Imbalanced Datasets via Compositional Diffusion Models** (CODI).

## Environment Installation

Install the MPE environment by running:

```
pip install -e offpymarl/src/envs/mpe/multi_agent_particle
```

Install the StarCraft Multi-Agent Challenge (SMAC) environment by running:

```
pip install -e offpymarl/src/envs/smac
```

Install the SMACv2 environment by running:

```
pip install -e offpymarl/src/envs/smacv2
```

## Run An Experiment

### Training Behavior Policies and Collecting Imbalanced Datasets

Codes are provided in `offpymarl/scripts/collect.sh`.

### Train Agent-Quality Labeler

Codes are provided in `CODI_llm_label/`.

### Training Stage

Before training the models required for trajectory stitching, you should set the correct paths for datasets and label models.

One example here for start training by run the following command:

```bash
python run_experiment.py -e exp_specs/<env_name>/<task>/<dataset>.yaml
```

You can modify the content in the yaml file to modify the specific experiment settings. 

### Stitching Stage

The models after training are placed under the directory `logs`. By specifying the path of the model and other hyperparameters. Start trajectory stitching by switching to `CODI_diffusion/` and run:

```bash
python exp_specs/syn.py
```

And the generated dataset for augmentation is under the path `<model_path/syn_datasets>` for subsequent policy training.

We also provide the code for the implementation of baseline MBTS in the `MBTS/` directory.

### Offline MARL Stage

Codes are provided in `offpymarl/scripts/offline.sh`.

## Publication

If you find this repository useful, please cite our paper:

```
@inproceedings{codi,
  title     = {Boosting Offline MARL under Imbalanced Datasets via Compositional Diffusion Models},
  author    = {Lihe Li and Shenghe Hu and Bingxuan Lan and Yuqi Bian and Huan Zhang and Ming Zhao and Chongjie Zhang and Lei Yuan and Yang Yu},
  booktitle = {Proceedings of the International Conference on Autonomous Agents and Multiagent Systems},
  year      = {2026}
}
```

