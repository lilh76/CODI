# Boosting Offline MARL under Imbalanced Datasets via Compositional Diffusion Models

This repository contains the official implementation for *Boosting Offline MARL under Imbalanced Datasets via Compositional Diffusion Models*.

## Environment Installation



## Run an experiment

### Generate expert agent labels and train label models

Expert agents relabeling and label model training are implemented under the `llm_label` subdirectory. To run these experiments, first change the current working directory to `llm_label`.

**Preparation**

- Fill in valid LLM configuration in  `llm_label/diffuser/models/call_llm.py` (specifically, `api_key_list` and `model_name` in `LLM`, and `openai.api_base` in `gpt_agent`). This ensures the LLM-assisted expert agent relabeling is executed correctly.
- Make sure that the synthesized imbalanced dataset is located at the path specified by `syn_dataset_dir` in the environment-specific configuration file. For example, to run an experiment for MPE spread-3 environment, the synthesized dataset should be at `diffuser/datasets/data/mpe/spread3/expert-imb2`.

**Relabel expert agents and collect the label batches**

```bash
python run_experiment.py -e exp_specs/[env_type]/[env_subtype]/mad_[specific_env_name]_label_collect.yaml -g [GPU_ID]
```

For example:

```bash
python run_experiment.py -e exp_specs/mpe/spread/mad_mpe_spread3_label_collect.yaml -g 1
```

**Train label models with the collected label batches**

```bash
python run_experiment.py -e exp_specs/[env_type]/[env_subtype]/mad_[specific_env_name]_label_train.yaml -g [GPU_ID]
```

For example:

```bash
python run_experiment.py -e exp_specs/mpe/spread/mad_mpe_spread3_label_train.yaml -g 1
```

Label model checkpoints will be saved under `logs`.

For detailed experiment preferences, please refer to the YAML configuration files.

## Publication