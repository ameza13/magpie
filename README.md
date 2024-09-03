<!-- # 🐦 Magpie Fork -->

# Magpie Fork
This is a fork of [**Magpie**](https://magpie-align.github.io/) project. We have modified the original scripts to work with open source models.

You can check the original project README -> [here](/README_ORIGINAL.md)

## Environment setup

#### Set Environment variables
```
export DATA_MGT="/path/to/data/dir" # These is the directory to save input, temporal checkpoints, and final output files.
export WORKSPACE="/path/to/parent/directory/of/this/repo" 
export HF_HOME="/path/to/HF_HOME/dir"
export HF_HUB_CACHE="/path/to/HF_HUB_CACHE/dir"
```

#### Access to HF gated models
- To get access to gated models, create a HF token and login via HF CLI.
```
pip install huggingface-cli
huggingface-cli login # Type your HF Token
```
- Make sure to visit the model card of all the models required by this project, and accept the terms of use (if any).

#### Create conda environment
```
conda create --prefix /path/to/conda/envs/dir/magpie python=3.10
conda activate /path/to/conda/envs/dir/magpie
```

#### Install requirements
`pip install -r requirements.txt`

## Data preprocessing
Convert input data to a unified schema (similar to ShareGPT schema). You can find an example of our schema [here](/data/input_schema_example.json)

## Data Generation

#### Generation Models
- `mistralai/Mistral-7B-Instruct-v0.3`
- `mistralai/Mixtral-8x22B-Instruct-v0.1`
- `mistralai/Mistral-Nemo-Instruct-2407`

#### How to run? 
WIP

## Data Tagging 

#### Models
- difficulty, quality, classification: `mistralai/Mistral-7B-Instruct-v0.3`
- safety model: `allenai/wildguard`
- reward model: WIP

#### How to run?
By default, the script runs all the tagging missions: difficulty, quality, classification, safety, reward model.

```
cd /magpie/
chmod +x ./scripts/blue-vela/unitag.sh
./scripts/blue-vela/unitag.sh
```

To run a specific tagging mission comment out the following line before running the script
```
# tag_mission="reward"
```

## Remove repetitions

#### How to run?
```
cd /magpie/
chmod +x ./scripts/blue-vela/remove-repetition.sh
./scripts/blue-vela/remove-repetition.sh
```

## Datasets filtering

Check notebook: ```./data/blue_data_filter.ipynb```