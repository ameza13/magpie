<!-- # 🐦 Magpie Fork -->

This is a fork of [**Magpie**](https://magpie-align.github.io/) project. We have modified the original scripts to work with open source models: 

## Environment setup

#### Set Environment variables
```
export DATA_MGT="/path/to/your/input/data/dir" 
export WORKSPACE="/path/to/your" 
export HF_HOME="/path/to/your/HF_HOME"
export HF_HUB_CACHE="/path/to/your/HF_HUB_CACHE"
```

#### Access to HF gated models
Create a HF token and login via HF cli

`huggingface-cli login` -> Type your HF Token

#### Create conda environment
```
conda create --prefix /path/to/conda/envs/dir/magpie python=3.10
conda activate /path/to/conda/envs/dir/magpie
```

#### Install requirements
`pip install -r requirements.txt`

## Data Generation Models
- `mistralai/Mistral-7B-Instruct-v0.3`
- `mistralai/Mixtral-8x22B-Instruct-v0.1`

#### How to run?

## Data Tagging Models
- difficulty, quality, classification: `mistralai/Mistral-7B-Instruct-v0.3`
- safety model: `allenai/wildguard`
- reward model: WIP

#### Data preprocessing
Convert input data to this schema: [here](/data/input_schema_example.json)

#### How to run?

```
cd /magpie/
chmod +x ./scripts/blue-vela/unitag.sh
./scripts/blue-vela/unitag.sh
```

Check the original project README -> [here](/README_ORIGINAL.md)
