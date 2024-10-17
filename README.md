# SUNAR (Semantic Uncertainty based Neighborhood Aware Retrieval for Complex QA)

# Setup
1) Clone the repo <br />
2) Create a conda environment conda create -n SUNAR  <br />
3) pip install -e .<br />

# Download data
curl https://gitlab.tudelft.nl/anonymous_arr/bcqa_data/-/raw/main/2wikimultihopQA.zip -o 2wikimultihopQA.zip

To download the <b> Document graph</b> of the corpus employed in neighborhood adaptive retrieval for MusiqueQa and WikimultihopQA use this link https://drive.google.com/drive/folders/1zyWtCyhQzxaMQpM6uXT5oqvtqFMg7nYl?usp=sharing 


# Config for running experiments
In evaluation/config.ini configure the corresponding paths to downloaded files
configure project root directory to PYTHONPATH variable

Additionally set the following environment variables in the terminal
```
export PYTHONPATH=/path

export OPENAI_KEY=<your openai key>

export huggingface_token = <your huggingface token to access models  >
```

# Running experiments
