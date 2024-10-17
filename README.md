
# SUNAR (Semantic Uncertainty based Neighborhood Aware Retrieval for Complex QA)
<p align="center">
  <img src="Screenshot 2024-10-17 at 6.06.46 PM.png" />
</p>


# Setup
1) Clone the repo <br />
2) Create a conda environment conda create -n SUNAR python=3.10.3  <br />
conda activate SUNAR
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
To run first-stage retrieval run
```
python evaluation/wikimultihop/run_splade_inference.py
```

To directly reproduce the best results in paper (Searchain+SUNAR) in Table 2 run

```
python evaluation/wikimultihop/llms/run_searchain_sunar.py
```
Note the above script runs using evidences saved froma  run of SUNAR algorithm and does inference to enable easy reproduction of the results in paper

If interested in running SUNAR end-end run 

```
python evaluation/wikimultihop/llms/run_sunar.py
```
