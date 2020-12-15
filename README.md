# QUCKIE
This repository contains code for the QUACKIE paper. For the benchmark website, please visit [QUCKIE](link).

## Usage
The file `run_experiment.py` can be used for running the experiment. To run the experiment with a custom interpreter on a specific dataset, load the interpreter in line 21-30 and run using the command `python run_experiment.py --run --dataset DATASET`, where DATASET is one of the supported datasets. The following command line arguments can be given
- `--run`: Flag to run the experiment
- `--analyze`: Flag to analyze the results
- `--dataset` : Dataset to use. Must be provided if `--run` is given.
- `--name` : Name of the interpreter to use in table after analyzing.
- `--info` : Further info given in the table, we recommend website or link to paper.
- `--no_cuda` : Flag to force CPU usage.

## Files
The other files are:
- `interpreters/.`: Interpreters provided as baseline.
- `accuracy.py`, `dataset_analysis.py` and `test_gt.ipynb`: Files for dataset and task exploration
- `models.py`: Models provided as classifiers to interpret.
- `run_sota.py`: File used to run SOTA/baselines
- `qa_experimenters.py`: QA experimentation engine.

## Submission
Submissions can be made using pull requests on the `gh-pages` branch by adding results to the file `results.json`.
