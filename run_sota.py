# run_sota.py
# Run state of the art models provided in the paper

import argparse
from datetime import datetime
import os
import torch

import qa_experimenters
from interpreters import baseline_interpreter, captum_interpreters, lime_interpreter, shap_interpreter
import models

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def log(text):
    now = datetime.now().strftime("%H:%M:%S")
    print('[{}]: {}'.format(now, text))


def run(args):
    # load the right model
    model = models.Model_QA(device='cuda:0' if torch.cuda.is_available() else 'cpu') if args.model == 'QA' else models.Model_Classification(device='cuda:0' if torch.cuda.is_available() else 'cpu')

    # initialize the right interpreter
    if args.interpreter == 'Random':
        interpreter = baseline_interpreter.random_interpreter
    elif args.interpreter == 'LIME':
        interpreter = lime_interpreter.LimeInterpreter(n_samples=args.n_samples,
                                                       aggregate_method=sum if args.aggregator == 'sum' else max,
                                                       batch_size=args.batch_size).interpret
    elif args.interpreter == 'SHAP':
        interpreter = shap_interpreter.ShapInterpreter(n_samples=args.n_samples,
                                                       aggregate_method=sum if args.aggregator == 'sum' else max,
                                                       batch_size=args.batch_size).interpret
    elif args.interpreter == 'IntegratedGradients':
        interpreter = captum_interpreters.IgInterpreter('roberta.embeddings' if args.model == 'Classification' else 'albert.embeddings',
                                                        aggregate_method=sum if args.aggregator == 'sum' else lambda l: max(l, default=0),
                                                        n_steps=args.n_samples,
                                                        internal_batch_size=args.batch_size).interpret
    elif args.interpreter == 'Gradient':
        interpreter = captum_interpreters.SaliencyInterpreter('roberta.embeddings' if args.model == 'Classification' else 'albert.embeddings',
                                                              aggregate_method=sum if args.aggregator == 'sum' else lambda l: max(l, default=0)).interpret
    else:
        interpreter = captum_interpreters.SmoothGradInterpreter('roberta.embeddings' if args.model == 'Classification' else 'albert.embeddings',
                                                                aggregate_method=sum if args.aggregator == 'sum' else lambda l: max(l, default=0),
                                                                n_samples=args.n_samples).interpret

    # initialize the right experimenter
    experiment = qa_experimenters.SQuADExperimenter(model, interpreter) if args.dataset == 'SQuAD' else qa_experimenters.SQuADShiftsExperimenter(model, interpreter, args.dataset.lower())
    # prepare the results path, run the experiment and save the results
    results_path = os.path.join('results', args.model + '_model', args.interpreter + '_' + args.aggregator, args.dataset) if args.interpreter in ['Gradient', 'Random'] \
        else os.path.join('results', args.model + '_model', args.interpreter + '_' + str(args.n_samples) + '_' + args.aggregator, args.dataset)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    experiment.experiment()
    experiment.save(path=results_path)


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description='Parameters of the experiment')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=['SQuAD', 'NEW_WIKI', 'NYT', 'Reddit', 'Amazon'],
                        help='Dataset to use, can be one of [\'SQuAD\', \'NEW_WIKI\', \'NYT\', \'Reddit\', \'Amazon\'], SQuAD for SQuAD dataset, others for SQuADShifts')
    parser.add_argument('--interpreter',
                        type=str,
                        required=True,
                        choices=['Random', 'LIME', 'SHAP', 'IntegratedGradients', 'Gradient', 'SmoothGrad'],
                        help='Interpreter, can be one of [\'Random\', \'LIME\', \'SHAP\', \'IntegratedGradients\', \'Gradient\', \'SmoothGrad\']')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        choices=['QA', 'Classification'],
                        help='Model to use, can be one of [\'QA\', \'Classification\']')
    parser.add_argument('--batch_size',
                        type=int,
                        required=False,
                        default=2,
                        help='Batch-Size to use for processing, default=2')
    parser.add_argument('--aggregator',
                        type=str,
                        required=False,
                        default='sum',
                        choices=['sum', 'max'],
                        help='Aggregate method to use to aggregate accross sentence.')
    parser.add_argument('--n_samples',
                        type=int,
                        required=False,
                        default=50,
                        help='Number of samples to use in LIME, SHAP & SmoothGrad. Number of steps to use in Integrated Gradients')

    # disable info logging for datasets
    qa_experimenters.datasets.logging.set_verbosity_error()

    # parse command line arguments and run experiment
    args = parser.parse_args()
    log('Running Experiment')
    run(args)
    log('Done')
