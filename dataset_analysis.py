import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

import models
import qa_experimenters
from interpreters import baseline_interpreter

# disable info logging for datasets
qa_experimenters.datasets.logging.set_verbosity_error()

# load models
model_classif = models.Model_Classification()
model_qa = models.Model_QA()


print('Classif')
print('new_wiki')
qa_experimenters.SQuADShiftsExperimenter(model_classif, baseline_interpreter.random_interpreter, 'new_wiki').describe()
print('reddit')
qa_experimenters.SQuADShiftsExperimenter(model_classif, baseline_interpreter.random_interpreter, 'reddit').describe()
print('nyt')
qa_experimenters.SQuADShiftsExperimenter(model_classif, baseline_interpreter.random_interpreter, 'nyt').describe()
print('amazon')
qa_experimenters.SQuADShiftsExperimenter(model_classif, baseline_interpreter.random_interpreter, 'amazon').describe()

print('QA')
print('SQuAD')
qa_experimenters.SQuADExperimenter(model_qa, baseline_interpreter.random_interpreter).describe()
print('new_wiki')
qa_experimenters.SQuADShiftsExperimenter(model_qa, baseline_interpreter.random_interpreter, 'new_wiki').describe()
print('reddit')
qa_experimenters.SQuADShiftsExperimenter(model_qa, baseline_interpreter.random_interpreter, 'reddit').describe()
print('nyt')
qa_experimenters.SQuADShiftsExperimenter(model_qa, baseline_interpreter.random_interpreter, 'nyt').describe()
print('amazon')
qa_experimenters.SQuADShiftsExperimenter(model_qa, baseline_interpreter.random_interpreter, 'amazon').describe()