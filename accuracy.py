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

def get_accuracy(model, experimenter, maxlength=512):
    # perform the testing
    gt = []
    prediction = []
    for i, sample in enumerate(experimenter.dataset):
        context = sample['context']
        question = sample['question'] if type(sample['question'])==str else sample['question'][0]
        # only compute accuracy for texts of appropriate length
        if len(model.tokenizer.encode(question, context, verbose=False))>maxlength:
            continue
        # save ground truth
        gt.append(int(len(sample['answers']['answer_start'])>0))
        # save the prediction
        prediction.append(
            model.predict(question, context)[0]
        )
        if i>=100:
            break
    df = pd.DataFrame(data={
        'prediction' : prediction,
        'true' : gt
    })
    print('Accuracy: {}'.format(accuracy_score(df.true, df.prediction)))
    print('Recall: {}'.format(recall_score(df.true, df.prediction)))
    return df

print('Classif Model')

print('SQuAD Dataset')
_=get_accuracy(model_classif, qa_experimenters.SQuADExperimenter(model_classif, baseline_interpreter.random_interpreter))

print('SQuADShifts Dataset new wiki')
_=get_accuracy(model_classif, qa_experimenters.SQuADShiftsExperimenter(model_classif, baseline_interpreter.random_interpreter, 'new_wiki'))

print('SQuADShifts Dataset reddit')
_=get_accuracy(model_classif, qa_experimenters.SQuADShiftsExperimenter(model_classif, baseline_interpreter.random_interpreter, 'reddit'))

print('SQuADShifts Dataset nyt')
_=get_accuracy(model_classif, qa_experimenters.SQuADShiftsExperimenter(model_classif, baseline_interpreter.random_interpreter, 'nyt'))

print('SQuADShifts Dataset amazon')
_=get_accuracy(model_classif, qa_experimenters.SQuADShiftsExperimenter(model_classif, baseline_interpreter.random_interpreter, 'amazon'))

print('QA Model')

print('SQuAD Dataset')
_=get_accuracy(model_qa, qa_experimenters.SQuADExperimenter(model_qa, baseline_interpreter.random_interpreter))

print('SQuADShifts Dataset new wiki')
_=get_accuracy(model_qa, qa_experimenters.SQuADShiftsExperimenter(model_qa, baseline_interpreter.random_interpreter, 'new_wiki'))

print('SQuADShifts Dataset reddit')
_=get_accuracy(model_qa, qa_experimenters.SQuADShiftsExperimenter(model_qa, baseline_interpreter.random_interpreter, 'reddit'))

print('SQuADShifts Dataset nyt')
_=get_accuracy(model_qa, qa_experimenters.SQuADShiftsExperimenter(model_qa, baseline_interpreter.random_interpreter, 'nyt'))

print('SQuADShifts Dataset amazon')
_=get_accuracy(model_qa, qa_experimenters.SQuADShiftsExperimenter(model_qa, baseline_interpreter.random_interpreter, 'amazon'))