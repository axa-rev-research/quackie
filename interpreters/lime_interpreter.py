# lime_interpreter.py
# Intrepreter using LIME
import numpy as np

from lime.lime_text import LimeTextExplainer
from nltk.tokenize import word_tokenize, sent_tokenize


class LimeInterpreter:
    '''
    Interpreter based on LIME

    Parameters
    ----------

    n_samples: (int, optional: default=10)
        The number of samples to draw from the neighborhood.

    use_proba: (bool, optional: default=True)
        Wether to use the probability output of the model.

    aggregate_method: (fctn handle, optional: default=sum)
        Method to aggregate the scores for words in each sentence
    '''
    def __init__(self, n_samples=10, use_proba=True, aggregate_method=sum, batch_size=2):
        self.n_samples = n_samples
        self.use_proba = use_proba
        self.aggregate_method = aggregate_method
        self.batch_size = batch_size

    def interpret(self, model, question, context):
        # handler for querying the model
        def has_answer(texts):
            if self.use_proba:
                res = []
                for i in range(0, len(texts), self.batch_size):
                    res.append(model.predict_proba(
                        [question] * len(texts[i: i + self.batch_size]),
                        texts[i: i + self.batch_size]
                    ))
                return np.vstack(res)
            else:
                res = []
                for i in range(0, len(texts), self.batch_size):
                    res.append(model.predict(
                        [question] * len(texts[i: i + self.batch_size]),
                        texts[i: i + self.batch_size]
                    ))
                res = np.hstack(res)
                return np.vstack((np.abs(1 - res), res)).T

        # generate LIME explanations
        exp = dict(LimeTextExplainer().explain_instance(context, has_answer, num_samples=self.n_samples).as_list())

        def contribution(word):
            if word in exp.keys():
                return exp[word]
            else:
                return 0

        # relate explanations to the context
        text_array = [word_tokenize(sent) for sent in sent_tokenize(context)]
        contribution_array = [[contribution(word) for word in sent_array] for sent_array in text_array]
        # aggregate the contributions to the sentence
        sent_contribution = [self.aggregate_method(word_conts) for word_conts in contribution_array]
        return sent_contribution
