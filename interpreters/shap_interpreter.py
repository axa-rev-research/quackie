# shap_interpreter.py
# interpreter using Kernel SHAP

from nltk.tokenize import word_tokenize, sent_tokenize
from shap import KernelExplainer
import string
import numpy as np


# SHAP-Wrapper for Text
class ShapTextWrapper:
    '''
    Handles the neighborhood defninition (converting binary features into text)

    Parameters
    ----------

    use_bow: (bool, optional: default=True)
        Wether to use BOW assumption (same word at different parts is 1 occurence)
    '''
    def __init__(self, use_bow=True):
        self._use_bow = use_bow

    def get_n_words(self, text):
        '''
        Return the number of words in the text.

        Parameters
        ----------

        text: (str)
            The text to count the words.
        '''
        if self._use_bow:
            return len(dict.fromkeys([w.lower() for w in word_tokenize(text) if w not in string.punctuation]))
        else:
            return len([w for w in word_tokenize(text) if w.isalnum()])

    def get_words(self, text):
        '''
        Return the words in the text.

        Parameters
        ----------

        text: (str)
            The text
        '''
        if self._use_bow:
            return list(dict.fromkeys([w.lower() for w in word_tokenize(text) if w not in string.punctuation]))
        else:
            return [w for w in word_tokenize(text) if w.isalnum()]

    def _mask_text_nobow(self, text, mask):
        i_mask = 0
        words = word_tokenize(text)
        masked_text = ''
        for i_word in range(len(words)):
            if words[i_word] in string.punctuation:
                # is punctuation, add it to the text
                masked_text += words[i_word]
            else:
                # add only if not masked
                if mask[i_mask] == 1:
                    if not words[i_word][0] in string.punctuation:
                        # add a space if word does not start with punctuation (eg ')
                        masked_text += ' '
                    # add the text and increase the word counter
                    masked_text += words[i_word]
                i_mask += 1
        return masked_text.strip()

    def _mask_text_bow(self, text, mask):
        words = list(dict.fromkeys([w.lower() for w in word_tokenize(text) if w not in string.punctuation]))
        words_keep = [w for w, m in zip(words, mask) if m == 1]
        masked_text = ''
        for w in word_tokenize(text):
            if w in string.punctuation:
                masked_text += w
            else:
                if w.lower() in words_keep:
                    # good word
                    if not w[0] in string.punctuation:
                        # add a space if word does not start with punctuation (eg ')
                        masked_text += ' '
                    masked_text += w
        return masked_text.strip()

    def mask_text(self, text, masks):
        '''
        Return the text with mask applied

        Parameters
        ----------

        text: (str)
            The original text

        masks: (2D np.array)
            The mask to apply, shape (#samples, #words)
        '''
        masked_texts = []
        for i in range(masks.shape[0]):
            if self._use_bow:
                masked_texts.append(self._mask_text_bow(text, masks[i]))
            else:
                masked_texts.append(self._mask_text_nobow(text, masks[i]))
        return masked_texts


class ShapInterpreter:
    '''
    Interpreter based on Kernel SHAP

    Parameters
    ----------

    n_samples: (int, optional: default=10)
        The number of samples to draw from the neighborhood.

    aggregate_method: (fctn handle, optional: default=sum)
        Method to aggregate the scores for words in each sentence

    use_bow: (bool, optional: default=True)
        Wether to use BOW assumption (same word at different parts is 1 occurence)

    link_function: (fctn handle, optional: default=shap.links.identity)
        Link function to apply for shap, see KernelShap documentation for more details
    '''
    def __init__(self, n_samples=10, aggregate_method=sum, use_bow=True, link_function='identity', batch_size=2):
        self.n_samples = n_samples
        self.aggregate_method = aggregate_method
        self.use_bow = use_bow
        self.link = link_function
        # init the shap wrapper
        self.shapwrapper = ShapTextWrapper(use_bow=use_bow)
        self.batch_size = batch_size

    def interpret(self, model, question, context):
        # handler for querying the model
        def has_answer(zs):
            texts = self.shapwrapper.mask_text(context, zs)
            res = []
            for i in range(0, len(texts), self.batch_size):
                res.append(model.predict_proba(
                    [question] * len(texts[i: i + self.batch_size]),
                    texts[i: i + self.batch_size]
                ))
            return np.vstack(res)

        # generate SHAP explanations
        n_words = self.shapwrapper.get_n_words(context)
        shap_values = KernelExplainer(has_answer, np.zeros((1, n_words)), link=self.link).shap_values(np.ones((1, n_words)), nsamples=self.n_samples, silent=True)[1]
        # turn the explanations into queriable form
        exp = dict(zip(self.shapwrapper.get_words(context), shap_values.flatten()))

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
