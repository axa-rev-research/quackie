# baseline_interpreter.py
# Interpreter baseline which always gives a random attribution.

import numpy as np
from nltk.tokenize import sent_tokenize


def random_interpreter(model, question, context):
    # create a random ranking for the sentences
    sentences = sent_tokenize(context)
    scores = np.arange(len(sentences))
    np.random.shuffle(scores)
    return scores


if __name__ == "__main__":
    # give an example
    text = 'Hello! This is a test. It is working.'
    print(text)
    print(random_interpreter(None, None, text))
