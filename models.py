# models.py
# classification models to use

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np


def _matrix_softmax(matrix):
    expd = torch.exp(matrix)
    return expd / expd.sum(dim=(1, 2), keepdim=True)


class Model_Classification:
    '''
    Model based on Classifier directly trained on predicting if an answer exists. Default model is
    https://huggingface.co/a-ware/roberta-large-squad-classification
    '''
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.network = AutoModelForSequenceClassification.from_pretrained("a-ware/roberta-large-squad-classification").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("a-ware/roberta-large-squad-classification")

    def predict(self, question, context):
        '''
        Predict class

        Parameters
        ----------

        question: (str or list(str))
            The questions

        context: (str or list(str))
            The contexts used to answer the questions
        '''
        input_dict = self.tokenizer(question, context, padding=True, truncation=True, return_tensors='pt').to(self.device)
        return self.network(**input_dict)[0].argmax(axis=1).cpu().numpy()

    def predict_proba(self, question, context):
        '''
        Predict class probability (numpy output)

        Parameters
        ----------

        question: (str or list(str))
            The questions

        context: (str or list(str))
            The contexts used to answer the questions
        '''
        input_dict = self.tokenizer(question, context, padding=True, truncation=True, return_tensors='pt').to(self.device)
        logits = self.network(**input_dict)[0].detach().cpu().numpy()
        return softmax(logits, axis=1)

    def predict_proba_torch(self, input_data, input_dict):
        '''
        Predict class probabilities (torch output with gradient), only to be used in conjunction with interpretable embeddings layer from captum

        Parameters
        ----------

        input_data:
            Input for torch model
        '''
        logits = self.network(input_data, attention_mask=input_dict['attention_mask'])[0]
        return torch.nn.functional.softmax(logits)[:, 1]


class Model_QA:
    '''
    Model based on Question Answering model. Default model is
    https://huggingface.co/twmkn9/albert-base-v2-squad2?text=What%27s+my+name%3F&context=My+name+is+Clara+and+I+live+in+Berkeley.
    '''
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.network = AutoModelForQuestionAnswering.from_pretrained("twmkn9/albert-base-v2-squad2").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")

    def predict(self, question, context):
        '''
        Predict class

        Parameters
        ----------

        question: (str or list(str))
            The questions

        context: (str or list(str))
            The contexts used to answer the questions
        '''
        return (self.predict_proba(question, context)[:, 1] > 0.5).astype(int)

    def predict_proba_torch(self, input_data, input_dict):
        '''
        Predict class probability (torch output with gradient), only to be used in conjunction with interpretable embeddings layer from captum

        Parameters
        ----------

        input_data:
            Input for torch model
        '''
        start_scores, end_scores = self.network(input_data, attention_mask=input_dict['attention_mask'])
        ps = _matrix_softmax(start_scores.unsqueeze(2).transpose(1, 2) + end_scores.unsqueeze(2))
        ltri = torch.tril(ps)
        for i, ids in enumerate(input_dict['token_type_ids']):
            ltri[i, :, (torch.nonzero(ids.squeeze())[0][0]):] = 0
        negative_answers = torch.triu(ps, diagonal=1).sum(dim=(1, 2))  # create the upper triangular matrix of invalid spans (end<start)
        answer_in_question = ltri.sum(dim=(1, 2))  # extract all the answers starting in the question
        P_answer = 1 - negative_answers - answer_in_question
        return P_answer

    def predict_proba(self, question, context):
        '''
        Predict class probability (numpy output)

        Parameters
        ----------

        question: (str or list(str))
            The questions

        context: (str or list(str))
            The contexts used to answer the questions
        '''
        input_dict = self.tokenizer(question, context, padding=True, truncation=True, return_tensors='pt').to(self.device)
        start_scores, end_scores = self.network(**input_dict)
        ps = _matrix_softmax(start_scores.unsqueeze(2).transpose(1, 2) + end_scores.unsqueeze(2))
        ltri = torch.tril(ps)
        for i, ids in enumerate(input_dict['token_type_ids']):
            ltri[i, :, (torch.nonzero(ids.squeeze())[0][0]):] = 0
        negative_answers = torch.triu(ps, diagonal=1).sum(dim=(1, 2))  # create the upper triangular matrix of invalid spans (end<start)
        answer_in_question = ltri.sum(dim=(1, 2))  # extract all the answers starting in the question
        P_answer = (1 - negative_answers - answer_in_question).detach().cpu().numpy()
        return np.vstack((1 - P_answer, P_answer)).T
