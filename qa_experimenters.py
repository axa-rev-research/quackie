# qa_experiemnters.py
# this file contains the code for performing ground-truth evaluation of interpretability approaches on question-answering datasets

import numpy as np
import pandas as pd
import datasets
from tqdm import tqdm
import warnings

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import jaccard_score as iou_score
from sklearn.metrics import precision_score, recall_score

import matplotlib.pyplot as plt
import pickle
import os


def findanswer(segments, answer_start):
    # turn answer starting-character information into ground truth importance
    true = [0] * len(segments)
    chars_remaining = answer_start
    for j, segment in enumerate(segments):
        if chars_remaining < len(segment):
            # the answer starts in the current chunk
            break
        else:
            # subract the number of chars in this chunk, don't forget the space that will follow this sentence
            chars_remaining -= len(segment) + 1
    true[j] = 1
    return true


def snr_score(gt, prediction):
    if len(gt) != len(prediction):
        # non comparable
        return np.nan
    else:
        # make sure prediction and ground truth are np.ndarray
        gt_ = (
            np.array(gt, dtype=bool)
            if not isinstance(gt, np.ndarray)
            else gt.astype(bool)
        )
        prediction_ = (
            np.array(prediction)
            if not isinstance(prediction, np.ndarray)
            else prediction
        )

        # calculate signal and noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            s = prediction_[gt_].mean() - prediction_[np.logical_not(gt_)].mean()
            n = prediction_[np.logical_not(gt_)].std()
        if n == 0:
            return np.nan
        else:
            return (s ** 2) / (n ** 2)


def hpd_score(gt, prediction):
    if len(gt) != len(prediction):
        # non comparable
        return np.nan
    else:
        # make sure prediction and ground truth are np.ndarray
        gt_ = (
            np.array(gt, dtype=bool)
            if not isinstance(gt, np.ndarray)
            else gt.astype(bool)
        )
        prediction_ = (
            np.array(prediction)
            if not isinstance(prediction, np.ndarray)
            else prediction
        )

        pred_gt = prediction_[gt_].min()
        detections_k = prediction_ >= pred_gt

        return precision_score(gt_, detections_k)


class QAExperimenter:
    """
    Experimenter base class for performing ground truth experiments.

    Parameters
    ----------

    model:
        The trained model to be interpreted. This is wrapped in a class which implements the predict and predict_proba function, similar to sklearn classifiers.
        Furhter, it has to give access to the neural network with model.network and to the tokenizer with model.tokenizer.

    interpreter: (fctn handle)
        The interpreter to use, returns a score for each sentence. Sentence segmentation is done with nltk.tokenize.sent_tokenize for the ground truth, so
        your method should return the same number of sentence scores. The interpreter is called as follows: interpreter(model, question, context)

    datset: (datasets.arrow_dataset.Dataset)
        The dataset to use for the experiment
    """

    def __init__(self, model, interpreter, dataset):
        self.model = model
        self.interpreter = interpreter
        self.dataset = dataset
        self.gt = []
        self.prediction = []
        self.sample_id = []

    def experiment(self, maxlength=512):
        """
        Perform the experiment.

        Parameters
        ----------

        maxlength: (int, optional: default=512)
            The maximum sequence length. Default is 512, which is the maximum length possible for the provided, pretrained model.
        """
        self.gt = []
        self.prediction = []
        self.sample_id = []

        for i, sample in enumerate(tqdm(self.dataset)):
            # get the context and question
            context = sample["context"]
            segments = sent_tokenize(context)
            question = (
                sample["question"]
                if type(sample["question"]) == str
                else sample["question"][0]
            )

            # check that sample has appropriate length:
            if (
                len(self.model.tokenizer.encode(question, context, verbose=False))
                > maxlength
            ):
                continue

            # make sure the question has an answer and the model thinks it does
            elif (
                len(sample["answers"]["answer_start"]) == 0
                or self.model.predict(question, context) == 0
            ):
                continue

            # --> we are dealing with a good sample
            # save the possible ground truths
            self.gt.append(
                [
                    findanswer(segments, answer_start)
                    for answer_start in sample["answers"]["answer_start"]
                ]
            )
            # save the sample id
            self.sample_id.append((i, sample["id"]))
            # save the prediction
            self.prediction.append(self.interpreter(self.model, question, context))

    def analyze(self, intperpreter_name, k_max=1):
        """
        Analyze the results after using running experiment. Returns a pandas dataframe with the results.

        Parameters
        ----------

        intperpreter_name : (str)
            Name of the interpreter to be used in the returned dataframe

        k_max : (int, optional: default=2)
            Maximum number of k. The top k segments will be used as prediction. Note that if there are several segments with the same effect,
            inclusion of whom results in more than k_max segments, they are still used.
        """
        # get the scores for all samples
        snrs = []
        hpds = []
        ious = []
        for k in range(1, k_max + 1):
            snrs.append([])
            hpds.append([])
            ious.append([])
            for pred, gts in zip(self.prediction, self.gt):
                # extract the prediction
                if len(pred) <= k:
                    pred_binary = np.ones_like(pred)
                else:
                    kth_largest = np.partition(pred, -k)[-k]
                    pred_binary = (np.array(pred) >= kth_largest).astype(int)
                # find the best corresponding ground truth
                if len(gts[0]) == len(pred_binary):
                    gt = gts[
                        # np.nanargmax([iou_score(t, pred_binary) if len(t)==len(pred_binary) else 0 for t in gts ])
                        np.nanargmax([iou_score(t, pred_binary) for t in gts])
                    ]

                    # get the scores
                    # tps[-1].append(tp_rate(np.array(gt), pred_binary) if len(gt)==len(pred_binary) else 0)
                    # tns[-1].append(tn_rate(np.array(gt), pred_binary) if len(gt)==len(pred_binary) else 0)
                    # ious[-1].append(iou_score(np.array(gt), pred_binary) if len(gt)==len(pred_binary) else 0)
                    snrs[-1].append(snr_score(np.array(gt), pred_binary))
                    hpds[-1].append(hpd_score(np.array(gt), pred_binary))
                    ious[-1].append(iou_score(np.array(gt), pred_binary))
                else:
                    snrs[-1].append(np.nan)
                    hpds[-1].append(np.nan)
                    ious[-1].append(np.nan)

        return pd.DataFrame(
            data={
                "interpreter": intperpreter_name,
                "mean_snr": [np.nanmean(p) for p in snrs],
                "mean_hpd": [np.nanmean(r) for r in hpds],
                "mean_iou": [np.nanmean(iou) for iou in ious],
                "std_snr": [np.nanstd(p) for p in snrs],
                "std_hpd": [np.nanstd(r) for r in hpds],
                "std_iou": [np.nanstd(iou) for iou in ious],
                "k": list(range(1, k_max + 1)),
                "fails": [np.isnan(p).sum() for p in ious],
                "no_snr": [np.isnan(snr).sum() for snr in snrs],
            }
        )

    def test_gt(self, maxlength=512):
        """
        Test the Ground truth.

        Parameters
        ----------

        maxlength: (int, optional: default=512)
            The maximum sequence length. Default is 512, which is the maximum length possible for the provided, pretrained model.
        return_df: (bool, optional: default=False)
            If the dataset of topics should be returned
        """
        delta_pred = []
        delta_proba = []
        delta_pred_only_gt = []
        delta_proba_only_gt = []

        for i, sample in enumerate(tqdm(self.dataset)):
            # get the context and question
            context = sample["context"]
            question = (
                sample["question"]
                if type(sample["question"]) == str
                else sample["question"][0]
            )

            # check that sample has appropriate length:
            if (
                len(self.model.tokenizer.encode(question, context, verbose=False))
                > maxlength
            ):
                continue

            # make sure the question has an answer and the model thinks it does
            elif (
                len(sample["answers"]["answer_start"]) == 0
                or self.model.predict(question, context) == 0
            ):
                continue

            # --> we are dealing with a good sample, get the statistics
            pred_original = self.model.predict(question, context)
            proba_original = self.model.predict_proba(question, context)[:, 1]
            answer_segments = [
                findanswer(sent_tokenize(context), answer_start)
                for answer_start in sample["answers"]["answer_start"]
            ]

            texts = [
                " ".join(
                    [
                        sentence
                        for sentence, is_gt in zip(sent_tokenize(context), ans)
                        if is_gt == 0
                    ]
                )
                for ans in answer_segments
            ]
            preds = self.model.predict([question] * len(answer_segments), texts)
            preds_proba = self.model.predict_proba(
                [question] * len(answer_segments), texts
            )[:, 1]
            delta_pred.append(preds - pred_original)
            delta_proba.append(preds_proba - proba_original)

            texts = [
                " ".join(
                    [
                        sentence
                        for sentence, is_gt in zip(sent_tokenize(context), ans)
                        if is_gt == 1
                    ]
                )
                for ans in answer_segments
            ]
            preds = self.model.predict([question] * len(answer_segments), texts)
            preds_proba = self.model.predict_proba(
                [question] * len(answer_segments), texts
            )[:, 1]
            delta_pred_only_gt.append(preds - pred_original)
            delta_proba_only_gt.append(preds_proba - proba_original)

        return delta_pred, delta_proba, delta_pred_only_gt, delta_proba_only_gt

    def describe(self, maxlength=512, return_df=False, verbose=False):
        """
        Describe the dataset used.

        Parameters
        ----------

        maxlength: (int, optional: default=512)
            The maximum sequence length. Default is 512, which is the maximum length possible for the provided, pretrained model.
        return_df: (bool, optional: default=False)
            If the dataset of topics should be returned
        """
        n_sentences = []
        n_words = []
        titles = []
        n_too_long = 0
        n_noanswer = 0

        for i, sample in enumerate(tqdm(self.dataset, disable=not verbose)):
            # get the context and question
            context = sample["context"]
            question = (
                sample["question"]
                if type(sample["question"]) == str
                else sample["question"][0]
            )

            # check that sample has appropriate length:
            if (
                len(self.model.tokenizer.encode(question, context, verbose=False))
                > maxlength
            ):
                n_too_long += 1
                continue

            # make sure the question has an answer and the model thinks it does
            elif (
                len(sample["answers"]["answer_start"]) == 0
                or self.model.predict(question, context) == 0
            ):
                n_noanswer += 1
                continue

            # --> we are dealing with a good sample, get the statistics
            n_sentences.append(len(sent_tokenize(context)))
            n_words.append(len(word_tokenize(context)))
            titles.append(sample["title"])

        # display the results
        plt.figure(figsize=(12, 6))
        plt.hist(n_sentences, bins=list(range(max(n_sentences) + 2)))
        plt.title("Number of Sentences, mean={:.2f}".format(np.mean(n_sentences)))
        plt.xlabel("Number of Sentences")
        plt.ylabel("Number of Samples")
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.hist(n_words)
        plt.title("Number of Words, mean={:.2f}".format(np.mean(n_words)))
        plt.xlabel("Number of Words")
        plt.ylabel("Number of Samples")
        plt.show()

        df = pd.DataFrame(data={"title": titles, "topics": 1})
        df.groupby("title").count().sort_values("topics", ascending=False).plot.bar(
            legend=False, figsize=(12, 6)
        )
        plt.ylabel("Number of Samples")
        plt.show()

        print("Mean nr of sentences {}".format(np.mean(n_sentences)))
        print("Mean nr of words {}".format(np.mean(n_words)))
        print("There are {} good samples in the dataset".format(len(n_words)))
        print("{} samples were discarded due to being too long".format(n_too_long))
        print(
            "{} samples were discarded due to having no answer/no predicted answer".format(
                n_noanswer
            )
        )
        if return_df:
            return df.groupby("title").count().sort_values("topics", ascending=False)

    def save(self, path="", name="experiment_results"):
        """
        Save the results from the experiment (before analysis)

        Parameters
        ----------

        path: (str, optional: default='')
            Path to the folder where the results should be saved
        name: (str, optional: default='experiment_results')
            Name of the file the results should be saved to (without filetype). Will create 3 files:
            name_gt.pickle, name_prediction.pickle, name_sample_id.pickle
        """
        # save ground truth
        with open(os.path.join(path, name + "_gt.pickle"), "wb") as f:
            pickle.dump(self.gt, f)
        # save prediction
        with open(os.path.join(path, name + "_prediction.pickle"), "wb") as f:
            pickle.dump(self.prediction, f)
        # save sample_id
        with open(os.path.join(path, name + "_sample_id.pickle"), "wb") as f:
            pickle.dump(self.sample_id, f)

    def load(self, path="", name="experiment_results"):
        """
        Load the results from the experiment (before analysis)

        Parameters
        ----------

        path: (str, optional: default='')
            Path to the folder where the results should be loaded from
        name: (str, optional: default='experiment_results')
            Name of the file the results should be loaded from (without filetype). Will load 3 files:
            name_gt.pickle, name_prediction.pickle, name_sample_id.pickle
        """
        # load ground truth
        with open(os.path.join(path, name + "_gt.pickle"), "rb") as f:
            self.gt = pickle.load(f)
        # load prediction
        with open(os.path.join(path, name + "_prediction.pickle"), "rb") as f:
            self.prediction = pickle.load(f)
        # load sample_id
        with open(os.path.join(path, name + "_sample_id.pickle"), "rb") as f:
            self.sample_id = pickle.load(f)


class SQuADExperimenter(QAExperimenter):
    """
    Experimenter class for performing ground truth experiments on SQuAD.

    Parameters
    ----------

    model:
        The trained model to be interpreted. This is wrapped in a class which implements the predict and predict_proba function, similar to sklearn classifiers.
        Furhter, it has to give access to the neural network with model.network and to the tokenizer with model.tokenizer.

    interpreter: (fctn handle)
        The interpreter to use, returns a score for each sentence. Sentence segmentation is done with nltk.tokenize.sent_tokenize for the ground truth, so
        your method should return the same number of sentence scores. The interpreter is called as follows: interpreter(model, question, context)

    split: (int, optional: default=None)
        The number of samples to use. Mainly used for debugging purposes where not the complete dataset is used. None: use the whole dataset.
    """

    def __init__(self, model, interpreter, split=None):
        super().__init__(
            model,
            interpreter,
            datasets.load_dataset(
                "squad_v2",
                split="validation[:{}]".format(split) if split else "validation",
            ),
        )


class SQuADShiftsExperimenter(QAExperimenter):
    """
    Experimenter class for performing ground truth experiments on SQuADShifts dataset, consisting of 4 new test sets for SQuAD from four different domains:
    Wikipedia articles, New York Times articles, Reddit comments, and Amazon product reviews.

    Parameters
    ----------

    model:
        The trained model to be interpreted. This is wrapped in a class which implements the predict and predict_proba function, similar to sklearn classifiers.
        Furhter, it has to give access to the neural network with model.network and to the tokenizer with model.tokenizer.

    interpreter: (fctn handle)
        The interpreter to use, returns a score for each sentence. Sentence segmentation is done with nltk.tokenize.sent_tokenize for the ground truth, so
        your method should return the same number of sentence scores. The interpreter is called as follows: interpreter(model, question, context)

    domain: (str)
        Domain of the dataset. Can be any of the following: ['new_wiki', 'nyt', 'reddit', 'amazon']
    """

    def __init__(self, model, interpreter, domain):
        super().__init__(
            model,
            interpreter,
            datasets.load_dataset("squadshifts", domain, split="test"),
        )

