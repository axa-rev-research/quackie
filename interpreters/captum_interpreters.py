# captum_interpreters.py
# Interpretability with captum
import torch
from nltk.tokenize import sent_tokenize
from Bio import pairwise2 as pw

from captum.attr import (
    IntegratedGradients,
    Saliency,
    NoiseTunnel,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)


class CaptumInterpreter:
    """
    Interpreter based on captum interpreters

    Parameters
    ----------

    captum_interpreter: (uninitialized captum attribution method)
        Attribution from captum. Must implement the .attribute method

    embedding_layer: (str)
        Name of the embedding layer of the nn model

    aggregate_method: (fctn handle, optional: default=sum)
        Method to aggregate the scores for words in each sentence

    kwargs:
        Keyword arguments to be passed to the .attribute method
    """

    def __init__(
        self, captum_interpreter, embedding_layer, aggregate_method=sum, **kwargs
    ):
        self.captum_interpreter = captum_interpreter
        self.aggregate_method = aggregate_method
        self.embedding_layer = embedding_layer
        self.kwargs = kwargs

    def _init_attributor(self, model):
        def f_wrapper(context_embedding, question_embedding, input_dict):
            text_embedding = torch.cat([question_embedding, context_embedding], 1)
            return model.predict_proba_torch(text_embedding, input_dict)

        return self.captum_interpreter(f_wrapper)

    def interpret(self, model, question, context):
        # get the input token ids
        input_dict = model.tokenizer(
            [question], [context], padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        # configure the embedding layer
        interpretable_embedding = configure_interpretable_embedding_layer(
            model.network, self.embedding_layer
        )
        input_embedding = interpretable_embedding.indices_to_embeddings(
            **{k: v for k, v in input_dict.items() if k != "attention_mask"}
        )
        # perform the attribution
        context_start = (
            input_dict["input_ids"] == model.tokenizer.sep_token_id
        ).nonzero()[-2, 1] + 1
        question_embedding = input_embedding[:, :context_start, :]
        context_embedding = input_embedding[:, context_start:, :]

        attribution = (
            torch.sum(
                self._init_attributor(model)
                .attribute(
                    context_embedding,
                    additional_forward_args=(question_embedding, input_dict),
                    **self.kwargs
                )
                .squeeze(),
                1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        # attribute scores to sentences
        # sentence_scores = self._token2sentence_score(attribution, input_dict, model)
        sentence_scores = self._token2sentence_score(
            attribution, input_dict, model, context
        )
        # remove the embedding layer
        remove_interpretable_embedding_layer(model.network, interpretable_embedding)
        return sentence_scores

    def _token2sentence_score(self, attribution, input_dict, model, context):
        context_start = (
            input_dict["input_ids"] == model.tokenizer.sep_token_id
        ).nonzero()[-2, 1] + 1
        context_ids = input_dict["input_ids"].squeeze()[context_start:]
        context_scores = attribution
        context_tokens = model.tokenizer.convert_ids_to_tokens(context_ids)
        # get tokens with sentence segmentation
        sent_tokens = [
            model.tokenizer.convert_ids_to_tokens(model.tokenizer(s)["input_ids"])[1:-1]
            for s in sent_tokenize(context)
        ]
        # find where the splits are and build all-sentence token list
        splits = []
        all_sent_tokens = []
        for s in sent_tokens:
            splits.append(len(all_sent_tokens))
            all_sent_tokens.extend(s)
        # find alignements
        alns = pw.align.globalxs(
            context_tokens, all_sent_tokens, -10, -0.5, gap_char=["[GAP]"]
        )[0]

        # use alignements to attribute scores to sentences
        splits_copy = splits
        sentence_scores = []
        ast_idx = 0
        ct_idx = 0
        for ct, ast in zip(alns.seqA, alns.seqB):
            if ast_idx in splits_copy:
                splits_copy = splits_copy[1:]
                sentence_scores.append([])
            if ct != "[GAP]":
                sentence_scores[-1].append(context_scores[ct_idx])
            ast_idx += int(ast != "[GAP]")
            ct_idx += int(ct != "[GAP]")

        return [self.aggregate_method(scores) for scores in sentence_scores]

    def _token2sentence_score3(self, attribution, input_dict, model, context):
        # extract the context part of input_dict (different for different classification models)
        if (
            str(type(model.network))
            .split("'")[1]
            .startswith("transformers.modeling_roberta")
        ):
            # roberta structure [<s>, question tokens..., '</s>', '</s>', context tokens, </s>]
            was_sep = False
            split = -1
            tokens = model.tokenizer.convert_ids_to_tokens(
                input_dict["input_ids"].squeeze()
            )
            for i, w in enumerate(tokens):
                if w == "</s>" and split < 0:
                    if was_sep:
                        split = i
                    else:
                        was_sep = True
                else:
                    was_sep = False
            context_tokens = tokens[split + 1 :]
            context_scores = attribution[split + 1 :]

        elif (
            str(type(model.network))
            .split("'")[1]
            .startswith("transformers.modeling_albert")
        ):
            # albert structure ['[CLS]', question tokens, '[SEP]' context tokens, '[SEP]']
            tokens = model.tokenizer.convert_ids_to_tokens(
                input_dict["input_ids"].squeeze()
            )
            for i, w in enumerate(tokens):
                if w == "[SEP]":
                    break
            context_tokens = tokens[i + 1 :]
            context_scores = attribution[i + 1 :]

        else:
            raise NotImplementedError(
                "Tokenizer structure parsing not implemented for this model"
            )

        # get tokens with sentence segmentation
        sent_tokens = [
            model.tokenizer.convert_ids_to_tokens(model.tokenizer(s)["input_ids"])[1:-1]
            for s in sent_tokenize(context)
        ]
        # find where the splits are and build all-sentence token list
        splits = []
        all_sent_tokens = []
        for s in sent_tokens:
            splits.append(len(all_sent_tokens))
            all_sent_tokens.extend(s)
        # find alignements
        alns = pw.align.globalxs(
            context_tokens, all_sent_tokens, -10, -0.5, gap_char=["[GAP]"]
        )[0]

        # use alignements to attribute scores to sentences
        splits_copy = splits
        sentence_scores = []
        ast_idx = 0
        ct_idx = 0
        for ct, ast in zip(alns.seqA, alns.seqB):
            if ast_idx in splits_copy:
                splits_copy = splits_copy[1:]
                sentence_scores.append([])
            if ct != "[GAP]":
                sentence_scores[-1].append(context_scores[ct_idx])
            ast_idx += int(ast != "[GAP]")
            ct_idx += int(ct != "[GAP]")

        return [self.aggregate_method(scores) for scores in sentence_scores]


class IgInterpreter(CaptumInterpreter):
    """
    Integrated Gradient Interpreter

    Parameters
    ----------

    embedding_layer: (str)
        Name of the embedding layer of the nn model

    aggregate_method: (fctn handle, optional: default=sum)
        Method to aggregate the scores for words in each sentence

    kwargs:
        Keyword arguments to be passed to the .attribute method
    """

    def __init__(self, embedding_layer, aggregate_method=sum, **kwargs):
        super().__init__(
            IntegratedGradients,
            embedding_layer,
            aggregate_method=aggregate_method,
            **kwargs
        )


class SaliencyInterpreter(CaptumInterpreter):
    """
    Saliency Interpreter

    Parameters
    ----------

    embedding_layer: (str)
        Name of the embedding layer of the nn model

    aggregate_method: (fctn handle, optional: default=sum)
        Method to aggregate the scores for words in each sentence

    kwargs:
        Keyword arguments to be passed to the .attribute method
    """

    def __init__(self, embedding_layer, aggregate_method=sum, **kwargs):
        super().__init__(
            Saliency, embedding_layer, aggregate_method=aggregate_method, **kwargs
        )


class SmoothGradInterpreter(CaptumInterpreter):
    """
    SmoothGrad/NoiseTunnel Interpreter

    Parameters
    ----------

    embedding_layer: (str)
        Name of the embedding layer of the nn model

    aggregate_method: (fctn handle, optional: default=sum)
        Method to aggregate the scores for words in each sentence

    kwargs:
        Keyword arguments to be passed to the .attribute method
    """

    def __init__(self, embedding_layer, aggregate_method=sum, **kwargs):
        super().__init__(
            NoiseTunnel, embedding_layer, aggregate_method=aggregate_method, **kwargs
        )

    def _init_attributor(self, model):
        return self.captum_interpreter(Saliency(model.predict_proba_torch))
