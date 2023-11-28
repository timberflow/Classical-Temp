import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

class TrainerForPretraining(object):
    def __init__(
        self,
        params,
        model, 
        data, 
        tokenizer
    ) -> None:
        super().__init__
        self.model = model
        self.tokenizer = tokenizer
        self.lang1_data = [item[0] for item in data]
        self.lang2_data = [item[1] for item in data]
        self.lang1_generator = self.build_generator(self.lang1_data)
        self.lang2_generator = self.build_generator(self.lang2_data)
        self.params = params

    def build_generator(self, data):
        return DataIterator(
            data = data,
            tokenizer = self.tokenizer,
            batch_size = self.params.batch_size,
            shuffle = self.params.shuffle,
            device = self.params.device
        )

    def mask_out(self, x):
        """
        Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        bs, slen = x.size()

        # define target words to predict
        if params.sample_alpha == 0:
            pred_mask = np.random.rand(bs, slen) <= params.word_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
        else:
            x_prob = params.mask_scores[x.flatten()]
            n_tgt = math.ceil(params.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(bs, slen)

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        pred_mask[0] = 0  # TODO: remove

        # mask a number of words == 0 [8] (faster with fp16)
        if params.fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
            pred_mask = pred_mask.view(bs, slen)
            assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(params.n_words)
        _x_mask = _x_real.clone().fill_(params.mask_index)
        probs = torch.multinomial(params.pred_probs, len(_x_real), replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (bs, slen)
        assert pred_mask.size() == (bs, slen)

        return x, _x_real, pred_mask
    
    def mlm_step(self, lambda_coeff):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        input_ids, attention_mask = next(self.lang1_generator)
        masked_input, input_ids, pred_mask = self.mask_out(input_ids)
        positions = torch.arange()

        # forward / loss
        tensor = model(masked_input, attention_mask, positions)
        _, loss = model(tensor, pred_mask, input_ids)
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
    
    def tlm_step(self, lang1, lang2, lambda_coeff):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, y, pred_mask = self.mask_out(x)

        # forward / loss
        tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size

class DataIterator(object):
    def __init__(
            self,
            data, 
            tokenizer,
            batch_size,
            max_length,
            shuffle = False, 
            device = "cpu"
        ):
        self.examples = np.asarray(data, dtype = object)
        if shuffle:
            self.shuffle()
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_iterator = self.generate_examples()

    def to(self, device):
        self.device = device

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            try:
                item = next(self.data_iterator)
                batch += [item]
            except StopIteration:
                break

        if not batch:
            raise StopIteration
        
        return self._collate_fn(batch)

    def _reset(self):
        self.data_iterator = self.generate_examples()

    def _collate_fn(self, batch):
        input_ids, attention_mask, labels, query_ids = [], [], [], []
        for item in batch:
            input_ids += [item[0].unsqueeze(0)]
            attention_mask += [item[1].unsqueeze(0)]
            labels += [item[2].unsqueeze(0)]
            query_ids += [item[3]]
        input_ids = torch.cat(input_ids, dim = 0)
        attention_mask = torch.cat(attention_mask, dim = 0)
        labels = torch.cat(labels, dim = 0)
        return {"input": (input_ids, attention_mask), "label": (labels,), "query_ids": query_ids}
    
    def __len__(self):
        length = len(self.nlu_iter.examples)
        return math.ceil(length / self.batch_size)
    
    def shuffle(self):
        permutation_idx = np.random.permutation(len(self.examples))
        self.examples = self.examples[permutation_idx]

    def generate_examples(self):
        for _, example in enumerate(self.examples):
            encoded_dict = self.tokenizer(
                text = example, 
                padding = "max_length", 
                max_length = self.max_length, 
                truncation = True)
            input_ids = encoded_dict["input_ids"]
            attention_mask = encoded_dict["attention_mask"]
            # truncated exmaples will trigger error
            if sum(attention_mask) == self.max_length:
                continue

            yield (
                torch.tensor(input_ids, device = self.device, dtype = torch.int64),
                torch.tensor(attention_mask, device = self.device, dtype = torch.int64),
            )
    