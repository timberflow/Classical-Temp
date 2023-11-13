import os
import json
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader

def load_data(path, task):
    examlpe_data = []
    for file in glob.glob(path + "/bert_data.[0-9].json"):
        with open(file, "r", encoding = "utf8") as f:
            if task == "multichoice":
                examlpe_data += json.load(f)
            elif task == "classify":
                groups = json.load(f)
                for group in groups:
                    examlpe_data += [{"query": group["query"], "cypher": line[0], "label": line[1]} for line in group["candidates"]]

    return examlpe_data

    
class DataLoader(object):
    def __init__(self, data_iterator, batch_size):
        self.data_iterator = data_iterator
        self.batch_size = batch_size

    
    def __len__(self):
        raise NotImplementedError
    
    def _reset(self):
        raise NotImplementedError
    
    def _load_data(self):
        raise NotImplementedError
    
    def __next__(self):
        raise NotImplementedError
    
    def __iter__(self):
        return self
    
    
class SingleBatchNLULoader(DataLoader):
    def __init__(
            self,
            path, 
            tokenizer,
            shuffle = False, 
            sample_rate = 1., 
            device = "cpu"
        ):
        self.nlu_iter = NLUIterator(
            path, 
            self._load_data(path), 
            tokenizer, 
            shuffle, 
            sample_rate, 
            device
        )

        super(SingleBatchNLULoader, self).__init__(self.nlu_iter.generate_example_groups())

    def to(self, device):
        self.nlu_iter.device = device

    def __next__(self):
        mini_batch = None
        try:
            item = next(self.data_iterator)
            mini_batch = item
        except StopIteration:
            pass
        if not mini_batch:
            raise StopIteration
        
        return mini_batch

    def _reset(self):
        self.data_iterator = self.nlu_iter.generate_example_groups()

    def _load_data(self, path):
        examlpe_data = []
        for file in glob.glob(path + "/bert_data.[0-9].json"):
            with open(file, "r", encoding = "utf8") as f:
                examlpe_data += json.load(f)
        return examlpe_data
    
    def __len__(self):
        return len(self.nlu_iter.examples)
    

class NLULoader(DataLoader):
    def __init__(
            self,
            path, 
            tokenizer,
            batch_size,
            shuffle = False, 
            sample_rate = 1., 
            device = "cpu"
        ):
        self.nlu_iter = NLUIterator(
            path, 
            self._load_data(path), 
            tokenizer, 
            shuffle, 
            sample_rate, 
            device
        )
        self.device = device
        super(NLULoader, self).__init__(self.nlu_iter.generate_examples(), batch_size)

    def to(self, device):
        self.nlu_iter.device = device

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
        self.data_iterator = self.nlu_iter.generate_examples()

    def _load_data(self, path):
        examlpe_data = []
        for file in glob.glob(path + "/bert_data.[0-9].json"):
            with open(file, "r", encoding = "utf8") as f:
                groups = json.load(f)
                for group in groups:
                    examlpe_data += [{
                        "query": group["query"], 
                        "cypher": line[0], 
                        "label": line[1]
                        } for line in group["candidates"]]

        return examlpe_data

    def _collate_fn(self, batch):
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            input_ids += [item[0].unsqueeze(0)]
            attention_mask += [item[1].unsqueeze(0)]
            labels += [item[2].unsqueeze(0)]
        input_ids = torch.cat(input_ids, dim = 0)
        attention_mask = torch.cat(attention_mask, dim = 0)
        labels = torch.cat(labels, dim = 0)
        return {"input": (input_ids, attention_mask), "label": (labels,)}
    
    def __len__(self):
        length = len(self.nlu_iter.examples)
        return math.ceil(length / self.batch_size)
    
class NLUIterator(object):
    def __init__(self, path, examples, tokenizer, shuffle, sample_rate, device):
        self.path = path
        self.tokenizer = tokenizer
        self.device = device

        self.examples = np.asarray(examples)

        if shuffle:
            self.shuffle()

        if sample_rate < 1.:
            length = len(self.examples)
            self.examples = self.examples[:int(sample_rate * length)]
        
    def shuffle(self):
        permutation_idx = np.random.permutation(len(self.examples))
        self.examples = self.examples[permutation_idx]

        
    def generate_example_groups(self):
        for example in self.examples:
            query = example["query"]
            candidate_labels = example["candidates"]

            input_ids = []
            attention_mask = []
            labels = []

            for cypher, label, _ in candidate_labels:
                encoded_dict = self.tokenizer(
                    f"Query: {query} Cypher: {cypher}", padding = "max_length", max_length = 514, truncation = True)
                input_ids += [encoded_dict["input_ids"]]
                attention_mask += [encoded_dict["attention_mask"]]
                labels += [label]

            target_prob = np.array(labels)
            target_prob = target_prob / target_prob.sum()

            yield (
                torch.tensor(input_ids, device = self.device, dtype = torch.int64),
                torch.tensor(attention_mask, device = self.device, dtype = torch.int64),
                torch.tensor(target_prob, device = self.device, dtype = torch.float32),
            )

    def generate_examples(self):
        for example in self.examples:
            query = example["query"]
            cypher = example["cypher"]
            label = example["label"]

            encoded_dict = self.tokenizer(
                f"Query: {query} Cypher: {cypher}", padding = "max_length", max_length = 514, truncation = True)
            input_ids = encoded_dict["input_ids"]
            attention_mask = encoded_dict["attention_mask"]

            yield (
                torch.tensor(input_ids, device = self.device, dtype = torch.int64),
                torch.tensor(attention_mask, device = self.device, dtype = torch.int64),
                torch.tensor(label, device = self.device, dtype = torch.int64),
            )

def get_dataloader(path, task, split, tokenizer, batch_size = 1, shuffle = True, sample_rate = 1., device = "cpu"):
    match task:
        case "classification":
            dataloader = NLULoader(
                path = path,
                tokenizer = tokenizer,
                batch_size = batch_size,
                shuffle = shuffle,
                sample_rate = sample_rate,
                device = device
            )
        case "multichoice":
            dataloader = SingleBatchNLULoader(
                path = path,
                tokenizer = tokenizer,
                shuffle = shuffle,
                sample_rate = sample_rate,
                device = device
            )
        case _:
            raise ValueError(f"Undefined task '{task}'")
        
    return dataloader
    