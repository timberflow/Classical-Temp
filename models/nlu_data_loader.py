import os
import re
import json
import glob
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader

def load_data_for_multichoice(path):
    example_data = []
    for file in glob.glob(path + "/bert_data.[0-9].json"):
        with open(file, "r", encoding = "utf8") as f:
            example_data += json.load(f)
    return (example_data, None)

def load_data_for_classification(file_path, query_file = "./bert_data/query.json"):
    with open(file_path, "r", encoding = "utf8") as f:
        example_data = json.load(f)
    with open(query_file, "r", encoding = "utf8") as f:
        query_data = json.load(f)
    return (example_data, query_data)

def query_simplify(cql):
    reg = r"(MATCH|WHERE|RETURN|WITH|ORDER BY|LIMIT)"
    splits = re.split(reg, cql)
    output_str = ""
    for i in range(len(splits)):
        if i >= 1 and splits[i-1] in ("MATCH", "RETURN"):
            output_str += " "
        else:
            output_str += splits[i]
    return output_str

    
class DataLoader(object):
    def __init__(self, data_iterator, batch_size):
        self.data_iterator = data_iterator
        self.batch_size = batch_size
    
    def __len__(self):
        raise NotImplementedError
    
    def _reset(self):
        raise NotImplementedError
    
    def __next__(self):
        raise NotImplementedError
    
    def __iter__(self):
        return self
    
    
class SingleBatchNLULoader(DataLoader):
    def __init__(
            self,
            data, 
            tokenizer,
            shuffle = False, 
            sample_rate = 1., 
            device = "cpu"
        ):
        self.nlu_iter = NLUIterator(
            data, 
            tokenizer, 
            shuffle, 
            sample_rate, 
            device
        )

        super(SingleBatchNLULoader, self).__init__(self.nlu_iter.generate_example_groups(), 1)

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
    
    def __len__(self):
        return len(self.nlu_iter.examples)
    

class NLULoader(DataLoader):
    def __init__(
            self,
            data, 
            tokenizer,
            batch_size,
            shuffle = False, 
            sample_rate = 1., 
            device = "cpu"
        ):
        self.nlu_iter = NLUIterator(
            data,
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
    
class NLUIterator(object):
    def __init__(self, examples, tokenizer, shuffle, sample_rate, device):
        self.tokenizer = tokenizer
        self.device = device
        
        self.examples = examples[0]
        self.queries = examples[1]
        self.examples = np.asarray(self.examples, dtype=object)

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
        for i, example in enumerate(self.examples):
            query_idx = example["query_idx"]
            query = self.queries[query_idx]
            cypher = example["cypher"]
            label = example["label"]
            
            cypher = query_simplify(cypher)
            encoded_dict = self.tokenizer(
                text = f"Query: {query} Cypher: {cypher}", 
                padding = "max_length", 
                max_length = 514, 
                truncation = True
            )
            input_ids = encoded_dict["input_ids"]
            attention_mask = encoded_dict["attention_mask"]
            # truncated exmaples will trigger error
            if sum(attention_mask) == 514:
                continue

            yield (
                torch.tensor(input_ids, device = self.device, dtype = torch.int64),
                torch.tensor(attention_mask, device = self.device, dtype = torch.int64),
                torch.tensor(label, device = self.device, dtype = torch.int64),
                query_idx
            )

def get_dataloader(path, task, split, tokenizer, batch_size = 1, shuffle = True, sample_rate = 1., device = "cpu"):
    if task == "classification":
        json_data = load_data_for_classification(path + f"{split}/{split}.json")
        dataloader = NLULoader(
            data = json_data,
            tokenizer = tokenizer,
            batch_size = batch_size,
            shuffle = shuffle,
            sample_rate = sample_rate,
            device = device
        )
    elif task == "multichoice":
        json_data = load_data_for_multichoice(path)
        dataloader = SingleBatchNLULoader(
            data = json_data,
            tokenizer = tokenizer,
            shuffle = shuffle,
            sample_rate = sample_rate,
            device = device
        )
    else:
        raise ValueError(f"Undefined task '{task}'")
        
    return dataloader
    