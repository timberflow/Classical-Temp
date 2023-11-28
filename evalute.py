import torch
import random
import numpy as np

from transformers import AutoTokenizer, AutoModel
from models.nlu_model import NLUModelForClassification
from models.nlu_data_loader import get_dataloader
from models.loss_function import CrossEntropyLoss
from models.evaluator import Evaluator
from utils.logging import setup_logger

batch_size = 8
task = "classification"
device = "cuda"
path = "./bert_data/"
model_path = "./bert_model/xlm-roberta-base/"
ckpt = "./bert_model/checkpoint/pytorch_model.bin"
logfile = "./log/logfile.txt"

def set_seed(seed = 7):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main():
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModel.from_pretrained(model_path)
    # model = NLUModelForMultiChoice(model)
    model = NLUModelForClassification(model)

    if ckpt:
        state_dict = torch.load(ckpt)
        model.load_state_dict(state_dict)
    model.to(device)

    eval_dataloader = get_dataloader(
        path = path, 
        task = task, 
        split = "test", 
        tokenizer = tokenizer, 
        batch_size = batch_size, 
        shuffle = False,
        device = device
    )
    # set up evaluator
    evaluator = Evaluator(
        model = model,
        loss_func = CrossEntropyLoss(),
        eval_dataloader = eval_dataloader,
        logger = setup_logger(logfile)
    )
    # start evaluation
    evaluator.evalute()

if __name__ == "__main__":
    main()