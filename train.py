import torch
import random
import numpy as np
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel
from models.nlu_model import NLUModelForClassification, NLUModelForMultiChoice
from models.nlu_data_loader import get_dataloader
from models.loss_function import CrossEntropyLoss, MultiChoiceCrossEntropyLoss
from models.trainer import Trainer
from utils.logging import setup_logger

def set_seed(seed = 7):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

batch_size = 8
task = "classification"
device = "cuda"
path = "./bert_data"
model_path = "./bert_model/xlm-roberta-base/"
ckpt_path = "./bert_model/checkpoint/"
logfile = "./log/logfile.txt"

def main():
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    # model = NLUModelForMultiChoice(model)
    model = NLUModelForClassification(model)
    model.to(device)


    # dataloader = SingleBatchNLULoader(path, "train", tokenizer, shuffle=True, device="cuda")
    dataloader = get_dataloader(
        path = path, 
        task = task, 
        split = "train", 
        tokenizer = tokenizer, 
        batch_size = batch_size, 
        device = device
    )

    # set up optimizer
    params_no_decay = []
    params_to_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif (param.dim() == 1) or name.endswith(".bias"):
            params_no_decay += [param]
        else:
            params_to_decay += [param]

    optimizer = optim.AdamW(
        params = [{"params":params_no_decay, "weight_decay":0.}, {"params":params_to_decay, "weight_decay":1e-4}],
        lr = 5e-4,
        betas = (0.9, 0.999)
    )
    # set up scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer,
        T_max = 1,
    )
    # set up trainer
    trainer = Trainer(
        model = model,
        loss_func = CrossEntropyLoss(),
        optimizer = optimizer,
        scheduler = scheduler,
        num_epoch = 5,
        train_dataloader = dataloader,
        eval_dataloader = None,
        checkpoint_path = None,
        logger = setup_logger(logfile),
    )
    # start training
    trainer.train()

    torch.save(model.state_dict(), ckpt_path + "pytorch_model.bin")

if __name__ == "__main__":
    main()