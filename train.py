import torch
import random
import numpy as np
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel
from models.nlu_model import NLUModelForClassification, NLUModelForMultiChoice
from models.nlu_data_loader import get_dataloader
from models.loss_function import CrossEntropyLoss, BinaryCrossEntropyLoss
from models.trainer import Trainer
from utils.logging import setup_logger

batch_size = 8
task = "classification"
device = "cuda"
path = "./bert_data/"
model_path = "./bert_model/xlm-roberta-base/"
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

    model.to(device)


    # dataloader = SingleBatchNLULoader(path, "train", tokenizer, shuffle=True, device="cuda")
    train_dataloader = get_dataloader(
        path = path, 
        task = task, 
        split = "test", 
        tokenizer = tokenizer, 
        batch_size = batch_size, 
        shuffle = True,
        device = device
    )

    eval_dataloader = get_dataloader(
        path = path, 
        task = task, 
        split = "test", 
        tokenizer = tokenizer, 
        batch_size = batch_size, 
        shuffle = False,
        device = device
    )

    # set up optimizer
    params_no_decay = []
    params_to_decay = []
    params_decoder = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif not name.startswith("bert"):
            params_decoder += [param]
        elif (param.dim() == 1) or name.endswith(".bias"):
            params_no_decay += [param]
        else:
            params_to_decay += [param]

    optimizer_bert = optim.AdamW(
        params = [{"params":params_no_decay, "weight_decay":0.}, {"params":params_to_decay, "weight_decay":1e-4}],
        lr = 5e-4,
        betas = (0.9, 0.999)
    )

    optimizer_dec = optim.AdamW(
        params = [{"params":params_decoder, "weight_decay":0.}],
        lr = 1e-2,
        betas = (0.9, 0.999)
    )

    optimizer = [optimizer_bert, optimizer_dec]
    
    # set up scheduler
    scheduler_bert = optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer_bert,
        T_max = 2,
    )
    scheduler_dec = optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer_dec,
        T_max = 2,
    )
    scheduler = [scheduler_bert, scheduler_dec]
    # set up trainer
    trainer = Trainer(
        model = model,
        loss_func = CrossEntropyLoss(),
        optimizer = optimizer,
        scheduler = scheduler,
        num_epoch = 10,
        train_dataloader = train_dataloader,
        eval_dataloader = eval_dataloader,
        checkpoint_path = None,
        logger = setup_logger(logfile),
    )
    # start training
    trainer.train(run_eval=False)
    torch.save(model.state_dict(), "./bert_model/checkpoint/pytorch_model.bin")

if __name__ == "__main__":
    main()