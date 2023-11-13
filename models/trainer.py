import time
import tqdm
import math
from torch.cuda.amp import GradScaler, autocast

class Trainer(object):
    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        scheduler,
        num_epoch,
        train_dataloader, 
        eval_dataloader,
        checkpoint_path,
        logger,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = num_epoch
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.checkpoint_path = checkpoint_path
        self.logger = logger

    def train(self, run_eval = True):
        # start training
        self.model.train()

        # set up scaler
        # scaler = GradScaler()

        for i in range(self.num_epoch):
            epoch_start_time = time.time()
            num_train = 0
            loss_total, ppl_total = 0., 0.
            for batch in tqdm.tqdm(self.train_dataloader):
                batch_output = self.model(*batch["input"])
                loss = self.loss_func(*batch_output, *batch["label"])
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                num_train += 1
                loss_total += loss.item()
                ppl_total += math.exp(loss.item())

            epoch_duration = time.time() - epoch_start_time
            self.logger.info(f"Epoch {i} finished. Time consumption: {epoch_duration}s.")
            self.logger.info(f"Average loss value: {loss_total / num_train}, average perplexity: {ppl_total / num_train}")

            self.train_dataloader._reset()
        
        if run_eval:
            self.evaluate()

    def evaluate(self):
        pass
                