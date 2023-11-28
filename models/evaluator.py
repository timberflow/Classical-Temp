import time
import json
import tqdm
import math
import torch

class Evaluator(object):
    def __init__(self, model, loss_func, eval_dataloader, logger):
        self.model = model
        self.loss_func = loss_func
        self.eval_dataloader = eval_dataloader
        self.logger = logger

    def evalute(self, eps = 1e-6):
        num_train = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        loss_total, ppl_total = 0., 0.
        group_pred = dict()
        epoch_start_time = time.time()
        for i, batch in enumerate(tqdm.tqdm(self.eval_dataloader)):
            logits = self.model(*batch["input"])[0]
            loss = self.loss_func(logits, *batch["label"])

            pred = logits.max(1)[1]
            gold = batch["label"][0]
            prob_pred = logits[:,1].cpu().tolist()
            for i, query_idx in enumerate(batch["query_ids"]):
                if query_idx not in group_pred:
                    group_pred[query_idx] = [(prob_pred[i], gold.cpu().tolist()[i])]
                group_pred[query_idx] += [(prob_pred[i], gold.cpu().tolist()[i])]

            tp += torch.sum((pred == gold) & (pred == 1)).item()
            tn += torch.sum((pred == gold) & (pred == 0)).item()
            fp += torch.sum((pred != gold) & (pred == 1)).item()
            fn += torch.sum((pred != gold) & (pred == 0)).item()

            num_train += 1
            loss_total += loss.item()
            ppl_total += math.exp(loss.item())

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precison = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * precison * recall / (precison + recall + eps)

        n_groups, n_correct = 0, 0
        for _, value in group_pred.items():
            n_groups += 1
            n_correct += int(sorted(value, key=lambda x:-x[0])[0][1] == 1)

        with open("result/result.json", "w", encoding="utf8") as f:
            json.dump(group_pred, f, indent=4, ensure_ascii=False)


        eval_duration = time.time() - epoch_start_time
        self.logger.info(f"Evaluate duration: {eval_duration}, Average loss value: {loss_total / num_train}, average perplexity: {ppl_total / num_train}")
        self.logger.info(f"Accuracy: {accuracy}, Precision: {precison}, Recall: {recall}, F1-score: {f1_score}")
        self.logger.info(f"Accuracy: {accuracy}, Precision: {precison}, Recall: {recall}, F1-score: {f1_score}")
        self.logger.info(f"Group accuracy: {n_correct / n_groups}, Group correct: {n_correct}, Group all: {n_groups}")
                