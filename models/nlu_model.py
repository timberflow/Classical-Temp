import torch
import torch.nn as nn

class NLUModelForMultiChoice(nn.Module):
    def __init__(self, bert_model) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.dense = nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids, attention_mask)[0]
        output_vec = output[:,0,:]
        logits = self.dense(output_vec)
        output = (logits,)
        return output
    
class NLUModelForClassification(nn.Module):
    def __init__(self, bert_model) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.dense = nn.Linear(bert_model.config.hidden_size, 2)
        self.actv = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids, attention_mask)[0]
        output_vec = torch.mean(output * attention_mask[:,:,None].float(), dim = 1)
        logits = self.dense(output_vec)
        # logits = self.actv(logits)
        output = (logits,)
        
        return output