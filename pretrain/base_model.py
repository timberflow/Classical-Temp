import torch
import torch.nn as nn

class BaseModel(object):
    def __init__(self, base_model) -> None:
        super().__init__()
        self.bert_model = base_model
        pass

    def merge_padding_tensors(self, tensor_1, mask_1, tensor_2, mask_2, max_length = 514):
        batch_size, _, d_model = tensor_1.size()
        output_tensor = torch.zeros(batch_size, max_length, d_model).to(tensor_1.device)
        for i in range(batch_size):
            length1 = torch.sum(mask_1[i])
            length2 = torch.sum(mask_2[i])
            tensor_1[i] = tensor_1[i][:,:length1,:]
            tensor_2[i] = tensor_2[i][:,:length2,:]
            output_tensor[i][:,:length1 + length2,:] = torch.cat((tensor_1[i], tensor_2[i]), 1)
        return output_tensor

    def forward(self, train_method, inputs, attention_masks, pred_ids):
        if train_method == "mlm":
            input_ids = inputs[0]
            output_vec = self.bert_model(input_ids, attention_masks[0])[0]
            output_logits = output_vec.index_select(1, pred_ids)
            return output_logits
        elif train_method == "tlm":
            lang1_ids, lang2_ids = inputs
            mask_1, mask_2 = attention_masks
            token_embedding_1 = self.bert_model.embedding(lang1_ids)
            token_embedding_2 = self.bert_model.embedding(lang2_ids)
            embedding = self.merge_padding_tensors(token_embedding_1, mask_1, token_embedding_2, mask_2)
            output_vec = self.bert_model.encoder(embedding)
            output_logits = output_vec.index_select(1, pred_ids)
            return output_logits