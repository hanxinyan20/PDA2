from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from batchbald_redux.consistent_mc_dropout import BayesianModule, ConsistentMCDropout
from torch import nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss   
class MCDropoutBertForSequenceClassification(BayesianModule):
    def __init__(self, dropout_rate, model:BertForSequenceClassification):
        super().__init__()
        self.model = model
        self.model.dropout = ConsistentMCDropout(p=dropout_rate)
    
    def mc_forward_impl(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )[1]
    def forward(self, inputs, k):
        BayesianModule.k = k
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs['labels']
        
        mc_input_BK = BayesianModule.mc_tensor(input_ids, k)
        mc_attention_mask_BK = BayesianModule.mc_tensor(attention_mask, k)
        mc_token_type_ids_BK = BayesianModule.mc_tensor(token_type_ids, k)
        mc_labels_BK = BayesianModule.mc_tensor(labels, k)
        mc_output_BK = self.mc_forward_impl(
            input_ids=mc_input_BK,
            attention_mask=mc_attention_mask_BK,
            token_type_ids=mc_token_type_ids_BK,
            labels=mc_labels_BK,
        )
        mc_output_B_K = BayesianModule.unflatten_tensor(mc_output_BK, k)
        return mc_output_B_K
        