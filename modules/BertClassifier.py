import copy
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .ToMeBertAttention import patch_bert_with_tome

class BertClassifier(nn.Module):
    def __init__(self, num_labels: int, use_tome: bool = False, tome_r: int = 8, model_name: str = "bert-base-uncased", pretrained_model: Optional[BertModel] = None):
        super().__init__()
        self.bert = copy.deepcopy(pretrained_model) if pretrained_model else BertModel.from_pretrained(model_name)

        if use_tome:
            self.bert = patch_bert_with_tome(self.bert, r=tome_r)
            print(f"[ToMe ON] Merging {tome_r} token pairs")
        else: print("[ToMe OFF] Standard BERT")

        hidden_size = self.bert.config.hidden_size 
        
        self.journal_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.paper_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.1))
        self.fusion_linear = nn.Sequential(nn.Linear(hidden_size + 1, hidden_size), nn.ReLU())
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, p_ids, p_mask, j_ids, j_mask):
        feat_paper = self.paper_projection(self.bert(input_ids=p_ids, attention_mask=p_mask).last_hidden_state[:, 0, :])
        feat_journal = self.journal_projection(self.bert(input_ids=j_ids, attention_mask=j_mask).last_hidden_state[:, 0, :])
        
        cosine_sim = F.cosine_similarity(feat_paper, feat_journal, dim=1).unsqueeze(1)
        return self.classifier(self.fusion_linear(torch.cat((feat_paper, cosine_sim), dim=1)))