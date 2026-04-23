import torch.nn as nn
from transformers import BertModel

from .ToMeBertAttention import patch_bert_with_tome


class BertClassifier(nn.Module):
    def __init__(self, num_labels: int, use_tome: bool = False, tome_r: int = 8):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if use_tome:
            self.bert = patch_bert_with_tome(self.bert, r=tome_r)
            print(f"[ToMe ON]  Merging {tome_r} token pairs per layer")
        else:
            print("[ToMe OFF] Standard BERT (no merging)")

        hidden_size = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        # CLS token representation
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.classifier(cls)