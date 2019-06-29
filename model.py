import torch.nn as nn


class BERTSPC(nn.Module):
    def __init__(self, bert, config):
        super(BERTSPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dense = nn.Linear(config.bert_dim, config.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
