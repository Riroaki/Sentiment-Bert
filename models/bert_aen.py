import torch
from torch import nn
from .attention import Attention
from .squeeze_embedding import SqueezeEmbedding
from .point_wise_feed_forward import PositionwiseFeedForward


class BERTAEN(nn.Module):
    def __init__(self, bert, config):
        super(BERTAEN, self).__init__()
        self.config = config
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(config.dropout_rate)

        self.attn_k = Attention(config.bert_dim, out_dim=config.hidden_dim,
                                n_head=8,
                                score_function='mlp',
                                dropout=config.dropout_rate)
        self.attn_q = Attention(config.bert_dim, out_dim=config.hidden_dim,
                                n_head=8,
                                score_function='mlp',
                                dropout=config.dropout_rate)
        self.ffn_c = PositionwiseFeedForward(config.hidden_dim,
                                             dropout=config.dropout_rate)
        self.ffn_t = PositionwiseFeedForward(config.hidden_dim,
                                             dropout=config.dropout_rate)

        self.attn_s1 = Attention(config.hidden_dim, n_head=8,
                                 score_function='mlp',
                                 dropout=config.dropout_rate)

        self.dense = nn.Linear(config.hidden_dim * 3, config.polarities_dim)

    def forward(self, inputs):
        context, target = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context, output_all_encoded_layers=False)
        context = self.dropout(context)
        target = self.squeeze_embedding(target, target_len)
        target, _ = self.bert(target, output_all_encoded_layers=False)
        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(
            self.config.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(
            self.config.device)

        hc_mean = torch.div(torch.sum(hc, dim=1),
                            context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1),
                            target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1),
                            context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out
