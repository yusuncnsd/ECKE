import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ner_unit(nn.Module):
    def __init__(self, args, ner2idx, ner_hidden_size,share_hidden_size):
        super(ner_unit, self).__init__()

        self.ner2idx = ner2idx
        self.hid2hid = nn.Linear(ner_hidden_size * 2 + share_hidden_size, ner_hidden_size)
        self.hid2tag = nn.Linear(ner_hidden_size, len(ner2idx))
        self.elu = nn.ELU()
        self.ln = nn.LayerNorm(ner_hidden_size)
        self.dropout = nn.Dropout(args.dropout)



    def forward(self, h_ner, h_share, mask):
        length, batch_size, _ = h_ner.size()


        st = h_ner.unsqueeze(1).repeat(1, length, 1, 1)
        en = h_ner.unsqueeze(0).repeat(length, 1, 1, 1)

        ner = torch.cat((st, en, h_share), dim=-1)


        ner = self.ln(self.hid2hid(ner))
        ner = self.elu(self.dropout(ner))
        ner = torch.sigmoid(self.hid2tag(ner))

        diagonal_mask = torch.triu(torch.ones(batch_size, length, length)).to(device)
        diagonal_mask = diagonal_mask.permute(1, 2, 0)

        mask_s = mask.unsqueeze(1).repeat(1, length, 1)
        mask_e = mask.unsqueeze(0).repeat(length, 1, 1)

        mask_ner = mask_s * mask_e
        mask = diagonal_mask * mask_ner
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, len(self.ner2idx))

        ner = ner * mask

        return ner

class re_unit(nn.Module):
    def __init__(self, args, re2idx,re_hidden_size,share_hidden_size):
        super(re_unit, self).__init__()
        self.relation_size = len(re2idx)
        self.re2idx = re2idx

        self.hid2hid = nn.Linear(re_hidden_size * 2+share_hidden_size, re_hidden_size)
        self.hid2rel = nn.Linear(re_hidden_size, self.relation_size)
        self.elu = nn.ELU()
        self.ln = nn.LayerNorm(re_hidden_size)

        self.dropout = nn.Dropout(args.dropout)



    def forward(self, h_re, h_share, mask):
        length, batch_size, _ = h_re.size()

        r1 = h_re.unsqueeze(1).repeat(1, length, 1, 1)
        r2 = h_re.unsqueeze(0).repeat(length, 1, 1, 1)

        re = torch.cat((r1, r2, h_share), dim=-1)

        re = self.ln(self.hid2hid(re))
        re = self.elu(self.dropout(re))
        re = torch.sigmoid(self.hid2rel(re))

        mask = mask.unsqueeze(-1).repeat(1, 1, self.relation_size)
        mask_e1 = mask.unsqueeze(1).repeat(1, length, 1, 1)
        mask_e2 = mask.unsqueeze(0).repeat(length, 1, 1, 1)
        mask = mask_e1 * mask_e2

        re = re * mask

        return re


class ConvAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pre_channels, channels, groups, dropout=0.1):
        super(ConvAttentionLayer, self).__init__()
        assert hid_dim % n_heads == 0
        self.n_heads = n_heads
        input_channels = hid_dim * 2 + pre_channels
        self.groups = groups

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.linear1 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.linear2 = nn.Linear(hid_dim, hid_dim, bias=False)

        self.conv = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_channels, channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.score_layer = nn.Conv2d(channels, n_heads, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, pre_conv=None, mask=None, residual=True, self_loop=True):
        ori_x, ori_y = x, y

        B, M, _ = x.size()
        B, N, _ = y.size()

        fea_map = torch.cat([x.unsqueeze(2).repeat_interleave(N, 2), y.unsqueeze(1).repeat_interleave(M, 1)],
                            -1).permute(0, 3, 1, 2).contiguous()
        if pre_conv is not None:
            fea_map = torch.cat([fea_map, pre_conv], 1)
        fea_map = self.conv(fea_map)

        scores = self.activation(self.score_layer(fea_map))

        if mask is not None:
            mask = mask.expand_as(scores)
            scores = scores.masked_fill(mask.eq(0), -9e10)

        x = self.linear1(self.dropout(x))
        y = self.linear2(self.dropout(y))
        out_x = torch.matmul(F.softmax(scores, -1), y.view(B, N, self.n_heads, -1).transpose(1, 2))
        out_x = out_x.transpose(1, 2).contiguous().view(B, M, -1)
        out_y = torch.matmul(F.softmax(scores.transpose(2, 3), -1), x.view(B, M, self.n_heads, -1).transpose(1, 2))
        out_y = out_y.transpose(1, 2).contiguous().view(B, N, -1)

        if self_loop:
            out_x = out_x + x
            out_y = out_y + y

        out_x = self.activation(out_x)
        out_y = self.activation(out_y)

        if residual:
            out_x = out_x + ori_x
            out_y = out_y + ori_y
        return out_x, out_y, fea_map


class ConvAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, pre_channels, channels, layers, groups, dropout):
        super(ConvAttention, self).__init__()

        self.layers = nn.ModuleList([ConvAttentionLayer(hid_dim, n_heads, pre_channels if i == 0 else channels,
                                                        channels, groups, dropout=dropout) for i in range(layers)])

    def forward(self, x, y, fea_map=None, mask=None, residual=True, self_loop=True):
        for layer in self.layers:
            x, y, fea_map = layer(x, y, fea_map, mask, residual, self_loop)

        return x, y, fea_map.permute(0, 2, 3, 1).contiguous()



class CARE(nn.Module):
    def __init__(self, args, input_size, ner2idx, rel2idx):
        super(CARE, self).__init__()
        self.args = args

        self.ner = ner_unit(args, ner2idx,ner_hidden_size=args.hidden_size,share_hidden_size=args.share_hidden_size)
        self.re = re_unit(args, rel2idx,re_hidden_size=args.hidden_size,share_hidden_size=args.share_hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.dist_emb = nn.Embedding(20, args.dist_emb_size)

        if args.embed_mode == 'bert_cased':
            self.tokenizer = AutoTokenizer.from_pretrained("pretrained/bert")
            self.bert = AutoModel.from_pretrained("pretrained/bert")
        elif args.embed_mode == 'scibert_cased':
            self.tokenizer = AutoTokenizer.from_pretrained("pretrained/scibert",do_lower_case=False)
            self.bert = AutoModel.from_pretrained("pretrained/scibert")


        self.conv_attention = ConvAttention(hid_dim=args.hidden_size, n_heads=1, pre_channels=args.dist_emb_size, channels=args.share_hidden_size, groups=1, layers=args.co_attention_layers, dropout=args.dropout)


    def forward(self, x, mask, dist):
        x = self.tokenizer(x, return_tensors="pt",
                                  padding='longest',
                                  is_split_into_words=True).to(device)
        x = self.bert(**x)[0]

        if self.training:
            x = self.dropout(x)
        length = x.size(1)
        dist = self.dist_emb(dist).permute(0,3,1,2)


        padding_mask = mask.unsqueeze(-1)
        mask1 = padding_mask.unsqueeze(1).repeat(1, length, 1, 1)
        mask2 = padding_mask.unsqueeze(0).repeat(length, 1, 1, 1)
        padding_mask = mask1 * mask2

        padding_mask = padding_mask.permute(2,3,0,1)

        h_ner, h_re, h_share = self.conv_attention(x, x, dist, padding_mask)
        h_ner = h_ner.permute(1,0,2)
        h_re = h_re.permute(1,0,2)
        h_share = h_share.permute(1,2,0,3)
        ner_score = self.ner(h_ner, h_share, mask)
        re_core = self.re(h_re, h_share, mask)
        return ner_score, re_core

