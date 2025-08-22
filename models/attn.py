import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
from utils.masking import TriangularCausalMask


import numpy as np

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)




#/......................NPCC....................../

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _npcc_score(self, Q, K):
        Q_mean = Q.mean(dim=-1, keepdim=True)
        K_mean = K.mean(dim=-1, keepdim=True)
        Q0 = Q - Q_mean
        K0 = K - K_mean
        cov = torch.einsum("bhid,bhjd->bhij", Q0, K0)
        normQ = torch.norm(Q0, dim=-1, keepdim=True)
        normK = torch.norm(K0, dim=-1, keepdim=True)
        denom = normQ * normK.transpose(-2, -1) + 1e-8
        return torch.abs(cov / denom)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        U = self.factor * math.ceil(math.log(L))
        U = min(U, S)

        Q = queries.permute(0, 2, 1, 3) 
        K = keys.permute(0, 2, 1, 3)  
        V = values.permute(0, 2, 1, 3)  

        index_sample = torch.randint(low=0, high=S, size=(B, H, L, U), device=queries.device)
        K_sample = K.unsqueeze(2).expand(-1, -1, L, -1, -1)
        K_rand = torch.gather(K_sample, 3, index_sample.unsqueeze(-1).expand(-1, -1, -1, -1, D))

        Q_mean = Q.mean(dim=-1, keepdim=True)
        Q0 = Q - Q_mean
        K_rand_mean = K_rand.mean(dim=-1, keepdim=True)
        K0 = K_rand - K_rand_mean

        cov = (Q0.unsqueeze(3) * K0).sum(dim=-1)
        normQ = torch.norm(Q0, dim=-1, keepdim=True)
        normK = torch.norm(K0, dim=-1)
        denom = normQ * normK + 1e-8
        scores_top = torch.abs(cov / denom)
        M_top = scores_top.mean(dim=-1).topk(U, dim=-1)[1] 

        
        Q_reduce = Q.gather(2, M_top.unsqueeze(-1).expand(-1, -1, -1, D)) 

       
        Q_norm = F.normalize(Q_reduce, dim=-1) 
        K_normed = F.normalize(K, dim=-1)  

        scores = torch.einsum("bhmd,bhsd->bhms", Q_norm, K_normed) 

        tau = 1/0.24
        log_scale = tau* math.log(S)
        scores = scores * log_scale  

        
        if self.mask_flag and attn_mask is not None:
            mask = attn_mask.bool()  
            M_top_unsq = M_top.unsqueeze(-1).expand(-1, -1, -1, S)  
            selected_mask = torch.gather(mask, 2, M_top_unsq)  
            scores = scores.masked_fill(selected_mask, float("-inf"))

       
        A = self.dropout(torch.softmax(scores, dim=-1))  

       
        V_selected = torch.einsum("bhms,bhsd->bhmd", A, V)  
        out = torch.zeros(B, H, L, D, device=queries.device)
        out.scatter_(2, M_top.unsqueeze(-1).expand(-1, -1, -1, D), V_selected)

        out = out.permute(0, 2, 1, 3).contiguous() 

        return (out, A) if self.output_attention else (out, None)






class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
