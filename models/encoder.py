import torch
import torch.nn as nn
import torch.nn.functional as F



class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvLayer(nn.Module):
    def __init__(self, c_in, kernel_size=3, causal=False):
        super(ConvLayer, self).__init__()
        self.norm = nn.LayerNorm(c_in)
        self.causal = causal
        self.pad = (kernel_size - 1) if causal else (kernel_size - 1) // 2

        self.proj = nn.Linear(c_in, c_in * 2)
        self.dwconv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=kernel_size,
            padding=0,
            groups=c_in
        )

        self.out_proj = nn.Linear(c_in, c_in)
        self.res_proj = nn.Linear(c_in, c_in)

    def forward(self, x):
        x_norm = self.norm(x)
        x_proj = self.proj(x_norm)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x_glu = x1 * torch.sigmoid(x2)

        x_glu = x_glu.permute(0, 2, 1)
        if self.causal:
            x_glu = F.pad(x_glu, (self.pad, 0))
        else:
            x_glu = F.pad(x_glu, (self.pad // 2, self.pad - self.pad // 2))
        x_conv = self.dwconv(x_glu)
        x_conv = F.gelu(x_conv).permute(0, 2, 1)

        min_len = min(x_conv.size(1), x_norm.size(1))
        x_conv = x_conv[:, :min_len, :]
        x_norm = x_norm[:, :min_len, :]

        out = self.out_proj(x_conv) + self.res_proj(x_norm)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu", drop_path=0.0):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1)
        )

        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, attn_mask=None):
        attn_out, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.drop_path(self.dropout(attn_out))
        x = self.norm1(x)

        y = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop_path(self.dropout(y))
        x = self.norm2(x)

        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers[:-1], self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :], attn_mask=attn_mask)
            x_stack.append(x_s)
            attns.append(attn)

        x_stack = torch.cat(x_stack, dim=1)
        return x_stack, attns



