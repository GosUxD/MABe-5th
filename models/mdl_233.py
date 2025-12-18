import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
from torch import Tensor
import math
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    """Gated Linear Unit activation"""
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm residual units
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module for handling variable sequence lengths
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.0,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9) if score.dtype == torch.float32 else score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Self-attention module with relative positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """Pointwise 1D convolution (kernel size = 1)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module with pointwise conv -> GLU -> depthwise conv -> normalization -> activation
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.0,
        use_bn: bool = True,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.inorm = nn.InstanceNorm1d(in_channels, affine=True)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
        self.use_bn = use_bn

    def forward(self, x, mask_pad):
        # Transpose for conv operations [B, T, C]
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        if self.use_bn:
            # Apply batch norm only to non-padded positions
            x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
            mask_bn = mask_pad.view(-1)
            x_bn[mask_bn] = self.bn(x_bn[mask_bn])
            x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        else:    
            x = self.inorm(x)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        # Mask batch padding again
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    """Create learnable scale and bias parameters"""
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None, None, :])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: MHSA -> FFN -> Conv -> FFN with residual connections
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        self.mhsa = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
        )
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
            use_bn=use_bn,
        )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        mask_pad = (mask).long().bool().unsqueeze(1)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape

        # MHSA
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad)

        # Skip padding for layer norm
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)

        # FFN after MHSA
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.reshape(bs, slen, nfeats)

        # Conv
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad=mask.bool().unsqueeze(1))

        # Skip padding for layer norm
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_conv(x)

        # FFN after Conv
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)

        # Restore shape
        x_skip[mask_flat] = x[0].to(dtype=x_skip.dtype)
        x = x_skip.reshape(bs, slen, nfeats)

        return x

class SqueezeformerEncoder(nn.Module):
    """
    Stack of Squeezeformer blocks
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.0,
        feed_forward_dropout_p: float = 0.0,
        attention_dropout_p: float = 0.0,
        conv_dropout_p: float = 0.0,
        conv_kernel_size: int = 31,
        use_bn: bool = True,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    use_bn=use_bn,
                )
            )

    def forward(self, x: Tensor, mask: Tensor):
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
        return x

from timm.layers.norm_act import BatchNormAct2d

class FeatureExtractor(nn.Module):
    def __init__(self,
                 n_landmarks,
                 out_dim):
        super().__init__()   

        self.in_channels = in_channels = (32//2) * n_landmarks
        self.stem_linear = nn.Linear(in_channels,out_dim,bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        
    def forward(self, data, mask):
        xc = data.permute(0,2,1,3)  # B,C,T,F
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        xc = xc.permute(0,2,3,1)
        xc = xc.reshape(*data.shape[:2], -1)
        
        m = mask.to(torch.bool)  
        x = self.stem_linear(xc)
        
        # Batchnorm without pads
        bs,slen,nfeat = x.shape
        x = x.view(-1, nfeat)
        x_bn = x[mask.view(-1)==1].unsqueeze(0)
        x_bn = self.stem_bn(x_bn.permute(0,2,1)).permute(0,2,1)
        x[mask.view(-1)==1] = x_bn[0]
        x = x.view(bs,slen,nfeat)
        # Padding mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        
        return x

class MABeFeatureExtractor(nn.Module):
    def __init__(self, input_dim=708, encoder_dim=144, dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        # Project input features to encoder dimension
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.input_norm = nn.LayerNorm(encoder_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Optional: Add a small CNN for local temporal patterns
        self.use_conv = True
        if self.use_conv:
            self.conv1 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1)
            self.conv_norm = nn.LayerNorm(encoder_dim)
            self.conv_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: Input features (batch, seq_len, 708)
            mask: Attention mask (batch, seq_len)
        Returns:
            Encoded features (batch, seq_len, encoder_dim)
        """
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        if self.use_conv:
            x_conv = x.transpose(1, 2)  # (batch, encoder_dim, seq_len)
            x_conv = F.relu(self.conv1(x_conv))
            x_conv = F.relu(self.conv2(x_conv))
            x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, encoder_dim)

            x = x + self.conv_dropout(self.conv_norm(x_conv))

        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)

        return x

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm

class SpatialMouseGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_mice=4, heads=4, dropout=0.1):
        super().__init__()
        self.num_mice = num_mice
        self.embedding = nn.Linear(in_channels, out_channels)
        
        self.conv1 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm1 = LayerNorm(out_channels) 
        
        self.conv2 = TransformerConv(out_channels, out_channels // heads, heads=heads, 
                                     dropout=dropout, beta=True)
        self.norm2 = LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def _get_fully_connected_edge_index(self, batch_size_time, device):
        """
        Creates edges so every mouse connects to every other mouse within the same frame.
        """

        base_src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], device=device)
        base_dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], device=device)

        offsets = torch.arange(batch_size_time, device=device) * self.num_mice
        
        src = base_src.unsqueeze(1) + offsets.unsqueeze(0)
        dst = base_dst.unsqueeze(1) + offsets.unsqueeze(0)
        
        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
        return edge_index

    def forward(self, x):
        """
        Input: [Batch, Frames, Mice, Features]
        Output: [Batch, Frames, Mice, Out_Channels]
        """
        B, T, M, D = x.shape
        
        x_flat = x.view(B * T * M, D)
        
        x_emb = F.relu(self.embedding(x_flat))
        
        edge_index = self._get_fully_connected_edge_index(B * T, x.device)
        
        h = self.conv1(x_emb, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 4. GNN Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)

        return h.view(B, T, M, -1)


class BehaviorClassificationHead(nn.Module):

    def __init__(self, encoder_dim=144, num_pairs=16, num_actions=39, dropout=0.0):
        super().__init__()

        self.num_pairs = num_pairs  # Number of mouse pairs (e.g., 4 mice -> 16 directed pairs)
        self.num_actions = num_actions  # Number of behavior classes

        self.shared_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(encoder_dim, num_pairs * num_actions)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.shared_proj(x)  # (batch, seq_len, encoder_dim)
        logits = self.classifier(x)  # (batch, seq_len, num_pairs * num_actions)
        logits = logits.view(batch_size, seq_len, self.num_pairs, self.num_actions)

        return logits

class PairwiseBehaviorHead(nn.Module):
    def __init__(self, encoder_dim, num_classes=39, dropout=0.0):
        super().__init__()
        
        self.in_features = encoder_dim * 2 
        
        self.layers = nn.Sequential(
            nn.Linear(self.in_features, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Output is logits for ONE pair
            nn.Linear(encoder_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Mice, Dim] - The unflattened Squeezeformer output
        Returns:
            logits: [Batch, Time, Num_Pairs (16), Num_Classes]
        """
        B, T, M, D = x.shape
        
        m1 = x.unsqueeze(3).expand(-1, -1, -1, M, -1)
        m2 = x.unsqueeze(2).expand(-1, -1, M, -1, -1)
        
        # Shape: [B, T, M, M, 2*D]
        pair_features = torch.cat([m1, m2], dim=-1)

        # Shape: [B, T, M*M, 2*D] -> [B, T, 16, 2*D]
        pair_features_flat = pair_features.view(B, T, M*M, -1)
        
        # 5. Classify
        logits = self.layers(pair_features_flat)
        
        return logits

class Net(nn.Module):
    """
    Squeezeformer model for MABe mouse behavior detection
    """
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        # Model dimensions
        self.encoder_dim = cfg.encoder_config.encoder_dim
        self.num_pairs = 16  # Number of mouse pairs
        self.num_actions = 39  # Number of behavior classes (including no_action)

        self.gnn = SpatialMouseGNN(
            in_channels=cfg.per_mouse_feature_dim,
            out_channels=cfg.encoder_config.encoder_dim,
            num_mice=4,
            heads=4,
            dropout=cfg.encoder_config.input_dropout_p
        )

        # Feature extractor
        if self.cfg.cnn_extractor:
            self.feature_extractor = FeatureExtractor(
                n_landmarks=cfg.per_mouse_feature_dim // 4,
                out_dim=cfg.encoder_config.encoder_dim)
        else:
            self.feature_extractor = MABeFeatureExtractor(
                input_dim=cfg.feature_dim,
                encoder_dim=self.encoder_dim,
                dropout=cfg.encoder_config.input_dropout_p
            )

        self.encoder = SqueezeformerEncoder(
            input_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            use_bn = cfg.use_bn
        )

        self.classifier = BehaviorClassificationHead(
            encoder_dim=self.encoder_dim,
            num_pairs=self.num_pairs,
            num_actions=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        self.classifier = PairwiseBehaviorHead(
            encoder_dim=self.encoder_dim,
            num_classes=self.num_actions,
            dropout=cfg.encoder_config.feed_forward_dropout_p
        )

        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights)
        else:
            pos_weight = cfg.pos_weight if hasattr(cfg, 'pos_weight') else 50.0 
            class_weights = torch.full((self.num_actions,), pos_weight) # High weight for all
            no_action_idx = cfg.action_id_map.get('no_action', -1)
            if no_action_idx != -1:
                class_weights[no_action_idx] = 1.0 # Low weight for no_action
            else:
                class_weights[-1] = 1.0

        self.register_buffer('class_weights', class_weights)

        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        self.unweighted_ce_fn = nn.CrossEntropyLoss(reduction='none')

        self.use_focal_loss = cfg.use_focal_loss if hasattr(cfg, 'use_focal_loss') else False
        self.focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0

        # Training settings
        self.return_logits = cfg.return_logits if hasattr(cfg, 'return_logits') else False

        print(f'Model initialized with {count_parameters(self):,} trainable parameters')
        # print(f'Loss: {"Multi-class Focal" if self.use_focal_loss else "Weighted CrossEntropy"}')
        # print(f'Class weights: {self.class_weights.cpu().numpy()}')

    def forward(self, batch):
        mask = batch['input_mask'].long()  # (batch, seq_len)
        mask_for_encoder = mask.repeat_interleave(4, dim=0)

        if self.cfg.cnn_extractor:
            x = batch['input_mice']
            x = self.gnn(x)  # (batch, seq_len, num_mice, encoder_dim)
            B, T, M, C = x.shape
            x = x.permute(0,2,1,3).reshape(B * M, T, C)  # (batch*num_mice, seq_len, encoder_dim)
        else:
            x = batch['input']  # (batch, seq_len, 708)
            B, T, _ = x.shape
            x = self.feature_extractor(x, mask)  # (batch, seq_len, encoder_dim)


        # Encode
        x = self.encoder(x, mask_for_encoder)  # (batch, seq_len, encoder_dim)
        x = x.reshape(B, M, T, C).permute(0,2,1,3)  # (batch, seq_len, num_mice, encoder_dim)

        # Classify
        logits = self.classifier(x)  # (batch, seq_len, num_pairs, num_actions)

        if 'behavior_mask' in batch and batch['behavior_mask'] is not None:
            behavior_mask = batch['behavior_mask']  # (batch, num_pairs, num_actions)
            behavior_mask_expanded = behavior_mask.unsqueeze(1).expand_as(logits)
            logits = torch.where(behavior_mask_expanded.bool(), logits,
                                torch.tensor(-1e10, dtype=logits.dtype, device=logits.device) if logits.dtype == torch.float32
                                else torch.tensor(-1e4, dtype=logits.dtype, device=logits.device))

        output = {}

        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels']

            if labels.dim() == 4:  # One-hot encoded (batch, seq_len, num_pairs, num_actions)
                labels = torch.argmax(labels, dim=-1)  # (batch, seq_len, num_pairs)

            # Reshape for loss calculation
            batch_size, seq_len, num_pairs, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)  # (batch*seq*pairs, num_actions)
            labels_flat = labels.reshape(-1)  # (batch*seq*pairs,)

            if self.use_focal_loss:
                unweighted_ce = self.unweighted_ce_fn(logits_flat, labels_flat)
                pt = torch.exp(-unweighted_ce)
                alpha = self.class_weights[labels_flat]  # Gather alpha for true classes
                focal_loss = alpha * (1 - pt) ** self.focal_gamma * unweighted_ce
                loss = focal_loss
            else:
                loss = self.loss_fn(logits_flat, labels_flat)

            loss = loss.view(batch_size, seq_len, num_pairs)

            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_expanded

            # Average over valid positions
            valid_positions = mask_expanded.sum()
            if valid_positions > 0:
                loss = loss.sum() / valid_positions
            else:
                loss = loss.sum()
            output['loss'] = loss

        probs = torch.softmax(logits, dim=-1)
        output['predictions'] = probs
        output['logits'] = logits

        if 'video_id' in batch:
            output['video_id'] = batch['video_id']
        if 'start_frame' in batch:
            output['start_frame'] = batch['start_frame']

        return output