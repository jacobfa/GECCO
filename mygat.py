from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter, Linear, Dropout, LayerNorm, ReLU


        
class Att(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
            nn.init.xavier_uniform_(self.lin_l.weight)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias)
                nn.init.xavier_uniform_(self.lin_r.weight)
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias)
            nn.init.xavier_uniform_(self.lin_l.weight)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias)
                nn.init.xavier_uniform_(self.lin_r.weight)

        self.att = Parameter(torch.empty(1, heads, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

    
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
        
    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        return_attention_weights: bool = None,
    ):
        H, C = self.heads, self.out_channels

        x_l, x_r = None, None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        x = x_l + x_r
        alpha = torch.sum( x * self.att, dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        out = x_r * alpha.unsqueeze(-1)
        
        if self.concat:
            out = torch.cat([out[:, i] for i in range(H)], dim=-1)
        else:
            out = torch.mean(out, dim=1)
            
        if self.bias is not None:
            out += self.bias
            
        return out
    