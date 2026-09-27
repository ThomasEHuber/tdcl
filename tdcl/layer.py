from typing import Callable, Union

import snntorch as snn
import torch
from snntorch import utils
from torch import nn
from util.tracer import Tracer

from tdcl import FullOnlineTDCLLayer


class Block(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float,
        spike_grad: Callable,
        init_hidden: bool,
    ) -> None:
        super().__init__()

        self.ff = nn.Linear(in_features, out_features, bias=False)
        nn.init.kaiming_uniform_(self.ff.weight, mode="fan_in", nonlinearity="tanh")
        # self.ff.weight = torch.nn.Parameter(self.ff.weight*3)
        self.leaky = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            init_hidden=init_hidden,
            output=True,
            learn_beta=False,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        input:
            x: presynaptic spikes: (T, B, C, H, W)
        output:
            spk_rec: (T, B, C', H', W')
            mem_rec: (T, B, C', H', W') | None
        """
        T = x.shape[0]
        B = x.shape[1]
        x = x.reshape(T, B, -1)
        spk_rec = []
        mem_rec = []
        utils.reset(self)  # resets hidden states for all LIF neurons in net
        for t in range(T):  # x.size(0) = number of time steps
            spk_out, mem_out = self.leaky(self.ff(x[t]))
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
        return spk_rec, mem_rec

    def step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input:
            x: presynaptic spikes: (B, N) or (B, C, H, W)
        output:
            spk_rec: (B, N')
            mem_rec: (B, N')
        """
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.ff(x)
        spk_out, mem_out = self.leaky(x)
        return spk_out, mem_out

    def get_in_dim(self) -> int:
        return self.ff.in_features

    def get_out_dim(self) -> int:
        return self.ff.out_features

    def get_weight(self) -> torch.Tensor:
        return self.ff.weight

    def set_weight(self, weight: torch.Tensor) -> None:
        if isinstance(weight, torch.Tensor):
            weight = nn.Parameter(weight)
        self.ff.weight = weight


def create_tdcl_block(
    in_features: int,
    out_features: int,
    beta: float,
    spike_grad: Callable,
    init_hidden: bool,
    lr: float,
    tracer: Tracer,
    warm_up: int,
    input_thr: float = 0.02,
    gamma: float = 0.1,
    temp: float = 0.5,
    spike_as_grad: bool = False,
) -> FullOnlineTDCLLayer:
    block = Block(
        in_features=in_features,
        out_features=out_features,
        beta=beta,
        spike_grad=spike_grad,
        init_hidden=init_hidden,
    )
    return FullOnlineTDCLLayer(
        layer=block,
        tracer=tracer,
        lr=lr,
        warm_up=warm_up,
        input_thr=input_thr,
        gamma=gamma,
        temp=temp,
        spike_as_grad=spike_as_grad,
    )
