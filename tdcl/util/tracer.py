import math
from typing import *

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class LI(nn.Module):
    def __init__(self, beta: float, carry_state: bool) -> None:
        super().__init__()
        self.beta = beta
        self.carry_state = carry_state
        if self.carry_state:
            self.state = torch.tensor((0,))
        else:
            self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (T, ...)
            beta: float
        output:
            li: (T, ...) same shape as x
        """
        T = x.shape[0]
        if self.carry_state:
            buff = self.state
        else:
            buff = 0

        li = []
        for t in range(T):
            buff = self.beta * buff + x[t]
            li.append(buff)
        li = torch.stack(li)

        if self.carry_state:
            self.state = buff

        return li

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (...)
            beta: float
        output:
            li: (...) same shape as x
        """
        assert self.carry_state
        self.state = self.beta * self.state + x
        return self.state

    def to(self, device: str) -> "LI":
        # self = super().to(device)
        if self.state is not None:
            self.state = self.state.to(device)
        return self

    def reset(self) -> None:
        if self.carry_state:
            device = self.state.device
            self.state = torch.tensor((0,)).to(device)


class Tracer:
    def __init__(self, carry_state: bool):
        self.carry_state = carry_state
        self.traces: list[tuple[torch.Tensor, LI]] = []

    def add(self, gain: float, beta: float) -> None:
        li = LI(beta, self.carry_state)
        self.traces.append([torch.tensor((gain,)), li])
        self.traces = sorted(self.traces, key=lambda x: x[1].beta, reverse=True)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        for gain, li in self.traces:
            out += gain * li(x)
        return out

    def step(self, x: torch.Tensor) -> None:
        out = 0
        for gain, li in self.traces:
            out += gain * li.step(x)

    def get(self) -> torch.Tensor:
        out = 0
        for gain, li in self.traces:
            out += gain * li.state
        return out

    def __str__(self) -> str:
        out = ""
        for gain, li in self.traces:
            out += f"({gain.item():.4f}_{li.beta:.2f})_"
        if self.carry_state:
            out += "carry"
        else:
            out = out[:-1]
        return out

    def to(self, device) -> "Tracer":
        for i in range(len(self.traces)):
            self.traces[i][0] = self.traces[i][0].to(device)
            self.traces[i][1] = self.traces[i][1].to(device)
        return self

    def reset(self) -> None:
        for gain, li in self.traces:
            li.reset()

    def clone(self) -> "Tracer":
        new_tracer = Tracer(self.carry_state)
        for gain, li in self.traces:
            new_tracer.traces.append([gain.clone(), LI(li.beta, self.carry_state)])
        return new_tracer

    def gen_plot(self, fig_scale: float = 1.0) -> tuple[plt.Figure, plt.Axes]:
        assert len(self.traces) == 2

        fig, ax = plt.subplots(figsize=(8 * fig_scale, 5 * fig_scale))
        steps = 10
        x = np.linspace(0, 100, steps * 100)[::-1]

        y = 0
        label = ""
        for i, (gain, li) in enumerate(self.traces):
            y += gain * li.beta**x
            if i > 0 and gain > 0:
                label += "+"
            label += f"{gain.item():.4f} \cdot {li.beta:.2f}^x"
        label = "$" + label + "$"

        ax.plot(x[::-1], y, c="C0", label=label)
        ax.set_xticks(np.arange(110, step=10))

        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i in range(len(labels)):
            idx = (i - len(labels) + 1) * 10
            if idx == 0:
                labels[i] = "$t_0$"
            else:
                labels[i] = "$t_0" + str(idx) + "$"
        ax.set_xticklabels(labels)

        ax.set_ylabel("Weighting")
        ax.set_xlabel("Timesteps")

        ax.plot(np.zeros_like(x), color="black")
        ax.set_xlim(0, 100)
        ax.grid()
        ax.legend(loc=2, prop={"size": 10})
        return fig, ax

    def get_info(self) -> dict[str, float | int]:
        assert len(self.traces) == 2

        area = -self.traces[0][0] / math.log(self.traces[0][1].beta) - self.traces[1][
            0
        ] / math.log(self.traces[1][1].beta)
        disc_area = self.traces[0][0] / (1 - self.traces[0][1].beta) + self.traces[1][
            0
        ] / (1 - self.traces[1][1].beta)

        y_intersect = self.traces[0][0] + self.traces[1][0]
        x_intersect = -(
            torch.log(self.traces[1][0]) - torch.log(-self.traces[0][0])
        ) / (math.log(self.traces[1][1].beta) - math.log(self.traces[0][1].beta))

        return {
            "num traces": len(self.traces),
            "continuous area": area.item(),
            "discrete area": disc_area.item(),
            "y intersection": y_intersect.item(),
            "x intersection": x_intersect.item(),
        }
