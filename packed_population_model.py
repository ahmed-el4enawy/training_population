"""Packed, functionally equivalent population model for GPU-efficient training.

This module preserves the effective 10-subnetwork architecture from
CGMOHSUSimStateSpaceModel_V2, but packs the first and second linear layers into
two dense masked matrices. Inactive connections are initialized to zero and
their gradients are masked to zero, so the effective trainable connections,
state dependencies, hidden widths, ReLU nonlinearity, and outputs are unchanged.

The final export converts back to the official CGMOHSUSimStateSpaceModel_V2
state_dict layout for evaluation/inference compatibility.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PackedPopulationModel(nn.Module):
    STATE_INPUT_DIM = 12  # 10 states + insulin + carbs

    # output order must stay [Q1,Q2,S1,S2,I,X1,X2,X3,C2,C1]
    SPECS = (
        ("Q1", "net_dQ1", (5, 7, 0, 1, 8)),
        ("Q2", "net_dQ2", (5, 6, 0, 1)),
        ("S1", "net_dS1", (2, 10)),
        ("S2", "net_dS2", (2, 3)),
        ("I",  "net_dI",  (3, 4)),
        ("X1", "net_dX1", (4, 5)),
        ("X2", "net_dX2", (4, 6)),
        ("X3", "net_dX3", (4, 7)),
        ("C2", "net_dC2", (9, 8)),
        ("C1", "net_dC1", (9, 11)),
    )

    def __init__(self, source_model, n_feat):
        super().__init__()
        self.n_feat = dict(n_feat)

        hidden_sizes = [int(self.n_feat[key]) for key, _, _ in self.SPECS]
        offsets = [0]
        for h in hidden_sizes:
            offsets.append(offsets[-1] + h)
        self.hidden_slices = tuple((offsets[i], offsets[i + 1]) for i in range(len(hidden_sizes)))
        total_hidden = offsets[-1]

        self.weight1 = nn.Parameter(torch.zeros(total_hidden, self.STATE_INPUT_DIM, dtype=torch.float32))
        self.bias1 = nn.Parameter(torch.zeros(total_hidden, dtype=torch.float32))
        self.weight2 = nn.Parameter(torch.zeros(len(self.SPECS), total_hidden, dtype=torch.float32))
        self.bias2 = nn.Parameter(torch.zeros(len(self.SPECS), dtype=torch.float32))

        mask1 = torch.zeros_like(self.weight1)
        mask2 = torch.zeros_like(self.weight2)

        with torch.no_grad():
            for out_i, ((key, attr, input_indices), (h0, h1)) in enumerate(zip(self.SPECS, self.hidden_slices)):
                src = getattr(source_model, attr)
                w0 = src[0].weight.detach()
                b0 = src[0].bias.detach()
                w1 = src[2].weight.detach()
                b1 = src[2].bias.detach()

                if w0.shape != (h1 - h0, len(input_indices)):
                    raise ValueError(f"Unexpected first-layer shape for {key}: {tuple(w0.shape)}")

                for local_col, global_col in enumerate(input_indices):
                    self.weight1[h0:h1, global_col].copy_(w0[:, local_col])
                    mask1[h0:h1, global_col] = 1.0

                self.bias1[h0:h1].copy_(b0)
                self.weight2[out_i, h0:h1].copy_(w1[0])
                self.bias2[out_i].copy_(b1[0])
                mask2[out_i, h0:h1] = 1.0

        self.register_buffer("mask1", mask1, persistent=True)
        self.register_buffer("mask2", mask2, persistent=True)

        # Keep inactive packed connections exactly zero during optimization.
        self.weight1.register_hook(lambda grad: grad * self.mask1)
        self.weight2.register_hook(lambda grad: grad * self.mask2)

    def forward(self, in_x, in_u):
        z = torch.cat((in_x, in_u), dim=-1)
        hidden = F.relu(F.linear(z, self.weight1, self.bias1))
        return F.linear(hidden, self.weight2, self.bias2)

    def active_parameter_count(self):
        return int(
            self.mask1.sum().item()
            + self.bias1.numel()
            + self.mask2.sum().item()
            + self.bias2.numel()
        )

    def assert_inactive_zero(self):
        with torch.no_grad():
            if torch.count_nonzero(self.weight1 * (1.0 - self.mask1)).item() != 0:
                raise RuntimeError("Inactive packed first-layer weights became non-zero")
            if torch.count_nonzero(self.weight2 * (1.0 - self.mask2)).item() != 0:
                raise RuntimeError("Inactive packed second-layer weights became non-zero")

    def export_official_state_dict(self):
        """Return the exact official module-key layout expected by CGMOHSUSimStateSpaceModel_V2."""
        out = OrderedDict()
        for out_i, ((key, attr, input_indices), (h0, h1)) in enumerate(zip(self.SPECS, self.hidden_slices)):
            cols = [self.weight1[h0:h1, idx] for idx in input_indices]
            first_weight = torch.stack(cols, dim=1)

            out[f"{attr}.0.weight"] = first_weight.detach().cpu().clone()
            out[f"{attr}.0.bias"] = self.bias1[h0:h1].detach().cpu().clone()
            out[f"{attr}.2.weight"] = self.weight2[out_i, h0:h1].unsqueeze(0).detach().cpu().clone()
            out[f"{attr}.2.bias"] = self.bias2[out_i].reshape(1).detach().cpu().clone()

        return out
