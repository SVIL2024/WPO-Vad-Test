"""
Mamba building blocks for single bidirectional Mamba VAD.
Contains only components used in the paper: MambaResBlock, MambaTemporalEncoder,
BidirectionalMambaBlock, BiMambaEncoder.
"""
from collections import OrderedDict
import torch
from torch import nn

from wpo_modules import TemporalWPO

HAS_MAMBA = False
Mamba = None
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    try:
        from mamba_ssm import Mamba2 as Mamba
        HAS_MAMBA = True
    except ImportError as e:
        print(f"[Warning] mamba_ssm not available: {e}")
        print("Please: pip install mamba-ssm causal-conv1d --no-build-isolation")


def create_mamba(mamba_version, d_model, d_state, d_conv, expand, **kwargs):
    # Direction A: swap the real-diagonal SSM for a complex-diagonal (2nd-order,
    # damped-oscillator) one. `wave_omega_init=0` freezes the imaginary part at
    # zero, which degenerates the layer to a real-diagonal SSM through the SAME
    # code path with the SAME parameter count -- the exact ablation.
    use_wave = kwargs.pop('wave_ssm', False)
    wave_omega_init = kwargs.pop('wave_omega_init', 'spread')
    wave_ckpt = kwargs.pop('wave_checkpoint_scan', False)
    wave_sc = kwargs.pop('wave_state_chunk', 1)
    if use_wave:
        from wave_ssm import WaveSSM
        return WaveSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                       omega_init=wave_omega_init, checkpoint_scan=wave_ckpt,
                       state_chunk=wave_sc)
    if mamba_version == 'mamba1':
        if Mamba is None:
            raise ImportError("mamba_ssm.Mamba not available.")
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    elif mamba_version == 'mamba2':
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError("mamba_ssm.Mamba2 not available.")
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                      headdim=kwargs.get('headdim', 64), ngroups=kwargs.get('ngroups', 1))
    else:
        raise ValueError(f"Unknown mamba_version: {mamba_version}")


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MambaResBlock(nn.Module):
    """Mamba block with an optional identity-anchored TemporalWPO.

    Serial  (wpo_parallel=False): Pre-LN → WPO → Mamba → residual → Pre-LN → MLP → residual.
    Parallel (wpo_parallel=True): the WPO deviation bypasses the SSM and is added
    to the residual stream directly — x + Mamba(ln(x)) + (WPO(ln(x)) − ln(x)) —
    so the SSM cannot absorb/undo the frequency shaping (the serial composition
    showed only +0.36 AP while the band views reached +2.75; the evidence points
    at the SSM washing the filter out).
    With the WPO gate at 0 both variants are bit-exact the baseline block.
    """
    def __init__(self, d_model: int, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 mamba_version='mamba1', mlp_scale=4, wpo=None, wpo_parallel=False,
                 **mamba_kwargs):
        super().__init__()
        self.mamba = create_mamba(mamba_version, d_model, d_state, d_conv, expand, **mamba_kwargs)
        self.wpo = wpo
        self.wpo_parallel = wpo_parallel
        self.ln_1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * mlp_scale)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * mlp_scale, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        h = self.ln_1(x)
        mamba_out = self.dropout1(self.mamba(h))
        if self.wpo is None:
            x = x + mamba_out
        elif self.wpo_parallel:
            x = x + mamba_out + (self.wpo(h, padding_mask) - h)
        else:
            x = x + self.dropout1(self.mamba(self.wpo(h, padding_mask)))
        x = x + self.dropout2(self.mlp(self.ln_2(x)))
        return x


class MambaTemporalEncoder(nn.Module):
    """Causal Mamba encoder. [T, B, D] → [T, B, D].

    `wpo_factory` (optional callable) overrides the default `TemporalWPO`
    construction, so any module with a `forward(x, padding_mask)` signature can
    be dropped into the block's WPO slot (e.g. `WaveResonatorBank`).
    """
    def __init__(self, d_model=512, n_layer=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, mamba_version='mamba1', mlp_scale=4,
                 use_wpo=False, wpo_kwargs=None, wpo_parallel=False,
                 wpo_factory=None, **mamba_kwargs):
        super().__init__()
        wpo_kwargs = dict(wpo_kwargs or {})
        self.layers = nn.ModuleList([
            MambaResBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                          dropout=dropout, mamba_version=mamba_version, mlp_scale=mlp_scale,
                          wpo=(wpo_factory() if wpo_factory is not None
                               else (TemporalWPO(d_model, **wpo_kwargs) if use_wpo else None)),
                          wpo_parallel=wpo_parallel,
                          **mamba_kwargs)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        # x: [T, B, D] → Mamba expects [B, T, D]
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.norm(x)
        return x.permute(1, 0, 2)  # [T, B, D]


class BidirectionalMambaBlock(nn.Module):
    """Mamba block with bidirectional processing.
    Forward scan + backward scan → concat → Linear(2D→D) → residual → MLP → residual.

    The optional identity-anchored TemporalWPO runs on the pre-norm stream; with
    wpo_parallel=True the filter deviation bypasses both scans and is added to
    the residual stream directly (gate 0 = exact baseline behaviour either way).
    """
    def __init__(self, d_model: int, d_state=16, d_conv=4, expand=2, dropout=0.0,
                 mamba_version='mamba1', mlp_scale=4, wpo=None, wpo_parallel=False,
                 **mamba_kwargs):
        super().__init__()
        self.fwd_mamba = create_mamba(mamba_version, d_model, d_state, d_conv, expand, **mamba_kwargs)
        self.bwd_mamba = create_mamba(mamba_version, d_model, d_state, d_conv, expand, **mamba_kwargs)
        self.wpo = wpo
        self.wpo_parallel = wpo_parallel
        self.ln_1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * mlp_scale)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * mlp_scale, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        residual = x
        x_norm = self.ln_1(x)
        if self.wpo is not None and not self.wpo_parallel:
            x_norm = self.wpo(x_norm, padding_mask)
        fwd = self.fwd_mamba(x_norm)
        bwd = self.bwd_mamba(torch.flip(x_norm, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])
        combined = torch.cat([fwd, bwd], dim=-1)
        combined = self.proj(combined)
        x = residual + self.dropout1(combined)
        if self.wpo is not None and self.wpo_parallel:
            x = x + (self.wpo(x_norm, padding_mask) - x_norm)
        x = x + self.dropout2(self.mlp(self.ln_2(x)))
        return x


class BiMambaEncoder(nn.Module):
    """Bidirectional Mamba encoder. [T, B, D] → [T, B, D]."""
    def __init__(self, d_model=512, n_layer=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, mamba_version='mamba1', mlp_scale=4,
                 use_wpo=False, wpo_kwargs=None, wpo_parallel=False,
                 wpo_factory=None, **mamba_kwargs):
        super().__init__()
        wpo_kwargs = dict(wpo_kwargs or {})
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                                    expand=expand, dropout=dropout,
                                    mamba_version=mamba_version, mlp_scale=mlp_scale,
                                    wpo=(wpo_factory() if wpo_factory is not None
                                         else (TemporalWPO(d_model, **wpo_kwargs)
                                               if use_wpo else None)),
                                    wpo_parallel=wpo_parallel,
                                    **mamba_kwargs)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        # x: [T, B, D]
        x = x.permute(1, 0, 2)  # [B, T, D]
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.norm(x)
        return x.permute(1, 0, 2)  # [T, B, D]
