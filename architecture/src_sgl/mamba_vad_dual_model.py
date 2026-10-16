"""
Single bidirectional Mamba VAD model (paper version).

Architecture (single Mamba, no GCN, no dual):
  CLIP features → Position Embed → BiMambaEncoder → Multi-scale fusion → Classifier

Only bidirectional configurable.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mamba_model import MambaTemporalEncoder, BiMambaEncoder
from wpo_modules import FixedBandFilter, TemporalWPO
from wave_resonator import WaveResonatorBank
from wave_tv import TimeVaryingWaveBank


def _wave_pq(periods, qs):
    """Parse the (P, Q) CLI strings for the time-varying wave bank.

    Empty / 'auto' -> let ``TimeVaryingWaveBank`` derive them from the kernel
    size: log-spaced P in [p_min, size/2] and Q = pi*size/(4P). That is the
    physically valid tiling (every kernel sees >= 2 periods and its envelope
    decays inside the window); a hand-written list is only for ablations.
    """
    sp = str(periods).strip().lower()
    sq = str(qs).strip().lower()
    _p = None if sp in ('', 'none', 'auto') else \
        tuple(float(v) for v in str(periods).split(',') if v.strip())
    _q = None if sq in ('', 'none', 'auto') else \
        tuple(float(v) for v in str(qs).split(',') if v.strip())
    if _q is not None and _p is not None and len(_q) == 1 and len(_p) > 1:
        _q = _q * len(_p)
    return _p, _q


class AdaptiveFusion(nn.Module):
    """Content-adaptive per-frame scale fusion (identity-anchored).

    The static concat+linear fusion is decomposed into per-scale blocks:
    fused = sum_k W_k s_k, where [W_1..W_K] are column slices of the original
    fusion weight. The adaptive version multiplies each block by a per-frame
    modulation m_k(t) = 1 + tanh(MLP(concat(scales))). The MLP is zero-initialised,
    so m_k == 1 at start and the module is bit-exact equal to the static fusion;
    it then learns content-dependent band emphasis (e.g. anomalous, fast-motion
    frames can up-weight the high-frequency band).
    """

    def __init__(self, dim, num_scales):
        super().__init__()
        self.num_scales = num_scales
        self.proj = nn.Linear(dim * num_scales, dim)
        self.mod_mlp = nn.Sequential(
            nn.Linear(dim * num_scales, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_scales),
        )
        nn.init.zeros_(self.mod_mlp[-1].weight)
        nn.init.zeros_(self.mod_mlp[-1].bias)

    def forward(self, scales):
        cat = torch.cat(scales, dim=-1)                     # (B, T, D*K)
        D = cat.shape[-1] // self.num_scales
        static_bias = self.proj.bias
        out = static_bias
        m = 1.0 + torch.tanh(self.mod_mlp(cat))             # (B, T, K), init 1
        for k, s in enumerate(scales):
            W_k = self.proj.weight[:, k * D:(k + 1) * D]    # (D, D) per-scale block
            out = out + torch.nn.functional.linear(s, W_k) * m[:, :, k:k + 1]
        return out


class SingleMambaVAD(nn.Module):
    """Single bidirectional Mamba VAD.

    Training objective:
        L = L_cls + alpha * L_proto + beta * L_future
    """

    def __init__(self, visual_width=512, visual_length=256,
                 mamba_layers=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.0,
                 num_class=14,
                 device='cuda', mamba_version='mamba1',
                 multi_scale=False, multi_scale_factors='1,8,16', ms_pool_mode='pool',
                 bidirectional=False, mlp_scale=4,
                 mamba2_headdim=64, mamba2_ngroups=1,
                 use_wpo=False, wpo_alpha_init=0.3, wpo_c_init=0.8, wpo_c_max=20.0,
                 wpo_multiscale=False, wpo_band_c='0.5,3.0',
                 wpo_band_mode='learnable',
                 adaptive_fusion=False, wpo_parallel=False,
                 wpo_adaptive_params=False, wpo_motion_cond=False,
                 use_class_proto=False,
                 use_wrb=False, wrb_kind='resonator', wrb_periods='64,16',
                 wrb_q='4,4', wrb_groups=1, wrb_adaptive=False,
                 wrb_motion_cond=False, wrb_where='fusion', wrb_gain_init=1.0,
                 wave_tv=False, wave_tv_where='pre',
                 wave_tv_periods='', wave_tv_q='',
                 wave_tv_size=33, wave_tv_gain=0.1,
                 wave_tv_k=5, wave_tv_pmin=4.0,
                 wave_ssm=False, wave_omega_init='spread',
                 wave_checkpoint_scan=False, wave_state_chunk=1):
        super().__init__()
        # Direction A: with --wave-ssm the mixer becomes a complex-diagonal
        # (2nd-order / damped-oscillator) SSM instead of Mamba. omega_init=0
        # freezes the imaginary part, degenerating to a real-diagonal SSM with
        # identical parameter count -- the strict ablation.
        self.wave_ssm = wave_ssm

        self.visual_length = visual_length
        self.visual_width = visual_width
        self.device = device
        self.num_class = num_class
        self.multi_scale = multi_scale
        self.use_wpo = use_wpo

        # Mamba kwargs
        mamba_kwargs = {}
        if wave_ssm:
            # overrides mamba_version entirely -- WaveSSM is pure PyTorch
            mamba_kwargs = {'wave_ssm': True, 'wave_omega_init': wave_omega_init,
                            'wave_checkpoint_scan': wave_checkpoint_scan,
                            'wave_state_chunk': wave_state_chunk}
        elif mamba_version == 'mamba2':
            mamba_kwargs = {'headdim': mamba2_headdim, 'ngroups': mamba2_ngroups}
            # causal_conv1d kernel: (d_model*expand + 2*d_state*ngroups) % 8 == 0
            orig = d_state
            while (visual_width * expand + 2 * d_state * mamba2_ngroups) % 8 != 0:
                d_state += 1
            if d_state != orig:
                print(f"[AutoFix] d_state bumped from {orig} → {d_state} "
                      f"to satisfy Mamba2 causal_conv1d %8 constraint")

        # ===== Position embedding =====
        max_pos = max(visual_length, 1024)
        self.frame_position_embeddings = nn.Embedding(max_pos, visual_width)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

        # ===== Wave resonator bank (driven damped-wave band-pass views) =====
        # Two insertion points, both cheap:
        #   fusion : extra scales appended to the multi-scale fusion. The existing
        #            pooled scales are avg_pool views = pure low-pass and highly
        #            redundant with each other; the resonator contributes genuinely
        #            band-pass evidence (|H(0)| = 1/sqrt(1+Q^2) < 1).
        #   block  : replaces TemporalWPO inside each Mamba block (same slot, needs
        #            --wpo-parallel semantics for the deviation to bypass the SSM).
        self.wrb_periods = tuple(float(v) for v in str(wrb_periods).split(','))
        self.wrb_q = tuple(float(v) for v in str(wrb_q).split(','))
        if len(self.wrb_q) == 1:
            self.wrb_q = self.wrb_q * len(self.wrb_periods)
        self.use_wrb = bool(use_wrb)
        self.wrb_where = wrb_where
        self.wrb_fusion = None
        wpo_factory = None

        # ===== Time-varying wave operator (see wave_tv.py) ==================
        # 'block' has to claim the in-block WPO slot BEFORE the encoder is built.
        self.wave_tv = None
        self.wave_tv_where = wave_tv_where
        if wave_tv and wave_tv_where == 'block':
            _p, _q = _wave_pq(wave_tv_periods, wave_tv_q)
            wpo_factory = (lambda: TimeVaryingWaveBank(
                visual_width, periods=_p, qs=_q, size=int(wave_tv_size),
                gate_hidden=max(16, visual_width // 8),
                gain_init=float(wave_tv_gain), residual=True,
                k_bands=int(wave_tv_k), p_min=float(wave_tv_pmin)))
            use_wpo = True

        if self.use_wrb and wrb_where in ('block', 'both'):
            p0, q0 = float(self.wrb_periods[0]), float(self.wrb_q[0])
            wpo_factory = (lambda: WaveResonatorBank(
                visual_width, num_bands=1, periods=(p0,), q=(q0,), kind=wrb_kind,
                n_groups=int(wrb_groups), adaptive=bool(wrb_adaptive),
                motion_cond=bool(wrb_adaptive and wrb_motion_cond),
                gain_init=0.1, residual=True))
            use_wpo = True

        # ===== Single Mamba encoder =====
        # Identity-anchored WPO (gate init 0) rides inside each encoder block when
        # enabled; wpo params get their own optimizer group in the train scripts.
        wpo_kwargs = None
        if use_wpo and wpo_factory is None:
            wpo_kwargs = dict(alpha_init=wpo_alpha_init, c_init=wpo_c_init,
                              c_max=wpo_c_max, internal_norm=False,
                              adaptive_params=wpo_adaptive_params,
                              motion_cond=(wpo_adaptive_params and wpo_motion_cond))
        if bidirectional:
            self.mamba = BiMambaEncoder(
                d_model=visual_width, n_layer=mamba_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mlp_scale=mlp_scale,
                mamba_version=mamba_version, use_wpo=use_wpo,
                wpo_kwargs=wpo_kwargs, wpo_parallel=wpo_parallel,
                wpo_factory=wpo_factory, **mamba_kwargs)
        else:
            self.mamba = MambaTemporalEncoder(
                d_model=visual_width, n_layer=mamba_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, attn_window=0, mlp_scale=mlp_scale,
                mamba_version=mamba_version, use_wpo=use_wpo,
                wpo_kwargs=wpo_kwargs, wpo_parallel=wpo_parallel,
                wpo_factory=wpo_factory, **mamba_kwargs)

        # ===== Future score prediction head =====
        self.future_score_head = nn.Linear(visual_width, 1)

        # ===== Multi-scale fusion =====
        # Plan B (wpo_multiscale): the scales are WPO band filters with per-band
        # learnable (alpha_k, c_k) instead of avg_pool views — no information is
        # destroyed by downsampling; each band keeps the full temporal resolution.
        # Each band is identity-anchored by its own gate (init 0), so with all
        # gates at 0 the model reduces to the single-scale baseline.
        self.wpo_multiscale = False
        self.wpo_bands = None
        if multi_scale:
            if isinstance(multi_scale_factors, str):
                parsed = tuple(int(x) for x in multi_scale_factors.split(','))
            else:
                parsed = tuple(multi_scale_factors)
            self.multi_scale_factors = parsed
            # 'pool'    = F.avg_pool1d(kernel=f, stride=f), the historical behaviour
            # 'decimate'= take every f-th frame; same stride and aliasing, no mixing.
            # Only consulted when multi_scale is on; the default keeps every prior
            # run bit-identical.
            self.ms_pool_mode = (ms_pool_mode or 'pool').lower()
            if use_wpo and wpo_multiscale:
                bands = [float(v) for v in (wpo_band_c.split(',')
                                            if isinstance(wpo_band_c, str) else wpo_band_c)]
                self.wpo_multiscale = True
                self.wpo_band_mode = wpo_band_mode
                if wpo_band_mode == 'fixed':
                    # Control arm: parameter-free canonical band views. Kernels
                    # come from the multi-scale factors (largest → low band,
                    # smallest → high band), mirroring the classic pooled
                    # multi-scale granularity.
                    ks = sorted((f for f in parsed if f > 1), reverse=True) or [16, 8]
                    self.wpo_bands = nn.ModuleList([
                        FixedBandFilter(visual_width, kernel=ks[i % len(ks)],
                                        kind=('low', 'high')[i % 2])
                        for i in range(len(bands))])
                    print("[PlanB] fixed band filters: "
                          + ", ".join(f"{b.kind}@k{b.kernel}" for b in self.wpo_bands))
                else:
                    self.wpo_bands = nn.ModuleList([
                        TemporalWPO(visual_width, alpha_init=wpo_alpha_init, c_init=bc,
                                    c_max=wpo_c_max, internal_norm=False)
                        for bc in bands])
                num_scales = len(bands) + 1
            else:
                num_scales = len([f for f in parsed if f > 1]) + 1
        else:
            self.multi_scale_factors = ()
            num_scales = 1
        # ===== Wave resonator bank (driven damped-wave band-pass views) =====
        # Two insertion points, both cheap:
        #   fusion : extra scales appended to the multi-scale fusion. The existing
        #            pooled scales are avg_pool views = pure low-pass and highly
        #            redundant with each other; the resonator contributes genuinely
        #            band-pass evidence (|H(0)| = 1/sqrt(1+Q^2) < 1).
        #   block  : replaces TemporalWPO inside each Mamba block (same slot, needs
        #            --wpo-parallel semantics for the deviation to bypass the SSM).
        if self.use_wrb and wrb_where in ('fusion', 'both'):
            self.wrb_fusion = WaveResonatorBank(
                visual_width, num_bands=len(self.wrb_periods), periods=self.wrb_periods,
                q=self.wrb_q, kind=wrb_kind, n_groups=int(wrb_groups),
                adaptive=bool(wrb_adaptive),
                motion_cond=bool(wrb_adaptive and wrb_motion_cond),
                gain_init=float(wrb_gain_init), residual=False)
            num_scales += len(self.wrb_periods)
            print(f"[WRB] fusion scales += {len(self.wrb_periods)} "
                  f"(kind={wrb_kind}, P={self.wrb_periods}, Q={self.wrb_q}, "
                  f"groups={wrb_groups}, adaptive={wrb_adaptive})")

        # 'pre' : wave operator on the CLIP features BEFORE the encoder - changes
        #         the dynamics Mamba sees, and leaves the fusion layer untouched
        #         (parameter-matched against the no-branch baseline).
        # 'fusion': one extra scale (fusion grows 3->4 scales; +512*512 params).
        if wave_tv and wave_tv_where in ('pre', 'fusion'):
            _p, _q = _wave_pq(wave_tv_periods, wave_tv_q)
            self.wave_tv = TimeVaryingWaveBank(
                visual_width, periods=_p, qs=_q, size=int(wave_tv_size),
                gate_hidden=max(16, visual_width // 8),
                gain_init=float(wave_tv_gain),
                residual=(wave_tv_where == 'pre'),
                k_bands=int(wave_tv_k), p_min=float(wave_tv_pmin))
            if wave_tv_where == 'fusion':
                self.wave_tv_proj = nn.Linear(visual_width, visual_width)
                num_scales += 1
            print(f"[WaveTV] {wave_tv_where} insertion, P={_p}, Q={_q}, "
                  f"size={wave_tv_size}, gain={wave_tv_gain}, "
                  f"k={wave_tv_k}, pmin={wave_tv_pmin}")

        self.adaptive_fusion = bool(adaptive_fusion and (multi_scale or self.wrb_fusion))
        self.fusion = AdaptiveFusion(visual_width, num_scales) if self.adaptive_fusion else None
        self.fusion_linear = nn.Linear(visual_width * num_scales, visual_width)

        # ===== MLP classifier (no GCN) =====
        self.classifier = nn.Sequential(
            nn.Linear(visual_width, visual_width // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(visual_width // 2, 1)
        )

        # ===== Class-prototype anchors for multi-class MIL (EMA, not learned) =====
        # Targeted fix for the failed learnable-prototype version (AP 0.7072 vs
        # 0.7713): (a) prototypes are an EMA of per-class pooled features instead
        # of gradient-learned parameters — no moving-target co-adaptation with the
        # features; (b) initialised from the first videos of each class instead of
        # random — data-driven anchors from the start, mirroring the dual branch's
        # FROZEN CLIP text anchors; (c) CLASM gradient flows to the features only
        # (the buffer has no grad), so it cannot vent through the WPO filters.
        self.use_class_proto = use_class_proto
        if use_class_proto:
            self.register_buffer('class_proto', torch.zeros(num_class, visual_width))
            self.register_buffer('class_proto_seen', torch.zeros(num_class))
            self.class_proto_momentum = 0.999

    @torch.no_grad()
    def update_class_proto(self, pooled, labels_onehot):
        """EMA update of per-class anchor = mean pooled feature of that class's
        videos. pooled: (B, D) detached; labels_onehot: (B, K) multi-hot."""
        for k in range(self.class_proto.shape[0]):
            mask = labels_onehot[:, k] > 0
            if mask.any():
                mean_k = F.normalize(pooled[mask].float().mean(dim=0), dim=-1)
                if self.class_proto_seen[k] > 0:
                    self.class_proto[k] = (self.class_proto_momentum * self.class_proto[k]
                                           + (1 - self.class_proto_momentum) * mean_k)
                else:
                    self.class_proto[k] = mean_k
                    self.class_proto_seen[k] = 1.0
        self.class_proto.copy_(F.normalize(self.class_proto.float(), dim=-1))

    def trainable_state_dict(self):
        """Alias for state_dict (CLIP removed from model, no filtering needed)."""
        return self.state_dict()

    @torch.no_grad()
    def wrb_describe(self):
        """One-line summary of the live resonator parameters, for the epoch log."""
        parts = []
        if self.wrb_fusion is not None:
            parts.append("fusion[" + self.wrb_fusion.describe() + "]")
        for i, layer in enumerate(getattr(self.mamba, 'layers', [])):
            w = getattr(layer, 'wpo', None)
            if isinstance(w, WaveResonatorBank):
                parts.append(f"block{i}[" + w.describe() + "]")
        return " | ".join(parts) if parts else "off"

    # ──────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────

    def forward(self, visual, lengths=None, normal_mask=None):
        B, T, D = visual.shape

        # padding mask (True = padding) from per-sample lengths; threaded into the
        # WPO replicate-tail (Mamba scans stay unmasked, matching the baseline).
        padding_mask = None
        if lengths is not None:
            lengths = lengths.to(visual.device)
            padding_mask = (torch.arange(T, device=visual.device).unsqueeze(0)
                            >= lengths.clamp(min=1).unsqueeze(1))

        # Time-varying wave operator, pre-encoder: identity-anchored residual on
        # the CLIP features, so the fusion layer and its parameter count are
        # unchanged - any delta is the wave operator, not extra capacity.
        if self.wave_tv is not None and self.wave_tv_where == 'pre':
            visual = self.wave_tv(visual, padding_mask)

        # Single Mamba encode
        x = self._encode_video(visual, self.mamba, padding_mask)       # [T, B, D]
        x_vis = x.permute(1, 0, 2)                       # [B, T, D]

        # Future score prediction from raw Mamba hidden state
        score_pred = self.future_score_head(x_vis[:, :-1, :])
        score_pred = F.pad(score_pred, (0, 0, 0, 1), mode='replicate')

        # Multi-scale fusion (shared Mamba encoder)
        scales = [x_vis]
        if self.multi_scale:
            if self.wpo_multiscale:
                # Band-split scales: WPO-filtered (identity-anchored) views of the
                # FULL-resolution sequence, each encoded separately.
                for wpo_b in self.wpo_bands:
                    xb = wpo_b(visual, padding_mask)
                    xs = self._encode_video(xb, self.mamba, padding_mask).permute(1, 0, 2)
                    scales.append(xs)
            else:
                for f in self.multi_scale_factors:
                    if f == 1:
                        continue
                    if getattr(self, 'ms_pool_mode', 'pool') == 'decimate':
                        # --- decimation view (the Reviewer-1-#3 control) -------
                        # Held fixed against the pooling path: the stride, hence
                        # the output sampling rate and the nominal passband; the
                        # output length; the mask convention; and every weight.
                        # Kept frame k is exactly input frame k*f, with no boxcar.
                        #
                        # What this does and does not separate.  The contrast we
                        # wanted to break apart is (1,8,16) against (1,2,4), which
                        # moves the stride and the pooling kernel at once.
                        # Replacing the kernel at a fixed stride removes one of
                        # the two, so the stride is no longer a candidate
                        # explanation for the pooling arm's advantage.
                        #
                        # It is NOT a clean test of "cross-frame mixing" on its
                        # own.  F.avg_pool1d(kernel=f, stride=f) is the standard
                        # anti-aliased downsample: the boxcar is simultaneously
                        # (a) an average over f consecutive frames and (b) a
                        # low-pass that suppresses the fold-over of content above
                        # the new Nyquist rate.  Dropping it introduces aliasing
                        # as well as removing the averaging, so a difference here
                        # is attributable to the kernel, not to (a) alone.
                        # Separating (a) from (b) needs a third arm (e.g. a
                        # windowed-sinc resampler, which anti-aliases without
                        # averaging a centred window), which we did not run.
                        pooled_t = visual.float()
                        # Length must match the pooling path exactly, or the two
                        # arms would differ in a second, unintended way.
                        # F.avg_pool1d(..., ceil_mode=False) yields
                        # floor((T - f) / f) + 1 frames and drops the tail, while
                        # a bare v[:, ::f, :] would yield ceil(T / f); truncate to
                        # the pooled length so the two agree on shape.
                        t_len = pooled_t.shape[1]
                        out_len = (t_len - f) // f + 1
                        pooled = pooled_t[:, ::f, :][:, :out_len, :].to(visual.dtype)
                        mask_s = None
                        if padding_mask is not None:
                            # Position k of the decimated view is input frame
                            # k*f, so its true mask is that frame's mask --
                            # more exact than avg_pool's "any real frame in the
                            # window" test, which the pooling path still uses.
                            mask_s = padding_mask[:, ::f][:, :out_len]
                    else:
                        pooled = F.avg_pool1d(visual.float().permute(0, 2, 1),
                                              kernel_size=f, stride=f).permute(0, 2, 1).to(visual.dtype)
                        mask_s = None
                        if padding_mask is not None:
                            m = F.avg_pool1d(padding_mask.float().unsqueeze(1),
                                             kernel_size=f, stride=f).squeeze(1)
                            mask_s = m > 0.5
                    xs = self._encode_video(pooled, self.mamba, mask_s).permute(1, 0, 2)
                    xs = F.interpolate(xs.permute(0, 2, 1), size=T, mode='linear', align_corners=False).permute(0, 2, 1)
                    scales.append(xs)
        if self.wrb_fusion is not None:
            # Band-pass views of the ENCODED sequence: no second Mamba pass, so
            # this costs one rFFT/irFFT per band instead of a full re-encode.
            scales.extend(self.wrb_fusion.band_views(x_vis, padding_mask))
        if self.wave_tv is not None and self.wave_tv_where == 'fusion':
            scales.append(self.wave_tv_proj(self.wave_tv(x_vis, padding_mask)))
        if self.multi_scale or self.wrb_fusion is not None:
            x_vis = torch.cat(scales, dim=-1)
            if self.adaptive_fusion:
                x_fused = self.fusion(scales)

        if not ((self.multi_scale or self.wrb_fusion is not None) and self.adaptive_fusion):
            x_fused = self.fusion_linear(x_vis)

        # Class-prototype similarity: (B, T, K) multi-class frame logits.
        # Prototypes are a no-grad EMA buffer, so this loss's gradient reaches
        # only the features (features align to slowly-moving class anchors).
        # / 0.07 replicates the dual branch's logit scale (CLIP temperature):
        # without it the similarities span only [-1, 1] and the per-class
        # softmax is nearly uniform — the multi-class signal is diluted ~14x.
        logits_proto = None
        if self.use_class_proto:
            x_n = F.normalize(x_fused.float(), dim=-1)
            p_n = F.normalize(self.class_proto.float(), dim=-1)
            logits_proto = (x_n @ p_n.t()) / 0.07

        # Classifier (MLP, no GCN)
        logits1 = self.classifier(x_fused)

        outputs = dict(
            hidden_state=x_vis,
            hidden_state_fused=x_fused,
            logits1=logits1,
            score_future=score_pred,
            logits_proto=logits_proto,
        )

        return outputs

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _encode_video(self, images, mamba_module, padding_mask=None):
        """[B, T, D] → position embed → mamba → [T, B, D]"""
        B, T, D = images.shape
        images = images.permute(1, 0, 2)
        pos_ids = torch.arange(T, device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
        pos_emb = self.frame_position_embeddings(pos_ids)
        pos_emb = pos_emb.permute(1, 0, 2)
        images = images + pos_emb
        x = mamba_module(images, padding_mask)
        return x
