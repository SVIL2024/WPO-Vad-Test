from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from wave_field import WaveField

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device,
                 wave_field: int = 0,
                 wave_gain: float = 0.1,
                 wave_fusion: str = "logits"):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        # ---- wave-equation evidence channel (see wave_field.py) -------------
        # wave_field == 0 keeps the model byte-identical to the baseline, so
        # every existing checkpoint and every A/B comparison stays valid.
        self.wave_field = None
        if wave_field and int(wave_field) > 0:
            self.wave_field = WaveField(dim=visual_width, n_modes=int(wave_field))
            # Learnable scalar. NOT zero-initialised: with gain == 0 the
            # gradient into the whole wave branch is exactly zero, so the mode
            # basis would never train.
            self.wave_gain = nn.Parameter(torch.tensor(float(wave_gain)))
            # Remembered so sanitize_wave_branch() can rebuild the branch from
            # scratch if a warm start hands us NaNs (see that method).
            self._wave_cfg = {"dim": int(visual_width),
                              "n_modes": int(wave_field),
                              "gain": float(wave_gain)}
        # 'logits' = E1/E2 behaviour (add to the attention logits).
        # 'feature' = E3 (gate the features that get pooled).
        self.wave_fusion = str(wave_fusion)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def sanitize_wave_branch(self):
        """Re-initialise the wave branch if a warm start poisoned it with NaN.

        Why this exists
        ---------------
        A dead wave branch reproduces the control's AP/AUC *exactly*, so its
        checkpoint is saved as `xd_best_ap0.8490_auc0.9390_ep1_wf192...pt` -
        which shares a prefix with the control `..._ep4.pt`. `_job_paperE2.sh`
        selected its control with `ls -1 <prefix>* | head -1`, and
        `_ep1_` sorts before `_ep4`, so the second launch warm-started from the
        previous run's poisoned weights (2026-09-16). Those carry all-NaN wave
        tensors, which `load_state_dict(strict=False)` copies straight into a
        fresh model. The branch is then dead *before step 0*: every batch
        reports non-finite energy, the gradient is exactly 0, and the run
        silently measures nothing while scoring identical to the control.

        Returns the number of tensors repaired (0 == the checkpoint was clean).
        Nothing is re-initialised on the clean path, so RNG consumption - and
        therefore the DataLoader's shuffle order - is unchanged.
        """
        if self.wave_field is None:
            return 0
        bad = [n for n, p in self.named_parameters()
               if "wave" in n and p.is_floating_point()
               and not bool(torch.isfinite(p).all())]
        if not bad:
            return 0
        cfg = getattr(self, "_wave_cfg", None)
        if not cfg:
            return 0
        dev = self.wave_field.mix.weight.device
        dt = self.wave_field.mix.weight.dtype
        fresh = WaveField(dim=cfg["dim"], n_modes=cfg["n_modes"]).to(device=dev,
                                                                     dtype=dt)
        self.wave_field.load_state_dict(fresh.state_dict())
        with torch.no_grad():
            self.wave_gain.fill_(cfg["gain"])
        print(f"[InitCkpt] REPAIRED {len(bad)} NaN wave tensor(s) handed over "
              f"by the warm start -- branch re-initialised: {bad[:4]}")
        return len(bad)

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def _wave_evidence(self, visual, lengths):
        """Per-video standardised wave-source energy, (B, T, 1).

        Standardising per video is what makes the fusion scale-free: the raw
        energy is O(0.5) but its useful part is the WITHIN-VIDEO contrast, and
        the absolute level depends on the input-dependent alpha/c.
        """
        energy = self.wave_field(visual, lengths=lengths)["energy"]     # (B, T)
        # Standardise in fp32. The pre-extracted CLIP features are stored as
        # float16, so `visual` (and therefore `energy`) is half precision here,
        # while the trunk casts its own input to fp32 inside encode_video.
        # The E0 probe measured +0.024 AUC with fp32 arithmetic - keeping the
        # standardisation in half would silently change the measurement, and a
        # near-zero variance clips to 0 in fp16 and blows the z-score up.
        energy = energy.float()
        # ---- non-finite guard --------------------------------------------
        # One non-finite energy entry poisons the mean and the variance, and
        # the z-score then turns the whole batch into NaN. It surfaces much
        # later and far away, as
        #     Loss.cu:94 Assertion `input_val >= zero && input_val <= one`
        # inside F.binary_cross_entropy (that is what killed E1 on 2026-09-16,
        # on its very first forward). Count it out loud instead of dying: if
        # this ever prints, the branch is partly dead and the run's number is
        # not trustworthy.
        n_bad = int(torch.isfinite(energy).logical_not().sum())
        if n_bad:
            self._wave_bad = getattr(self, "_wave_bad", 0) + n_bad
            if self._wave_bad == n_bad:        # first occurrence only
                print(f"[WaveField] WARNING: non-finite wave energy "
                      f"({n_bad}/{energy.numel()} entries) -- zeroing them. "
                      f"dtype in={visual.dtype}")
            energy = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
        B, T = energy.shape
        if lengths is None:
            mask = torch.ones_like(energy)
        else:
            mask = (torch.arange(T, device=energy.device)[None, :]
                    < lengths.to(energy.device).clamp(max=T)[:, None]).to(energy.dtype)
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        mu = (energy * mask).sum(1, keepdim=True) / denom
        var = ((energy - mu) ** 2 * mask).sum(1, keepdim=True) / denom
        # ---- variance floor must be RELATIVE, not absolute ----------------
        # This is what killed the branch (found 2026-09-16 by replaying real
        # training steps: clean for ~10 steps, then every batch went non-finite
        # forever, gradient 0, evidence exactly 0 - E2a's AP was then
        # bit-identical to the A0 control and the run was measuring nothing).
        #
        # A short clip, or one with near-static content, has an energy trace
        # whose variance sits at the floating-point noise floor. Dividing by
        # `clamp(min=1e-6)` turns rounding error into a z-score of ~1e6; the
        # resulting gradient NaNs the wave branch's parameters within a dozen
        # steps, and because every entry then shares the NaN parameter the
        # whole (B, T) block goes non-finite at once - which is exactly the
        # 24576/24576 that looked like "bad data" for so long.
        #
        # Floor the scale at 0.1% of the signal level instead: variation below
        # that really is noise, and above it nothing changes.
        # The epsilon is INSIDE the sqrt on purpose: d/dvar sqrt(var) = 1/(2
        # sqrt(var)) is Inf at var == 0, and that Inf is what actually NaNs the
        # branch (the smoke test's near-constant clip has var EXACTLY 0 once the
        # ripple rounds away in fp32). With the epsilon the derivative is
        # bounded by 1/(2*eps^0.5) no matter how degenerate the clip is.
        sig = (var + 1e-12).sqrt()
        # Relative to the signal, BUT never below an absolute 1e-4: for a clip
        # whose energy is identically zero (static content) mu is 0, a purely
        # relative floor collapses to ~1e-11, and 1/sig then reaches 1e11 -
        # which is exactly the gradient that NaNs the branch. Capping 1/sig at
        # 1e4 is what makes the whole thing safe.
        floor = torch.clamp(1e-3 * mu.abs(), min=1e-4)
        sig = torch.maximum(sig, floor)
        z = (energy - mu) / sig
        # Belt and braces: a standardised score beyond +-8 is either a genuine
        # outlier or the degenerate case above. Either way it must not be the
        # thing that decides the gradient, so saturate it (clamp zeroes the
        # gradient outside the range, which is what we want).
        z = torch.clamp(z, -8.0, 8.0)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        out = (z * mask).unsqueeze(-1)
        # Telemetry. E2a's wave branch changed the REPORTED probability by
        # 8e-5 (mean|diff| against the control's dump) and we only found out
        # by diffing score dumps after the fact - nothing in the training log
        # distinguished "branch is alive but has no lever" from "branch is
        # dead". Keep the last evidence so the epoch heartbeat can print its
        # magnitude. Detached, and .item() is only called once per epoch, so
        # this costs no extra synchronisation.
        self._ev_last = out.detach()
        return out

    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        if self.wave_field is not None:
            # Under autocast `visual` is fp16/bf16 while wave_gain stays an fp32
            # parameter; plain multiplication would promote the sum to fp32 and
            # the later `logits_attn @ visual_features` would then hit a dtype
            # mismatch. Pin both operands to logits1's dtype.
            evid = self._wave_evidence(visual, lengths).to(logits1.dtype)
            if self.wave_fusion == "feature":
                # ---- E3: gate the FEATURES, not the attention weights -------
                # E2 measured both arms against A0 and both missed by orders of
                # magnitude (E2a +0.0000255, E2b -0.0184) even though the branch
                # was alive and AUC1 rose in both. Adding a per-video z-scored
                # term to the attention logits has almost no lever: it mostly
                # shifts all frames together, and the pooled vector is a
                # weighted average that shrugs off a common offset.
                # Multiplying the features instead changes WHAT gets pooled.
                # tanh keeps the gate in (1-|g|, 1+|g|) so a large gain can
                # never zero out or blow up a frame's contribution, and the
                # gate is 1 at initialisation-adjacent gains so step 0 is close
                # to the warm-started control.
                gate = 1.0 + self.wave_gain.to(logits1.dtype) * torch.tanh(evid)
                visual_features = visual_features * gate
                # logits1 is derived from the features, so recompute it -
                # otherwise the gate would change the pooling but not the
                # frame-level score, and the two would disagree.
                logits1 = self.classifier(
                    visual_features + self.mlp2(visual_features))
            else:
                logits1 = logits1 + self.wave_gain.to(logits1.dtype) * evid

        text_features_ori = self.encode_textprompt(text)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2
    