import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.distributions.normal import Normal

# Vectorization notes:
# 1) multi-kernel trend decomposition uses one cumulative-sum pipeline;
# 2) factorized WeightGenerator batches all requested matrices/biases;
# 3) Transformer_Layer already folds patch index into batch dimension;
# 4) the outer AMS expert loop is intentionally retained because experts
#    have heterogeneous patch sizes/parameter shapes and sparse top-k routing.



class SparseDispatcher(object):
    """
    GPU-friendlier sparse dispatcher.

    Keeps the original sparse top-k routing semantics, while:
      - calling nonzero() only once
      - sorting routes once
      - avoiding einsum for scalar gate multiplication
      - avoiding requires_grad=True on the zero accumulation tensor

    torch.split() still requires Python split sizes, so one small
    GPU->CPU synchronization remains here.
    """
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        # [num_routes, 2] -> (batch_id, expert_id)
        routes = torch.nonzero(
            gates,
            as_tuple=False
        )

        # Group routes by expert.
        order = torch.argsort(routes[:, 1])
        routes = routes.index_select(0, order)

        self._batch_index = routes[:, 0]
        self._expert_index = routes[:, 1:2]

        # torch.split currently needs host-side integer sizes.
        counts = torch.bincount(
            routes[:, 1],
            minlength=num_experts
        )
        self._part_sizes = counts.tolist()

        # One scalar gate for each routed sample.
        self._nonzero_gates = gates[
            self._batch_index,
            self._expert_index.squeeze(1)
        ].unsqueeze(1)

    def dispatch(self, inp):
        routed = inp.index_select(
            0,
            self._batch_index
        )

        return torch.split(
            routed,
            self._part_sizes,
            dim=0
        )

    def combine(self, expert_out, multiply_by_gates=True):
        if len(expert_out) == 0:
            raise RuntimeError(
                "SparseDispatcher received no expert outputs."
            )

        stitched = torch.cat(
            expert_out,
            dim=0
        )

        if multiply_by_gates:
            gate = self._nonzero_gates.view(
                -1, 1, 1, 1
            )
            stitched = stitched * gate

        output = stitched.new_zeros(
            self._gates.size(0),
            stitched.size(1),
            stitched.size(2),
            stitched.size(3),
        )

        output = output.index_add(
            0,
            self._batch_index,
            stitched
        )

        return output

    def expert_to_gates(self):
        return torch.split(
            self._nonzero_gates,
            self._part_sizes,
            dim=0
        )

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(
            1,
            self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2),
            1
        )
        end = x[:, -1:, :].repeat(
            1,
            math.floor((self.kernel_size - 1) // 2),
            1
        )
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    """
    Vectorized multi-kernel moving-average trend decomposition.

    The original Pathformer implementation evaluates one AvgPool1d module
    per kernel. Here all moving-average windows are computed together from
    one replicate-padded cumulative sum.

    Input:
        x: [B, L, C]

    Output:
        res:         [B, L, C]
        moving_mean: [B, L, C]

    This preserves the original asymmetric endpoint replication:
        left_pad  = kernel // 2
        right_pad = (kernel - 1) // 2
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()

        if len(kernel_size) == 0:
            raise ValueError("kernel_size must contain at least one kernel.")

        kernels = tuple(int(k) for k in kernel_size)
        if any(k <= 0 for k in kernels):
            raise ValueError(f"All moving-average kernels must be positive, got {kernels}.")

        self.kernel_size = kernels
        self.num_kernels = len(kernels)
        self.max_left_pad = max(k // 2 for k in kernels)
        self.max_right_pad = max((k - 1) // 2 for k in kernels)

        # Non-persistent because these values are structural constants and
        # should not change existing checkpoint state_dict compatibility.
        self.register_buffer(
            "_kernels",
            torch.tensor(kernels, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_left_pad",
            torch.tensor([k // 2 for k in kernels], dtype=torch.long),
            persistent=False,
        )

        self.layer = nn.Linear(1, self.num_kernels)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"series_decomp_multi expects [B, L, C], got {tuple(x.shape)}"
            )

        batch_size, seq_len, channels = x.shape

        # [B, L, C] -> [B, C, L]
        x_t = x.transpose(1, 2)

        # One common replicate padding for every kernel.
        # Each individual kernel uses an offset inside this common padded
        # sequence so that its result is exactly aligned with the original
        # moving_avg implementation.
        x_pad = F.pad(
            x_t,
            (self.max_left_pad, self.max_right_pad),
            mode="replicate",
        )

        # Prefix sums with an explicit zero at index 0.
        # Shape: [B, C, L_pad + 1]
        prefix = F.pad(
            x_pad.cumsum(dim=-1),
            (1, 0),
            mode="constant",
            value=0.0,
        )

        kernels = self._kernels.to(device=x.device)
        left_pad = self._left_pad.to(device=x.device)

        # For kernel k, original moving_avg uses left padding k//2.
        # Since x_pad uses max_left_pad for every kernel, shift each window
        # start by (max_left_pad - k//2).
        positions = torch.arange(seq_len, device=x.device)
        starts = (
            (self.max_left_pad - left_pad).unsqueeze(1)
            + positions.unsqueeze(0)
        )
        ends = starts + kernels.unsqueeze(1)

        # Gather every kernel/window in one tensorized operation.
        # prefix_expanded is a view; expand does not materialize copies.
        prefix_expanded = prefix.unsqueeze(2).expand(
            batch_size,
            channels,
            self.num_kernels,
            -1,
        )

        gather_shape = (
            batch_size,
            channels,
            self.num_kernels,
            seq_len,
        )
        start_idx = starts.view(
            1, 1, self.num_kernels, seq_len
        ).expand(gather_shape)
        end_idx = ends.view(
            1, 1, self.num_kernels, seq_len
        ).expand(gather_shape)

        window_sum = (
            torch.gather(prefix_expanded, dim=-1, index=end_idx)
            - torch.gather(prefix_expanded, dim=-1, index=start_idx)
        )

        kernel_scale = kernels.to(
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, self.num_kernels, 1)

        # [B, C, K, L] -> [B, L, C, K]
        moving_mean_all = (
            window_sum / kernel_scale
        ).permute(0, 3, 1, 2)

        weights = F.softmax(
            self.layer(x.unsqueeze(-1)),
            dim=-1,
        )

        moving_mean = torch.sum(
            moving_mean_all * weights,
            dim=-1,
        )

        res = x - moving_mean
        return res, moving_mean



class FourierLayer(nn.Module):
    """
    Fourier seasonality extraction.

    Optimized version avoids building meshgrid index tensors on
    every forward pass and uses gather() directly.
    """
    def __init__(
        self,
        pred_len,
        k=None,
        low_freq=1,
        output_attention=False
    ):
        super().__init__()
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        b, t, d = x.shape

        x_freq = fft.rfft(
            x,
            dim=1
        )

        full_f = fft.rfftfreq(
            t,
            device=x.device
        )

        if t % 2 == 0:
            x_freq = x_freq[
                :,
                self.low_freq:-1
            ]
            f = full_f[
                self.low_freq:-1
            ]
        else:
            x_freq = x_freq[
                :,
                self.low_freq:
            ]
            f = full_f[
                self.low_freq:
            ]

        k = self.k
        if k is None:
            k = x_freq.shape[1]

        k = min(
            k,
            x_freq.shape[1]
        )

        _, indices = torch.topk(
            x_freq.abs(),
            k,
            dim=1,
            largest=True,
            sorted=True
        )

        # [B, k, D]
        x_freq = torch.gather(
            x_freq,
            dim=1,
            index=indices
        )

        # Frequency grid [B, F, D], then gather using
        # exactly the same top-k indices.
        f_grid = f.view(
            1, -1, 1
        ).expand(
            b, -1, d
        )

        f = torch.gather(
            f_grid,
            dim=1,
            index=indices
        ).unsqueeze(2)

        return self.extrapolate(
            x_freq,
            f,
            t
        ), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat(
            [x_freq, x_freq.conj()],
            dim=1
        )

        f = torch.cat(
            [f, -f],
            dim=1
        )

        t_val = torch.arange(
            t + self.pred_len,
            dtype=torch.float,
            device=x_freq.device
        ).view(
            1, 1, -1, 1
        )

        amp = x_freq.abs().unsqueeze(2) / t
        phase = x_freq.angle().unsqueeze(2)

        x_time = amp * torch.cos(
            2 * math.pi * f * t_val + phase
        )

        return x_time.sum(dim=1)

class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            return torch.matmul(input, weights) + biases


class WeightGenerator(nn.Module):
    """
    Vectorized Pathformer dynamic/factorized weight generator.

    ParameterList layout is intentionally preserved so checkpoints produced
    by the previous implementation remain loadable. The forward path stacks
    P/Q/B once and generates all dynamic weights/biases in batched tensor
    operations instead of one Python matmul chain per requested weight.
    """
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()

        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim

        if self.factorized:
            self.memory = nn.Parameter(
                torch.randn(num_nodes, mem_dim),
                requires_grad=True
            )

            self.generator = nn.Sequential(
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            )

            self.mem_dim = 10

            self.P = nn.ParameterList([
                nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True)
                for _ in range(number_of_weights)
            ])
            self.Q = nn.ParameterList([
                nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True)
                for _ in range(number_of_weights)
            ])
            self.B = nn.ParameterList([
                nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True)
                for _ in range(number_of_weights)
            ])
        else:
            self.P = nn.ParameterList([
                nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
                for _ in range(number_of_weights)
            ])
            self.B = nn.ParameterList([
                nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True)
                for _ in range(number_of_weights)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if not self.factorized:
            return self.P, self.B

        # Generator output:
        # [N, mem_dim_in] -> [N, 100] -> [N, 10, 10]
        memory_flat = self.generator(
            self.memory.unsqueeze(1)
        ).squeeze(1)

        memory = memory_flat.view(
            self.num_nodes,
            self.mem_dim,
            self.mem_dim,
        )

        # Preserve ParameterList names/state_dict, but batch the actual math.
        # P: [W, I, M]
        # Q: [W, M, O]
        # B: [W, M*M, O]
        P = torch.stack(tuple(self.P), dim=0)
        Q = torch.stack(tuple(self.Q), dim=0)
        B = torch.stack(tuple(self.B), dim=0)

        # All W dynamic matrices at once:
        #   P_w @ Memory_n @ Q_w
        # -> [W, N, I, O]
        weights = torch.einsum(
            "wim,nmk,wko->wnio",
            P,
            memory,
            Q,
        )

        # All W dynamic biases at once:
        # [N, M*M] x [W, M*M, O] -> [W, N, O]
        bias = torch.einsum(
            "nm,wmo->wno",
            memory_flat,
            B,
        )

        # Existing callers use weights_distinct[i]/biases_distinct[i].
        # Returning tuples of tensor views keeps that interface unchanged.
        return weights.unbind(0), bias.unbind(0)



class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super(Intra_Patch_Attention, self).__init__()

        self.head = 2

        if d_model % self.head != 0:
            raise ValueError(
                "Hidden size is not divisible by "
                "the number of attention heads."
            )

        self.head_size = d_model // self.head
        self.custom_linear = CustomLinear(
            factorized
        )

    def forward(
        self,
        query,
        key,
        value,
        weights_distinct,
        biases_distinct,
        weights_shared,
        biases_shared
    ):
        """
        Shapes:
          query: [B, Lq, N, D]
          key:   [B, Lk, N, D]
          value: [B, Lk, N, D]

        The original implementation used split()+cat() to move the
        two attention heads into the batch dimension. reshape/permute
        does the same operation without allocating those temporary
        concatenations.
        """
        batch_size = query.shape[0]
        q_len = query.shape[1]
        k_len = key.shape[1]
        n_nodes = query.shape[2]

        key = self.custom_linear(
            key,
            weights_distinct[0],
            biases_distinct[0]
        )

        value = self.custom_linear(
            value,
            weights_distinct[1],
            biases_distinct[1]
        )

        h = self.head
        hd = self.head_size

        # [B, H, N, Lq, Hd]
        query = query.reshape(
            batch_size,
            q_len,
            n_nodes,
            h,
            hd
        ).permute(
            0, 3, 2, 1, 4
        )

        # [B, H, N, Hd, Lk]
        key = key.reshape(
            batch_size,
            k_len,
            n_nodes,
            h,
            hd
        ).permute(
            0, 3, 2, 4, 1
        )

        # [B, H, N, Lk, Hd]
        value = value.reshape(
            batch_size,
            k_len,
            n_nodes,
            h,
            hd
        ).permute(
            0, 3, 2, 1, 4
        )

        attention = torch.matmul(
            query,
            key
        )

        attention = attention * (
            hd ** -0.5
        )

        attention = F.softmax(
            attention,
            dim=-1
        )

        # [B, H, N, Lq, Hd]
        x = torch.matmul(
            attention,
            value
        )

        # -> [B, Lq, N, D]
        x = x.permute(
            0, 3, 2, 1, 4
        ).contiguous().reshape(
            batch_size,
            q_len,
            n_nodes,
            h * hd
        )

        x = self.custom_linear(
            x,
            weights_shared[0],
            biases_shared[0]
        )

        x = F.relu(
            x,
            inplace=False
        )

        x = self.custom_linear(
            x,
            weights_shared[1],
            biases_shared[1]
        )

        return x, attention

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention

        head_dim = d_model // n_heads
        self.scale = nn.Parameter(
            torch.tensor(head_dim ** -0.5),
            requires_grad=lsa
        )
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale

        if prev is not None:
            attn_scores = attn_scores + prev

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -np.inf
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class Inter_Patch_Attention(nn.Module):
    def __init__(
        self,
        d_model,
        out_dim,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.,
        proj_dropout=0.,
        qkv_bias=True,
        lsa=False
    ):
        super().__init__()

        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa
        )

        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, out_dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)

        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = self.W_Q(Q).view(bs, Q.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, K.shape[1], self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, V.shape[1], self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )

        output = output.transpose(1, 2).contiguous().view(
            bs,
            Q.shape[1],
            self.n_heads * self.d_v
        )

        output = self.to_out(output)
        return output, attn_weights


class Transformer_Layer(nn.Module):
    """
    Pathformer multi-scale Transformer expert:
    intra-patch attention + inter-patch attention.
    """
    def __init__(
        self,
        device,
        d_model,
        d_ff,
        num_nodes,
        patch_nums,
        patch_size,
        dynamic,
        factorized,
        layer_number,
        batch_norm
    ):
        super(Transformer_Layer, self).__init__()

        self.device = device
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.layer_number = layer_number
        self.batch_norm = batch_norm

        self.intra_embeddings = nn.Parameter(
            torch.rand(self.patch_nums, 1, 1, self.num_nodes, 16),
            requires_grad=True
        )

        self.embeddings_generator = nn.ModuleList([
            nn.Sequential(nn.Linear(16, self.d_model))
            for _ in range(self.patch_nums)
        ])

        self.intra_d_model = self.d_model
        self.intra_patch_attention = Intra_Patch_Attention(
            self.intra_d_model,
            factorized=factorized
        )

        self.weights_generator_distinct = WeightGenerator(
            self.intra_d_model,
            self.intra_d_model,
            mem_dim=16,
            num_nodes=num_nodes,
            factorized=factorized,
            number_of_weights=2
        )

        self.weights_generator_shared = WeightGenerator(
            self.intra_d_model,
            self.intra_d_model,
            mem_dim=None,
            num_nodes=num_nodes,
            factorized=False,
            number_of_weights=2
        )

        self.intra_Linear = nn.Linear(
            self.patch_nums,
            self.patch_nums * self.patch_size
        )

        self.stride = patch_size
        self.inter_d_model = self.d_model * self.patch_size

        self.emb_linear = nn.Linear(self.inter_d_model, self.inter_d_model)
        self.W_pos = nn.Parameter(
            torch.zeros(1, self.patch_nums, self.inter_d_model),
            requires_grad=True
        )

        n_heads = self.d_model
        d_k = self.inter_d_model // n_heads
        d_v = self.inter_d_model // n_heads

        self.inter_patch_attention = Inter_Patch_Attention(
            self.inter_d_model,
            self.inter_d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=0,
            proj_dropout=0.1,
            res_attention=False
        )

        self.norm_attn = nn.Sequential(
            nn.LayerNorm(self.d_model)
        )
        self.norm_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model)
        )

        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_ff, self.d_model, bias=True)
        )


    def forward(self, x):
        """
        Vectorized Pathformer expert.

        Original code launched intra-patch attention once per patch.
        With L=600:
          patch=3  -> 200 Python/CUDA launches
          patch=5  -> 120
          patch=15 -> 40

        Here the patch dimension is folded into the batch dimension,
        so one expert performs intra-patch attention in one batched
        operation.
        """
        new_x = x

        batch_size = x.size(0)
        seq_len = x.size(1)
        num_nodes = x.size(2)
        d_model = x.size(3)

        expected_len = (
            self.patch_nums
            * self.patch_size
        )

        if seq_len != expected_len:
            raise ValueError(
                f"Transformer_Layer expected temporal "
                f"length {expected_len}, got {seq_len}"
            )

        weights_shared, biases_shared = (
            self.weights_generator_shared()
        )

        weights_distinct, biases_distinct = (
            self.weights_generator_distinct()
        )

        # ----------------------------------------------------
        # Vectorized intra-patch embeddings.
        #
        # Original:
        #   for i in range(patch_nums):
        #       embeddings_generator[i](
        #           intra_embeddings[i]
        #       )
        #
        # Keep the original Parameter/ModuleList layout for
        # checkpoint compatibility, but stack their parameters
        # and evaluate all patch embeddings together.
        # ----------------------------------------------------

        embed_input = self.intra_embeddings[
            :, 0, 0, :, :
        ]
        # [P, N, 16]

        embed_weights = torch.stack(
            [
                module[0].weight
                for module
                in self.embeddings_generator
            ],
            dim=0
        )
        # [P, D, 16]

        embed_bias = torch.stack(
            [
                module[0].bias
                for module
                in self.embeddings_generator
            ],
            dim=0
        )
        # [P, D]

        patch_embeddings = torch.einsum(
            "pni,pdi->pnd",
            embed_input,
            embed_weights
        )

        patch_embeddings = (
            patch_embeddings
            + embed_bias[:, None, :]
        )
        # [P, N, D]

        patch_embeddings = (
            patch_embeddings
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
                -1
            )
        )
        # [B, P, N, D]

        # ----------------------------------------------------
        # [B, L, N, D]
        # -> [B, P, S, N, D]
        # ----------------------------------------------------

        patches = x.reshape(
            batch_size,
            self.patch_nums,
            self.patch_size,
            num_nodes,
            d_model
        )

        # Query:
        # [B*P, 1, N, D]
        intra_query = (
            patch_embeddings
            .reshape(
                batch_size
                * self.patch_nums,
                num_nodes,
                d_model
            )
            .unsqueeze(1)
        )

        # Key/value:
        # prepend one learned embedding to every patch.
        intra_kv = torch.cat(
            [
                patch_embeddings.unsqueeze(2),
                patches
            ],
            dim=2
        )

        intra_kv = intra_kv.reshape(
            batch_size * self.patch_nums,
            self.patch_size + 1,
            num_nodes,
            d_model
        )

        intra_out, intra_attention = (
            self.intra_patch_attention(
                intra_query,
                intra_kv,
                intra_kv,
                weights_distinct,
                biases_distinct,
                weights_shared,
                biases_shared
            )
        )

        # [B*P,1,N,D] -> [B,P,N,D]
        intra_out_concat = (
            intra_out
            .squeeze(1)
            .reshape(
                batch_size,
                self.patch_nums,
                num_nodes,
                d_model
            )
        )

        # Same projection as original implementation:
        # [B,P,N,D] -> [B,L,N,D]
        intra_out_concat = (
            intra_out_concat.permute(
                0, 3, 2, 1
            )
        )

        intra_out_concat = (
            self.intra_Linear(
                intra_out_concat
            )
        )

        intra_out_concat = (
            intra_out_concat.permute(
                0, 3, 2, 1
            )
        )

        # ----------------------------------------------------
        # Inter-patch path.
        #
        # Avoid unfold() and construct the same D*S ordering
        # directly from the existing patch view.
        # ----------------------------------------------------

        inter_x = patches.permute(
            0, 3, 1, 4, 2
        )
        # [B, N, P, D, S]

        inter_x = inter_x.reshape(
            batch_size * num_nodes,
            self.patch_nums,
            d_model * self.patch_size
        )

        inter_x = self.emb_linear(
            inter_x
        )

        inter_x = self.dropout(
            inter_x + self.W_pos
        )

        inter_out, inter_attention = (
            self.inter_patch_attention(
                Q=inter_x,
                K=inter_x,
                V=inter_x
            )
        )

        inter_out = inter_out.reshape(
            batch_size,
            num_nodes,
            self.patch_nums,
            self.patch_size,
            d_model
        )

        # Preserve the ORIGINAL Pathformer memory layout exactly.
        # Do NOT permute here.
        inter_out = inter_out.reshape(
            batch_size,
            self.patch_size * self.patch_nums,
            num_nodes,
            d_model
        )

        out = (
            new_x
            + intra_out_concat
            + inter_out
        )

        out = self.dropout(
            out
        )

        out = self.ff(
            out
        ) + out

        return out, inter_attention

class AMS(nn.Module):
    """
    Adaptive Multi-Scale Block from Pathformer.
    """
    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        device,
        num_nodes=1,
        d_model=256,
        d_ff=1024,
        dynamic=False,
        patch_size=(3, 5, 15),
        noisy_gating=True,
        k=2,
        layer_number=1,
        residual_connection=1,
        batch_norm=False
    ):
        super(AMS, self).__init__()

        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()

        for patch in patch_size:
            if input_size % patch != 0:
                raise ValueError(
                    f"Pathformer AMS requires input_size divisible by patch_size. "
                    f"Got input_size={input_size}, patch_size={patch}."
                )

            patch_nums = int(input_size / patch)

            self.experts.append(
                Transformer_Layer(
                    device=device,
                    d_model=d_model,
                    d_ff=d_ff,
                    dynamic=dynamic,
                    num_nodes=num_nodes,
                    patch_nums=patch_nums,
                    patch_size=patch,
                    factorized=True,
                    layer_number=layer_number,
                    batch_norm=batch_norm
                )
            )

        self.w_noise = nn.Linear(input_size, num_experts)
        self.w_gate = nn.Linear(input_size, num_experts)

        self.residual_connection = residual_connection
        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)

        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(
            batch,
            device=clean_values.device
        ) * m + self.k

        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in),
            1
        )

        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out),
            1
        )

        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf(
            (clean_values - threshold_if_in) / noise_stddev
        )
        prob_if_out = normal.cdf(
            (clean_values - threshold_if_out) / noise_stddev
        )

        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        x = self.start_linear(x).squeeze(-1)

        clean_logits = self.w_gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(
            min(self.k + 1, self.num_experts),
            dim=1
        )

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(
                clean_logits,
                noisy_logits,
                noise_stddev,
                top_logits
            ).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def forward(self, x, loss_coef=1e-2):
        new_x = self.seasonality_and_trend_decompose(x)

        gates, load = self.noisy_top_k_gating(new_x, self.training)

        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss = balance_loss * loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)

        expert_outputs = []
        for i in range(self.num_experts):
            if expert_inputs[i].shape[0] == 0:
                continue
            expert_outputs.append(self.experts[i](expert_inputs[i])[0])

        output = dispatcher.combine(expert_outputs)

        if self.residual_connection:
            output = output + x

        return output, balance_loss
class PathformerEncoder(nn.Module):
    """
    Adapter wrapper for MESM Transformer encoder output.

    Inserted after DETR Transformer Encoder and before Decoder.

    Input:
        memory_local: [B, L, D_in]

    Bottleneck:
        D_in -> path_d_model -> Pathformer AMS -> D_in

    Internal Pathformer format:
        [B, L_pad, N=1, path_d_model]

    Output:
        enhanced_memory_local: [B, L, D_in]
        balance_loss: scalar

    Only adaptations:
        1. Use bottleneck projection to avoid huge Pathformer parameters.
        2. Pad variable temporal length to fixed AMS input length, then crop back.
    """

    def __init__(
        self,
        input_size,
        d_model,
        d_ff,
        patch_size=(3, 5, 15),
        top_k=2,
        noisy_gating=True,
        residual_connection=1,
        batch_norm=False,
        path_d_model=32,
        path_d_ff=64,
    ):
        super(PathformerEncoder, self).__init__()

        self.input_size = input_size          # usually args.max_video_l, e.g. 194
        self.input_dim = d_model              # MESM hidden dim, e.g. 256
        self.path_d_model = path_d_model      # Pathformer internal dim, e.g. 32
        self.path_d_ff = path_d_ff            # Pathformer internal FFN dim, e.g. 64

        self.patch_size = tuple(patch_size)
        self.num_experts = len(self.patch_size)

        if top_k > self.num_experts:
            raise ValueError(
                f"path_top_k={top_k} must be <= number of experts={self.num_experts}"
            )

        for p in self.patch_size:
            if p <= 0:
                raise ValueError(f"Invalid patch_size={p}. It must be positive.")

        # Compute lcm of patch sizes.
        patch_lcm = 1
        for p in self.patch_size:
            patch_lcm = abs(patch_lcm * p) // math.gcd(patch_lcm, p)

        self.patch_lcm = patch_lcm

        # Build AMS with a fixed length that can be divided by all patch sizes.
        # Example: max_video_l=194, patch_size=[3,5,15] => ams_input_size=195.
        self.ams_input_size = int(math.ceil(input_size / patch_lcm) * patch_lcm)

        # Bottleneck adapter: 256 -> 32 -> 256
        self.down_proj = nn.Linear(self.input_dim, self.path_d_model)
        self.up_proj = nn.Linear(self.path_d_model, self.input_dim)

        self.ams = AMS(
            input_size=self.ams_input_size,
            output_size=self.ams_input_size,
            num_experts=self.num_experts,
            device=None,
            num_nodes=1,
            d_model=self.path_d_model,
            d_ff=self.path_d_ff,
            dynamic=False,
            patch_size=self.patch_size,
            noisy_gating=noisy_gating,
            k=top_k,
            layer_number=1,
            residual_connection=residual_connection,
            batch_norm=batch_norm
        )

        self.out_norm = nn.LayerNorm(self.input_dim)

    def forward(self, memory_local, loss_coef=1e-2):
        if memory_local.dim() != 3:
            raise ValueError(
                f"PathformerEncoder expects [B, L, D], got {tuple(memory_local.shape)}"
            )

        bsz, length, dim = memory_local.shape

        if dim != self.input_dim:
            raise ValueError(
                f"PathformerEncoder input dim mismatch: expected {self.input_dim}, got {dim}"
            )

        if length > self.ams_input_size:
            raise ValueError(
                f"Current sequence length={length} is larger than ams_input_size={self.ams_input_size}. "
                f"Please increase path_input_size or max_video_l."
            )

        residual = memory_local

        # 1) bottleneck down projection
        x = self.down_proj(memory_local)  # [B, L, path_d_model]

        # 2) pad temporal length to fixed AMS length
        pad_len = self.ams_input_size - length
        if pad_len > 0:
            pad_feat = x[:, -1:, :].expand(-1, pad_len, -1)
            x = torch.cat([x, pad_feat], dim=1)

        # 3) Pathformer AMS expects [B, L, N, D], use N=1
        x = x.unsqueeze(2)  # [B, L_pad, 1, path_d_model]

        x, balance_loss = self.ams(x, loss_coef=loss_coef)

        x = x.squeeze(2)  # [B, L_pad, path_d_model]

        # 4) crop back to original batch temporal length
        x = x[:, :length, :]

        # 5) bottleneck up projection
        x = self.up_proj(x)  # [B, L, input_dim]

        # 6) residual connection outside AMS
        x = self.out_norm(x + residual)

        return x, balance_loss