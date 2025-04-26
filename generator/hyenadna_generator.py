import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from einops import rearrange
from functools import partial
from torchvision.ops import StochasticDepth
import numpy as np
import math

def fftconv(u, k, D):
    """
    We apply a convolution through the fourier domain (from the Convolution Theorem)

    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    """The Sin activation function for the Hyena Filter function."""
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    """The window function applied to the output of the (MLP) filter function."""
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool=True,
        shift: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(
            self,
            d_model,
            emb_dim=3, # dim of input to MLP, augments with positional encoding
            order=16, # width of the implicit MLP
            fused_fft_conv=False,
            seq_len=1024,
            lr=1e-3,
            lr_pos_emb=1e-5,
            dropout=0.0,
            w=1, # frequency of periodic activations
            wd=0, # weight decay of kernel parameters
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
        ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()

        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2,
            filter_order=64,
            dropout=0.0,
            filter_dropout=0.0,
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')

        uc = self.short_filter(u)[...,:l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 return_residual=False, device=None, dtype=None):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input

class Block(nn.Module):
    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0.,
                 return_residual=False,
                 residual_in_fp32=False):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mixer_cls is None:
            mixer_cls = partial(HyenaOperator, d_model=dim, l_max=1024)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls()
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out

            hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                        + hidden_states).to(dtype=self.norm1.weight.dtype))

            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out

                hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                            + hidden_states).to(dtype=self.norm2.weight.dtype))

            return hidden_states


def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model

    mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate='tanh'), **factory_kwargs)

    return mlp_cls

def create_block(d_model, d_inner=None,
                 layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False,
                 layer_idx=None,
                 device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = partial(HyenaOperator, **layer)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner,
                             **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2,residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)


class GPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None,
                 word_embed_proj_dim=None, device=None, dtype=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
            If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
                the project up to embed_dim
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx,
                                                **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim,
                                                padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False,
                                        **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim,
                                                    **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

class LMBackbone(nn.Module):
    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.residual_in_fp32 = residual_in_fp32
        # note max_position_embeddings is 0 for Hyena, and therefore isn't used
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings,
                                             **factory_kwargs)

        self.layers = nn.ModuleList([create_block(
            d_model, d_inner=d_inner,
            layer=layer, attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg, layer_norm_epsilon=layer_norm_epsilon,
            resid_dropout1=embed_dropout if i == 0 else resid_dropout,
            resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32,layer_idx=i,
            **factory_kwargs,
        ) for i in range(n_layer)])

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)

        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        """
        Forward pass through the backbone model.
        
        Args:
            input_ids: Optional tensor of token IDs [batch_size, seq_len]
            position_ids: Optional tensor of position IDs [batch_size, seq_len]
            inputs_embeds: Optional pre-computed embeddings [batch_size, seq_len, d_model]
            
        Returns:
            hidden_states: Output hidden states [batch_size, seq_len, d_model]
        """
        # Either input_ids or inputs_embeds must be provided
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Generate embeddings if not provided directly
        if inputs_embeds is None:
            hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        else:
            hidden_states = inputs_embeds
        
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

        return hidden_states

class HyenaDNAModel(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 layer=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        # check if layer (config) has d_model (HF code differs from main Safari code)
        if 'd_model' not in layer:
            layer['d_model'] = d_model

        self.backbone = LMBackbone(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            layer=layer, 
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg, residual_in_fp32=residual_in_fp32,
            **factory_kwargs, **kwargs
        )

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

        # if self.use_head:
        #     self.tie_weights()

    # def tie_weights(self):
    #     self.head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, state=None): # state for the repo interface
        hidden_states = self.backbone(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)

        if self.use_head:
            return self.head(hidden_states)
        else:
            return hidden_states

class ConditionedHyenaDNA(nn.Module):
    """HyenaDNA model conditioned on a latent vector representing a distribution of DNA sequences."""
    
    def __init__(
        self, 
        latent_dim=32,
        condition_dim=128,
        d_model=128,
        n_layer=6,
        vocab_size=11,  # A, C, G, T, N
        max_seq_len=512,
        condition_method="prefix"
    ):
        """
        Initialize a conditioned HyenaDNA model.
        
        Args:
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            d_model: Model dimension
            n_layer: Number of layers
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            condition_method: How to condition the model ('prefix' or 'additive')
        """
        super().__init__()
        
        # Adjust max_seq_len to account for prefix token if using prefix conditioning
        self.actual_max_seq_len = max_seq_len + 1 if condition_method == "prefix" else max_seq_len
        
        # Configure parameters for the HyenaDNA model
        # The layer parameter needs to be a dictionary with specific settings
        hyena_layer_config = {
            'd_model': d_model,
            'l_max': self.actual_max_seq_len,  # Use adjusted max length for HyenaOperator
            'order': 2,            # Default Hyena recurrence depth
            'filter_order': 64     # Default filter order
        }
        
        self.model = HyenaDNAModel(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=4 * d_model,
            vocab_size=vocab_size,
            max_position_embeddings=self.actual_max_seq_len,  # Use adjusted max length
            layer=hyena_layer_config,  # Pass the layer config as a dictionary
            attn_layer_idx=None,       # No attention layers
            attn_cfg=None,             # No attention config
            layer_norm_epsilon=1e-5,
            embed_dropout=0.1,
            resid_dropout=0.0,
            residual_in_fp32=False
        )
        
        # Add a language model head for next token prediction
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.condition_method = condition_method
        self.hidden_dim = d_model
        self.max_seq_len = max_seq_len  # Keep original max_seq_len for input validation
        self.vocab_size = vocab_size
        
        # Project latent to condition dimension
        self.condition_proj = nn.Sequential(
            nn.Linear(latent_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, self.hidden_dim)
        )
        
        if self.condition_method == "prefix":
            # No additional projection needed for prefix conditioning
            # We'll directly use the condition as a prefix embedding
            pass
        elif self.condition_method == "additive":
            # For additive conditioning
            self.condition_adapter = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU()
            )
        else:
            raise ValueError(f"Unknown conditioning method: {condition_method}")
    
    def forward(self, input_ids, attention_mask, latent):
        """
        Forward pass through the conditioned HyenaDNA model.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len] or [batch_size, set_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] or [batch_size, set_size, seq_len] 
            latent: Latent distribution embedding [batch_size, latent_dim]
        
        Returns:
            Logits for next token prediction
        """
        batch_size = input_ids.shape[0]
        
        # Validate sequence length
        if len(input_ids.shape) == 3:
            seq_len = input_ids.shape[-1]
        else:
            seq_len = input_ids.shape[1]
            
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum allowed length {self.max_seq_len}")
        
        # Project latent to correct dimension
        condition = self.condition_proj(latent)
        
        if self.condition_method == "prefix":
            # Handle set_size dimension if present
            if len(input_ids.shape) == 3:  # [batch_size, set_size, seq_len]
                batch_size, set_size, seq_len = input_ids.shape
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            # Get input embeddings from the model
            input_embeds = self.model.backbone.embeddings.word_embeddings(input_ids)
            
            # Create prefix embedding from condition
            prefix_embeds = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Concatenate prefix with input embeddings
            combined_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
            
            # Generate position IDs for the combined sequence
            combined_seq_length = combined_embeds.size(1)
            position_ids = torch.arange(0, combined_seq_length, dtype=torch.long, device=combined_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Forward pass through the model using inputs_embeds
            hidden_states = self.model.backbone(
                position_ids=position_ids,
                inputs_embeds=combined_embeds
            )
            
            # Get logits from language model head
            logits = self.lm_head(hidden_states)
            
            # Remove prefix logits
            logits = logits[:, 1:, :]  # Remove the first position (prefix)
            
        elif self.condition_method == "additive":
            # Handle set_size dimension if present
            if len(input_ids.shape) == 3:  # [batch_size, set_size, seq_len]
                set_size, seq_len = input_ids.shape[1:]
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            # Process with the backbone first
            hidden_states = self.model.backbone(input_ids=input_ids)
            
            # Process condition
            processed_condition = self.condition_adapter(condition).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Broadcast processed_condition to all positions in the sequence
            processed_condition = processed_condition.expand(-1, hidden_states.size(1), -1)
            
            # Add condition to hidden states
            hidden_states = hidden_states + processed_condition
            
            # Project to vocabulary with language model head
            logits = self.lm_head(hidden_states)
        
        return logits


class HyenaDNAGenerator:
    """Generator class using conditioned HyenaDNA for the distribution embeddings framework."""
    
    def __init__(
        self,
        latent_dim=32,
        condition_dim=128,
        d_model=128,
        n_layer=6,
        # vocab_size=None,  # Will be determined from the tokenizer
        max_seq_len=512,
        condition_method="prefix",
        temperature=1.0,
    ):
        """
        Initialize the HyenaDNA generator.
        
        Args:
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            d_model: Model dimension
            n_layer: Number of layers
            vocab_size: If provided, sets the vocabulary size. Otherwise determined from tokenizer.
            max_seq_len: Maximum sequence length
            condition_method: How to condition the model ('prefix' or 'additive')
            temperature: Sampling temperature
        """
        # Initialize tokenizer first so we can determine the vocab size
        from datasets.hyena_tokenizer import CharacterTokenizer
        dna_vocab = ["A", "C", "G", "T", "N"]
        self.tokenizer = CharacterTokenizer(characters=dna_vocab, model_max_length=max_seq_len)

        # The tokenizer adds 7 special tokens plus the actual vocabulary
        # [CLS], [SEP], [MASK], [PAD], [UNK]
        vocab_size = len(dna_vocab) + 4
        
        # Now initialize the model with the correct vocabulary size
        self.model = ConditionedHyenaDNA(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            condition_method=condition_method
        )
        
        self.temperature = temperature
        self.max_seq_len = max_seq_len
    
    def loss(self, x, latent):
        """
        Calculate the loss for the generator.
        
        Args:
            x: Dictionary containing 'hyena_input_ids' and 'hyena_attention_mask'
            latent: Latent distribution embedding
        
        Returns:
            Negative log likelihood loss
        """
        input_ids = x['hyena_input_ids']
        attention_mask = x['hyena_attention_mask']
        
        # Debug shape
        batch_size, *rest_dims = input_ids.shape

        # Handle set_size dimension consistently if present
        if len(input_ids.shape) == 3:  # [batch_size, set_size, seq_len]
            batch_size, set_size, seq_len = input_ids.shape
            
            # Flatten batch and set dimensions for both inputs
            # This must match how forward() handles the dimensions
            flat_input_ids = input_ids.reshape(batch_size * set_size, seq_len)
            flat_attention_mask = attention_mask.reshape(batch_size * set_size, seq_len)
            
            # Repeat latent for each element in the set
            flat_latent = latent.unsqueeze(1).repeat(1, set_size, 1).reshape(batch_size * set_size, -1)
        else:
            # No set dimension, use as is
            flat_input_ids = input_ids
            flat_attention_mask = attention_mask
            flat_latent = latent
        
        # Get model outputs
        logits = self.model(flat_input_ids, flat_attention_mask, flat_latent)

        # For causal language modeling, shift the targets
        shift_logits = logits[:, :-1, :]  # Remove last position (shape: [batch*set, seq_len-1, vocab])
        shift_labels = flat_input_ids[:, 1:]   # Remove first position (shape: [batch*set, seq_len-1])
        shift_attention_mask = flat_attention_mask[:, 1:]  # Remove first position
        

        # Flatten the tensors to prepare for loss calculation
        # shift_logits: [batch*set, seq_len-1, vocab] -> [(batch*set)*(seq_len-1), vocab]
        # shift_labels: [batch*set, seq_len-1] -> [(batch*set)*(seq_len-1)]
        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        flat_labels = shift_labels.reshape(-1)
        flat_attention = shift_attention_mask.reshape(-1)
        

        # Only compute loss on tokens that are valid (attention mask = 1)
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        active_loss = flat_attention == 1
        active_logits = flat_logits[active_loss]
        active_labels = flat_labels[active_loss]

        loss = loss_fct(active_logits, active_labels)
        
        return loss
    
    def _generate_sequences(self, input_ids, attention_mask, latent, max_length):
        """
        Generate sequences using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            latent: Latent distribution embedding
            max_length: Maximum sequence length
        
        Returns:
            Generated sequences
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        cur_len = input_ids.size(1)
        
        # Keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        
        while cur_len < max_length:
            # Get next-token logits
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, latent)[:, -1, :]
            
            # Apply temperature
            logits = logits / self.temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Append next tokens to sequences
            next_tokens = next_tokens * unfinished_sequences + self.tokenizer.sep_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1)
            
            # Update which sequences are finished
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(self.tokenizer.sep_token_id).long())
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
                
            cur_len += 1
        
        return input_ids
    
    def sample(self, latent, num_samples=1, return_texts=False):
        """
        Sample DNA sequences from the generator.
        
        Args:
            latent: Latent distribution embedding
            num_samples: Number of samples to generate per latent
            return_texts: Whether to also return decoded texts
        
        Returns:
            Generated token IDs, and optionally decoded texts
        """
        self.model.eval()
        
        batch_size = latent.size(0)
        device = latent.device
        
        # Prepare for generation
        start_tokens = torch.full(
            (batch_size, 1),
            self.tokenizer.cls_token_id,  # Use BOS token ID = 2
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.ones_like(start_tokens)
        
        # Prepare to collect results
        all_generated_ids = []
        
        for _ in range(num_samples):
            # Generate DNA sequences
            with torch.no_grad():
                # Use a copy of the latent for each sample
                generated_ids = self._generate_sequences(
                    input_ids=start_tokens.clone(),
                    attention_mask=attention_mask.clone(),
                    latent=latent,
                    max_length=self.max_seq_len
                )
            
            all_generated_ids.append(generated_ids)
        
        # Make sure all tensors have the same length by padding to the maximum length
        max_len = max(ids.size(1) for ids in all_generated_ids)
        padded_ids = []
        for ids in all_generated_ids:
            if ids.size(1) < max_len:
                # Pad with EOS tokens
                padding = torch.full(
                    (batch_size, max_len - ids.size(1)),
                    self.tokenizer.sep_token_id,
                    dtype=ids.dtype,
                    device=device
                )
                ids = torch.cat([ids, padding], dim=1)
            padded_ids.append(ids)
        
        # Combine samples
        result = torch.stack(padded_ids, dim=1)  # [batch_size, num_samples, seq_len]
        
        if return_texts:
            # Decode texts
            all_texts = []
            for batch_idx in range(batch_size):
                batch_texts = []
                for sample_idx in range(num_samples):
                    ids = result[batch_idx, sample_idx]
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    # Make sure we have at least some text
                    if not text.strip():
                        text = "Generated sequence was empty."
                    batch_texts.append(text)
                all_texts.append(batch_texts)
            print("text shape", len(all_texts), len(all_texts[0]), len(all_texts[0][0]))
            
            return result, all_texts
        
        return result