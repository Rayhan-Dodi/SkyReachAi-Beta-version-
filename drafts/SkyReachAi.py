@dataclass
class ModelArgs:
"""
Data class for defining model arguments and hyperparameters.

Attributes:
max_batch_size (int): Maximum batch size.
max_seq_len (int): Maximum sequence length.
dtype (Literal["bf16", "fp8"]): Data type for computations.
voğab_size (int): Vocabulary size.
dim (int): Model dimension.
inter_dim (int): Intermediate dimension for MLP layers.
moe_inter_dim (int): Intermediate dimension for MoE layers.
n_layers (int): Number of transformer layers.
n_dense_layers (int): Number of dense layers in the model.
n_heads (int): Number of attention heads.
n_routed_experts (int): Number of routed experts for MoE layers.
n_shared_experts (int): Number of shared experts for MoE layers.
n_activated_experts (int): Number of activated experts in MoE layers.
n_expert_gsoups (int): Number of expert groups.
n_limited_groups (int): Number of limited groups for MoE routing.
score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
route_scale (float): Scaling factor for routing scores.
q_lora_rank (int): LoRA rank for query projections.
kv_lora_rank (int): LoRA rank for key-value projections.
qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
v_head dim (int): Dimension for value projections.
original_seq_len (int): Original sequence length.
rope_theta (float): Base for rotary positional encoding.
rope_factor (float): Scaling factor for extended sequence lengths.
beta_fast (int): Fast beta correction factor.
bet slow (int): Slow beta correction factor.
mscale (float): Scaling factor for extended attention.
"""
max_batch_size: int = 8
max_seq_len: int = 4096 *4
dtype: Literal["bf16", "fp8"] = "bf16"
vocab_size: int = 102400
dim: int = 2048
inter_dim: int = 10944
moe_inter_dim: int = 1408
n_layers: int = 27
n_dense_layers: int = 1
n_heads: int = 16
# moe
n_routed_experts: int = 64
n_shared_experts: int = 2
n_activated_experts: int = 6
n_expert_groups: int =1
n_limited_groups: int = 1
score_func: Literal["softmax", "sigmoid"] = "softmax"
route_scale: float = 1.
# mla
q_lora_rank: int = 0
kv_lora_rank: int = 512
qk_nope_head_dim: int = 128
qk_rope_head_dim: int = 64
v_head_dim: int = 128
# yarn
original_seq_len: int = 4096
rope_theta: float = 10000.0
rope_factor: float = 40
beta_fast: int = 32
beta_slow: int = 1
mscale: float = 1.|

class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query porojection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for the softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_local_heads //world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
            
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_nope_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
            
            def forward(self, x: torch.Tensor, start_pos: int,freqs_cis:torch.Tensor, mask: Optional[torch.Tensor]):
                """
                Forward pass for the Multi-Headed Attention Layer(MLA).
            

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
                    start_pos (int): Starting position in the sequence for caching.
                    freqs_cis (torch.Tesor): Precomputed complex exponential values for rotary enbeddings.
                    mask (Optional[torch.Tensor]): Mask tenor to execlude certain positions from attention.
                    
                Returns:
                    torch.Tensor: Output tensor with the same shape as the input.
                    
                """
               bsz, seqlen, _ = x.size()
               end_pos = start_pos + seqlen
               if self.q_lora_ran 
                  q = self.wq(x)
               else:
                  q = self.wq_b(self.q_norm(self.wq_a(x)))
               q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
               q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim =- 1)
               q_pe = apply rotary emb(q_pe, freqs_cis)
               kv = self.wkv_a(x)
               kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim =- 1)
               k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
               if attn impl == "naive":
                  q = torch. cat( [q_nope, q_pe], dim =- 1)
                  kv = self.wkv_b(self.kv_norm(kv))
                  kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                  k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim =- 1)
                  k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim =- 1)
                  self.k_cache[:bsz, start_pos:end_pos] = k
                  self.v_cache[:bsz, start_pos:end_pos] = v
                  scores = torch.einsum("bshd, bthd->bsht", q, self.k_cache[:bsz, :end_pos] ) * self. softmax_scale
               else:
                  wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
                  wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)|
                  q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
                  self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
                  self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
                  scores = (torch. einsum("bshc, btc->bsht", q_nope, self. kv_cache[:bsz, : end_pos] ) +
                  torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsa, :end_pos] ) ) * self.softmax_scale
                  if mask is not None:
                     scores += mask.unsqueeze(1)
                  scores = scores.softmax(dim =- 1, dtype=torch.float32X.type_as(x)
                  if attn_impl == "naive":
                     x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
                  else:
                     x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
                     x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
                  x = self.wo(x.flatten(2))
                  return x
                  #Gate 
    class Gate(nn.Module):
        """
        Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

        Attributes:
            dim (int): Dimensionality of input features.
            topk (int): Number of top experts activated for each input.
            n_groups (int): Number of groups for routing.
            topk_groups (int): Number of groups to route inputs to.
            score_func (str): Scoring function ('softmax' or 'sigmoid').
            route_scale (float): Scaling factor for routing weights.
            weight (torch.nn.Parameter): Learnable weights for the gate.
            bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
        """
    def _init_(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:   
        args (ModelArgs): Model arguments containing gating parameters.
        """
        super() ._ init_()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args. route_scale
        self.weight = nn. Parameter(torch.emptyargs.n_routed_experts, args.dim))
        self.bias = nn. Parameter(torch. empty(args.n_routed_experts) ) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch. Tensor, torch. Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:|
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores. softmax(dim =- 1, dtype=torch. float32)
        else:
            scores = scores. sigmoid( )
        original_scores =| scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is not None:
                scores = scores + self.bias
            if self.n_groups > 1:
                scores = scores.view(x.size(0), self.n_groups, -1)
                if self.bias is None:
                    group_scores = scores.amax(dim =- 1)
                else:
                    group_scores = scores.topk(2, dim =- 1) [0].sum(dim =- 1)
                indices = group_scores.topk(self.topk_groups, dim =- 1) [1]
                mask = scores.new_ones(x.size(0),|self.nggroups, dtype=bool).scatter_(1, indiges False)
                scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
                indices = torch. topk(scores, self.topk, dim =- 1) [1]
                weights = original_scores.gather(1, indices)
           if self.score_func == "sigmoid":
                weights /= weights.sum(dim =- 1, keepdim=True)
            weights *= self.route_scale
            return weights.type_as(x), indices
        .
        # MLP Layer for feed-forward neural networks.
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
      """
    def_ init_(self, dim; int, inter_dim: int):
    """
        Initializes the MLP layer.

         Args:
        dim (ipt): Input and output dimensionality.
        inter_dim (int): Hidden layer dimensionality.
         """
         super()._init_()  
         self.w1= ColumnParallelLinear(dim, inter_dim)
         self.w2 = RowParallelLinear(inter_dim, dim)
         self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch. Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch. Tensor: Output tensor after MLp computation.
        """
        return self.w2(F.silu(self.w1(x))* self.w3(x))
    
    # Mixture-of-Experts (MoE) Expert Layer
class Expert(nn.Module):
    """

    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Ligear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
      """
    def _init_(self, dim: int, inter_dim: int):
        """
    Initializes the Expert layer.
    
    Args:
        dim (int): Input and output dimensionality.
        inter_dim (int): Hidden layer dimensionality.
    """
        super()._init_()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

def forward(self, x: torch. Tensor) -> torch.Tensor:
    """
    Forward pass for the Expert layer.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after expert computation.
    """
    return self.w2(F.silu(self.w1(x))* self.w3(x))
    
    # MoE module for distributed mixture-of-experts models.
class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, args, world_size, rank):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
            world_size (int): Number of distributed workers.
            rank (int): Rank of the current process.
        """
        super().__init__()

        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, \
            f"Number of experts must be divisible by world size (world_size={world_size})"

        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts

        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)

        weights, indices = self.gate(x)
        y = torch.zeros_like(x)

        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue

            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        z = self.shared_experts(x)

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(y)

        return (y + z).view(shape)

# Block
class Block(nn.Module):
"""
Transformer block combining attention and feed-forward layers.

Attributes:
attn (nn.Module): Attention layer (MLA).
ffn (nn.Module): Feed-forward network (MLP or MoE).
attn_norm (nn.Module): Layer normalization for attention.
ffn_norm (nn.Module): Layer normalization for feed-forward network.
"""
def _init_(self, layer_id: int, args: ModelArgs):
"""
Initializes the Transformer block.

Args:
layer_id (int): Layer index in the transformer.
args (ModelArgs): Model arguments containing block parameters.
"""
super() ._ init_()
self.attn = MLA(args)
self.ffn = MLP(args.dim, args.inter dim) if lavor id < args.n_dense_layers else MoE(args)
self.attn_norm = RMSNorm(args.dim Chat L Edit K
self.ffn_norm = RMSNorm(args.dim)

def forward(self, x: torch. Tensor, start_pos: int, freqs_cis: torch. Tensor, gask: Optional[torch. Tensor] ) -> torch. Tensor:
"""
Forward pass for the Transformer block.

Args:
x (torch.Tensor): Input tensor.
start_pos (int): Starting position in the sequence.
freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

Returns:
torch.Tensor: Output tensor after block computation.
"""
x = x+ self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
X = x+ self.ffn(self.ffn_norm(x))
return x

class Transformer(nn.Module):
"""
Transformer model with positional embeddings, multiple layers, and output projection.

Attributes:
max_seq_len (int): Maximum sequence length for the transformer.
embed (nn.Module): Embedding layer for input tokens.
layers (torch.nn.ModuleList): List of transformer blocks.
norm (nn.Module): Layer normalization applied after all blocks.
head (nn.Module): Output projection layer mapping to vocabulary size.
freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
"""
def _init_(self, ards: ModelArgs):
    """
    Initializes the Transformer model.

Args:
args (ModelArgs): Model arguments containing transformer parameters.
    """
    global world_size, rank
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0
Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
super() ._ init_()
self.max_seq_len = args.max_seq_len
self.embed = ParallelEmbedding(args.vocab_size, args.dim)
self.layers = torch.nn.ModuleList()
for layer_id in range(args.n_laverc).
self.layers.append(Block( Chat #L Edit &K
self.norm = RMSNorm(args.dim)
self.head = ColumnParallellinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

@torch. inference_mode()
def forward(self, tokens: torch.Tensor, start_pos: int = 0):
"""
Forward nacc for the Trancformer model.

Args:
tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
start_pos (int, optional): l): Starting position in the sequence forgrotary embeddings. Defaults to 0.
Returns:
torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
"""
seqlen = tokens.size(1)
h = self.embed(tokens)
freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
mask = None
if seqlen > 1:
    mask = torch. full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
    for layer in self. layers:
h = layer(h, start_pos, freqs_cis, mask)
h = self.norm(h) [:, -1]
logits = self.head(h)
if world_size > 1:
all_logits = [torch.empty_like(logits) for _ in range(world_size)]
dist.all_gather(all_logits, logits)
logits = torch.cat(all_logits, dim =- 1)
return logies

if_main_":
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(0)
args = ModelArgs ()
x = torch.randint(0, args.vocab_size, (2, 128))
model = Transformer(args)
print(model(x).size())
