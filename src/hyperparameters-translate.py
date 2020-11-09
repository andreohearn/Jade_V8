import trax.layers
import trax.models
import trax.optimizers
import trax.data.inputs
import trax.supervised.trainer_lib

# Parameters that will vary between experiments:
# ==============================================================================
train.model = @trax.models.ReformerLM
# Our model will have 6 layers, alternating between the LSH attention proposed
# in the Reformer paper and local attention within a certain context window.
n_layers = 12
attn_type = [
  @trax.layers.SelfAttention,
  @LSHSelfAttention,  
  @trax.layers.SelfAttention,
  @LSHSelfAttention,
  @trax.layers.SelfAttention,
  @LSHSelfAttention,
  ]
share_qk = False  # LSH attention ignores this flag and always shares q & k
n_heads = 8
attn_kv = 64
dropout = 0.1
n_tokens = 256

# Parameters for multifactor:
# ==============================================================================
multifactor.constant = 0.025
multifactor.factors = 'constant * linear_warmup * rsqrt_decay'
multifactor.steps_per_cycle = 100000
multifactor.warmup_steps = 8000

# Parameters for Adam:
# ==============================================================================
Adam.weight_decay_rate=0.0
Adam.b1 = 0.86
Adam.b2 = 0.92
Adam.eps = 1e-9

# Parameters for SelfAttention:
# ==============================================================================
trax.layers.SelfAttention.attention_dropout = 0.1
trax.layers.SelfAttention.chunk_len = 64
trax.layers.SelfAttention.n_chunks_before = 1
trax.layers.SelfAttention.n_parallel_heads = 1

# Parameters for LSHSelfAttention:
# ==============================================================================
LSHSelfAttention.attention_dropout = 0.0
LSHSelfAttention.chunk_len = 64
LSHSelfAttention.n_buckets = [64, 128]
LSHSelfAttention.n_chunks_after = 0
LSHSelfAttention.n_chunks_before = 1
LSHSelfAttention.n_hashes = 2
LSHSelfAttention.n_parallel_heads = 1
LSHSelfAttention.predict_drop_len = 128
LSHSelfAttention.predict_mem_len = 1024

# Parameters for ReformerLM:
# ==============================================================================
ReformerLM.attention_type = %attn_type
ReformerLM.d_attention_key = %attn_kv
ReformerLM.d_attention_value = %attn_kv
ReformerLM.d_model = 512
ReformerLM.d_ff = 1024
ReformerLM.dropout = %dropout
ReformerLM.ff_activation = @trax.layers.Relu
ReformerLM.max_len = %n_tokens
ReformerLM.mode = 'train'
ReformerLM.n_heads = %n_heads
ReformerLM.n_layers = %n_layers
ReformerLM.vocab_size = 20000
ReformerLM.axial_pos_shape = (16,16)
ReformerLM.d_axial_pos_embs= (128, 384)