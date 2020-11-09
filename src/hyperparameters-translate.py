import trax.layers
import trax.models
import trax.optimizers
import trax.data.inputs
import trax.supervised.trainer_lib

# Parameters that will vary between experiments:
# ==============================================================================
train.model = @trax.models.ReformerLM
# Parameters for multifactor:
# ==============================================================================
multifactor.constant = 0.088
multifactor.decay_factor = 0.5
multifactor.factors = 'constant * linear_warmup * rsqrt_decay'
multifactor.steps_per_cycle = 100000
multifactor.steps_per_decay = 20000
multifactor.warmup_steps = 8000

# Parameters for Adam:
# ==============================================================================
Adam.b1 = 0.9
Adam.b2 = 0.98
Adam.eps = 1e-09
Adam.weight_decay_rate = 1e-05

# Parameters for SelfAttention:
# ==============================================================================
trax.layers.SelfAttention.attention_dropout = 0.1
trax.layers.SelfAttention.chunk_len = 64
trax.layers.SelfAttention.n_chunks_before = 1
trax.layers.SelfAttention.n_parallel_heads = 1

# Parameters for ReformerLM:
# ==============================================================================
Reformer.d_model = 512
Reformer.d_ff = 2048
Reformer.dropout = 0.1
Reformer.ff_activation = @trax.layers.Relu
Reformer.max_len = 512
Reformer.mode = 'train'
Reformer.n_decoder_layers = 6
Reformer.n_encoder_layers = 6
Reformer.n_heads = 8
Reformer.input_vocab_size = 30