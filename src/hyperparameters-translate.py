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

# Parameters for ReformerLM:
# ==============================================================================
ReformerLM.d_model = 32
ReformerLM.d_ff = 128
ReformerLM.dropout = 0.1
ReformerLM.ff_activation = @trax.layers.Relu
ReformerLM.max_len = 512
ReformerLM.mode = 'train'
ReformerLM.n_layers = 2
ReformerLM.n_heads = 1
ReformerLM.vocab_size = 32