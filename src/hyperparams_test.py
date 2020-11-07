import trax.models
import trax.optimizers
import trax.supervised.inputs
import trax.supervised.trainer_lib

# Parameters for _is_jit_init:
# ==============================================================================
_is_jit_init.value = True

# Parameters for _jit_predict_fn:
# ==============================================================================
_jit_predict_fn.jit = True

# Parameters for _jit_update_fn:
# ==============================================================================
_jit_update_fn.jit = True

# Parameters for Adam:
# ==============================================================================
Adam.b1 = 0.9
Adam.b2 = 0.98
Adam.eps = 1e-09
Adam.weight_decay_rate = 1e-05

# Parameters for backend:
# ==============================================================================
backend.name = 'jax'

# Parameters for batch_fn:
# ==============================================================================
batch_fn.batch_shuffle_size = 128
batch_fn.batch_size = None
batch_fn.batch_size_per_device = 256
batch_fn.bucket_length = 32
batch_fn.buckets = None
batch_fn.buckets_include_inputs_in_length = True
batch_fn.eval_batch_size = 64
batch_fn.max_eval_length = 512

# Parameters for EncDecAttention:
# ==============================================================================
EncDecAttention.masked = True
EncDecAttention.n_parallel_heads = None
EncDecAttention.use_python_loop = False
EncDecAttention.use_reference_code = False

# Parameters for inputs:
# ==============================================================================
inputs.data_dir = None
inputs.dataset_name = 't2t_translate_ende_wmt32k'
inputs.input_name = None

# Parameters for MultifactorSchedule:
# ==============================================================================
multifactor.constant = 0.088
multifactor.decay_factor = 0.5
multifactor.factors = 'constant * linear_warmup * rsqrt_decay'
multifactor.steps_per_cycle = 100000
multifactor.steps_per_decay = 20000
multifactor.warmup_steps = 8000

# Parameters for num_devices:
# ==============================================================================
num_devices.value = None

# Parameters for Reformer:
# ==============================================================================
Reformer.d_ff = 2048
Reformer.d_model = 512
Reformer.dropout = 0.1
Reformer.ff_activation = @trax.layers.Relu
Reformer.ff_dropout = 0.1
Reformer.input_vocab_size = 33300
Reformer.max_len = 2048
Reformer.n_decoder_layers = 6
Reformer.n_encoder_layers = 6
Reformer.n_heads = 8
Reformer.output_vocab_size = None

# Parameters for SelfAttention:
# ==============================================================================
SelfAttention.causal = False
SelfAttention.chunk_len = None
SelfAttention.masked = False
SelfAttention.n_chunks_after = 0
SelfAttention.n_chunks_before = 0
SelfAttention.n_parallel_heads = None
SelfAttention.predict_drop_len = 64
SelfAttention.predict_mem_len = 192
SelfAttention.share_qk = False
SelfAttention.use_python_loop = False
SelfAttention.use_reference_code = False

# Parameters for shuffle_and_batch_data:
# ==============================================================================
shuffle_and_batch_data.preprocess_fun = @trax.supervised.inputs.wmt_preprocess
shuffle_and_batch_data.shuffle_buffer_size = 1024

# Parameters for train:
# ==============================================================================
train.checkpoint_highest = None
train.checkpoint_lowest = None
train.checkpoints_at = None
train.eval_frequency = 1000
train.eval_steps = 10
train.has_weights = False
train.id_to_mask = 0
train.inputs = @trax.supervised.inputs.inputs
train.metrics = None
train.model = @trax.models.Reformer
train.nontrainable_param_map = None
train.optimizer = @trax.optimizers.Adam
train.random_seed = None
train.save_backward_graph = False
train.save_graphs = True
train.steps = 500000

# Parameters for train_and_eval_dataset:
# ==============================================================================
train_and_eval_dataset.eval_holdout_size = 0
train_and_eval_dataset.eval_shuffle_files = False
train_and_eval_dataset.train_shuffle_files = True

# Parameters for wmt_preprocess:
# ==============================================================================
wmt_preprocess.max_eval_length = 512
wmt_preprocess.max_length = 256