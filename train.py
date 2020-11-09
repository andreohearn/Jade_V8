import gin
import os
import jax
import trax
from trax.data import inputs
import io
import numpy as np
import jax.numpy as jnp
from scipy.special import softmax
import json
import sentencepiece as spm

#get config params
config_general=json.loads(open("config.json","r").read())["train"]

# The following is required to use TPU Driver as JAX's backend.
from jax.config import config
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://" + config_general["tpu-ip"] + ":8470"
print(config.FLAGS.jax_backend_target)
print(f'{jax.host_count()} available devices')
print(f'{jax.devices()} available cores')

output_dir = config_general["out-dir"]
try:os.makedirs(output_dir)
except:pass

"""
if not os.path.exists(os.path.join(config_general['out-dir'], 'bpe.model')):
    spm.SentencePieceTrainer.Train(input=[os.path.join(config_general["data"],filename) for filename in os.listdir(config_general["data"])], model_prefix=os.path.join(config_general['out-dir'], 'bpe'), model_type='bpe', vocab_size=config_general["vocab_size"], unk_id=1,bos_id=3, eos_piece="|dividertoken|", user_defined_symbols=["|dividertoken|", "|br|"])

TOKENIZER = spm.SentencePieceProcessor()
TOKENIZER.Load(os.path.join(config_general['out-dir'], 'bpe.model'))
print()
print(TOKENIZER.Encode("1234 Hello this is a 😊!!!! .. .\n|dividertoken| oh, that's unfortunate |dividertoken|\nI'm so |br| flabbergasted!  |dividertoken| oop".replace("\n"," ")))
print(TOKENIZER.EncodeAsPieces("1234 Hello this is a 😊!!!! .. .\n|dividertoken| oh, that's unfortunate |dividertoken|\nI'm so |br| flabbergasted!  |dividertoken| oop".replace("\n"," ")))

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Tokenize
print("\nBEGIN SAMPLING")
if config_general["mode"] == "conversation":
    IDS=[]
    for training_data in os.listdir(config_general["data"]):
        with io.open(os.path.join(config_general["data"],training_data), mode="r", encoding="utf-8") as f:
            IDS+=" ".join([str(token) for token in TOKENIZER.Encode(f.read().strip().replace("\n"," "))]).split(f' {2} ')
    IDS=[[int(toint) for toint in token.split(" ")]+[2] for token in IDS]
    n=450
    IDS=[IDS[i * n:(i + 1) * n] for i in range((len(IDS) + n - 1) // n )]
    DE_SPLIT=[]
    for sequence in IDS:
        DE_SPLIT.append([j for i in sequence for j in i])
    IDS=DE_SPLIT
elif config_general["mode"] == "translate":
    for training_data in os.listdir(config_general["data"]):
        IDS=[]
        with io.open(os.path.join(config_general["data"],training_data), mode="r", encoding="utf-8") as f:
            IDS+=[TOKENIZER.Encode(split_value) for split_value in f.read().split("\n")]
print(IDS[0][-300:])
print(TOKENIZER.Decode(IDS[0][-300:]))
MAX_DIMENSIONS=len(max(IDS,key=len))
print(f"{MAX_DIMENSIONS} is the longest array subset")
print(f"{len(IDS)} training sequences")

#a test check to see how many of our samples are forcefully removed due to being too long
for SELECT in range(0,len(IDS)):
    CONTEXT_IDS=IDS[SELECT]
    if (config_general["size"])- len(CONTEXT_IDS) <= 0:
        print(f"Index {SELECT} is out of bounds by {(config_general['size'])- len(CONTEXT_IDS)}")
print("These samples will not be included.")

# Set up the data pipeline:
# What this does is take the big list of list question/answer tokenized arrays and concatanate the data split into fittable sizes
# These are looped through, in order, to generate next outputs with context.

def gen_inputs(n_devices):
    while True:
        inputs = []
        mask = []
        #sometimes the pad amount is :((, so skip those; the data will be made up later
        for i in range(n_devices):
            PAD_AMOUNT=0
            while PAD_AMOUNT <= 0:
                current_sample = np.random.choice(len(IDS)-1, 1)[0]
                SELECT=[2]+IDS[current_sample]
                PAD_AMOUNT = (config_general["size"]) - len(SELECT)
            SELECT=np.asarray(SELECT, dtype=np.int32)
            pad_amount = np.random.choice(PAD_AMOUNT, 1)[0]
            inputs.append(np.pad(SELECT, (pad_amount, PAD_AMOUNT - pad_amount),
                                    mode='constant'))
            mask.append(np.pad(np.ones_like(SELECT, dtype=np.float32),
                                (pad_amount, PAD_AMOUNT - pad_amount),
                                mode='constant'))
        inputs = np.stack(inputs)
        mask = np.stack(mask)
        yield (inputs, inputs, mask)
"""

def gen_inputs(batch_size, length=255):
  while True:
    random_ints = m = np.random.randint(1, 31, (batch_size,length))
    source = random_ints

    target = np.flip(source, 1)

    zero = np.zeros([batch_size, 1], np.int32)
    x = np.concatenate([zero, source, zero, target], axis=1)

    loss_weights = np.concatenate([np.zeros((batch_size, length+2)),
                                    np.ones((batch_size, length))], axis=1)
    yield (x, x, loss_weights)  # Here inputs and targets are the same.

#test it's working on sample index 10
print("(device count, tokens per device) = ",
    next(gen_inputs(trax.fastmath.device_count()))[0].shape)

# Configure hyperparameters.
print(f"Using {config_general['mode']} params")
if config_general["mode"] == "conversation":
    hyperparams=open("src/hyperparameters.py","r").read()
elif config_general["mode"] == "translate":
    hyperparams=open("src/hyperparameters-translate.py","r").read()
gin.parse_config(hyperparams)
with open(os.path.join(output_dir,"hyperparameters.py"), "w") as f:
    f.write(hyperparams)

# Set up a Trainer.
print("SETTING UP TRAINER/MODEL (depending on your layer sizes this might take a while)")
if config_general["mode"] == "conversation":
    trainer = trax.supervised.Trainer(
        model=trax.models.ReformerLM,
        loss_fn=trax.layers.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam,
        lr_schedule=trax.lr.multifactor(),
        inputs=trax.data.inputs.Inputs(gen_inputs),
        output_dir=output_dir)
elif config_general["mode"] == "translate":
    trainer = trax.supervised.Trainer(
        model=trax.models.Reformer,
        loss_fn=trax.layers.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam,
        lr_schedule=trax.lr.multifactor(),
        inputs=trax.data.inputs.Inputs(gen_inputs),
        output_dir=output_dir)
print("DONE, BEGIN TRAINING")

# Run one training step, to make sure the model fits in memory.
# The first time trainer.train_epoch is called, it will JIT the entire network
# architecture, which takes around 2 minutes. The JIT-compiled model is saved
# so subsequent runs will be much faster than the first.
for i in range(config_general["epochs"]):
    print(f'Epoch {i} starting')
    trainer.train_epoch(n_steps=config_general["steps"], n_eval_steps=1)