import gin
import json
import trax
import os
import jax
import numpy as np
import time
import sentencepiece as spm

print(f'{jax.host_count()} available devices')
print(f'{jax.devices()} available cores')

config_general=json.loads(open("config.json","r").read())

gin.parse_config(open(os.path.join(config_general["eval"]["model-dir"],"hyperparameters.py"),"r").read())
gin.parse_config("""LSHSelfAttention.n_hashes = 8""")

model = trax.models.ReformerLM(mode='predict')
model.init_from_file(os.path.join(config_general["eval"]["model-dir"],'model.pkl.gz'), weights_only=True)

TOKENIZER = spm.SentencePieceProcessor()
TOKENIZER.Load(os.path.join(config_general["eval"]["model-dir"],"bpe.model"))

def generate(inp):
    input_encoded=np.asarray([[2] + TOKENIZER.Encode(inp) + [1]])
    # Sample from ReformerLM
    output_token_ids = trax.supervised.decoding.autoregressive_sample_stream(model, input_encoded, temperature=0.9, accelerate=True)
    result, start = [], time.time()
    while True:
        result.append(next(output_token_ids).tolist()[0])
        if result[len(result)-1] == 2 or len(result)>=100:
            break
    print(TOKENIZER.Decode(result))
    print()
    print(input_encoded.tolist()[0])
    print(result)
    elapsed=time.time()-start
    print(elapsed)
    return {"input": inp, "output": TOKENIZER.Decode(result), "input_encoded":input_encoded.tolist()[0], "output_encoded":result, "time":elapsed}
    
if __name__ == "__main__":
    while True:
        inp= input("> ")
        generate(inp)