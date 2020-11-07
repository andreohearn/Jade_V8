# JadeV8 - Training and Finetuning a Reformer Model

## Overview
The reformer model is a significant improvement over the transformer model in terms of speed, memory consumption, and storage usage. It enables massive context windows (transformer models such as GPT-2 only have context windows of 1024 tokens). This reformer model attempts to handle context windows of 33k tokens, with the goal of increasing the model's ability to handle long-term contexts in conversations.

The hope is that these improvements are worthwhile to retrain an entirely new model for others to finetune.

The reformer model does not come without its downsides. With the current release of TRAX and Reformer (1.3.6):
- There is no way to use large vocab sizes. 
- There is no way to increase the efficiency of evaluation sampling at the moment (it seems to be a issue with JITing the entire model into memory every time you need to sample). 
- There is no way to effectively synchronously run the model on multiple TPUs at once to speed up training in fewer steps. (seems to be an [issue with JAX](https://github.com/google/jax/issues/3004), no idea when it'll be fixed)

This repo is based on the example code that Google provided in their github repo for text generation, but with a few key differences:
- Instead of using a pretrained 310-token pretrained tokenizer, this repo trains its own 500-token tokenizer on the input text
- The context window is reduced from 500k to 32k to improve memory efficiency (you don't need such a massive context window for conversations)
- Instead of storing a copy of the text in each of the TPU cores with different padding, the text is automatically divided into feedable chunks and each TPU core receives a different sample from the chunks (still padded randomly)
- Dropout is increased from 0.05 to 0.2 (this is acceptable because with the size of the input dataset, the TPUs may be dealing with 2k different 33k token samples)

## Training
The majority of the model's hyperparameters are already set for you (and have been maxed out for GCP's 3-8 TPUs), but if you would like to modify it, it is in src/hyperparameters.py - put a issue request if you'd like me to explain what each parameter means.

### Creating and formatting data
The data format is simple:
```
Good morning! 
|aigenerationstart| Good morning! |endofgeneration|
How are you doing? 
|aigenerationstart| Doing good, thank you. |endofgeneration|
What is your last name? 
|aigenerationstart| As a robot, I don't have a family name. |br| Good afternoon! |endofgeneration|
```
The rules are as follows:
- Each line is in its own row
- Line breaks for users (if a user sends two messages individually after one another, or sends a multiline text) should be differentiated with `|br|`
- Anything the AI generates should start with `|aigenerationstart|` and and with `|endofgeneration|` (these are editable on [line 28 of train.py](https://github.com/JEF1056/Jade_V8/blob/c1a65390c5439d311bf6d51750bade928327934f/train.py#L29))
- Avoid any really long repeating character(s), such as `HIIIIIIIIIIIIIIIIII`, `wow!!!!!!!!!!!`, etc. as they can cause the AI to get stuck anytime it sees those characters
- The last line should always end with `|endofgeneration|`, e.g. should always end with the AI talking

> You may completely ignore these rules if you intend on using this repo for a different purpose, like generating long pieces of text. Just note that `evaluate.py` will no longer work as intended.

When complete, you may put the text in a folder (to support more than one text as input data)

The pretrained models were trained using these datasets:
- [Cornell Movie Dialogues]()
- [Anime Subtitles](https://www.kaggle.com/jef1056/anime-subtitles)
- [Google Datasets' Taskmaster](https://github.com/google-research-datasets/Taskmaster)
- [Google Datasets' Coached Conversational Preference Elicitation](https://research.google/tools/datasets/coached-conversational-preference-elicitation/)
- Some of Jess' texts and DMs XD
- Custom personality and standard greeting examples

### Configuring the script settings
> This section refers to the `train` part of `config.json`.

Copy `config.EXAMPLE.json` and rename it to `config.json`. Modify the parameters in the file, relative to Jade_V8:
| Setting | Default | Purpose |
|:-:|:-:|-|
| out-dir | train | Folder to output model, tokenizer, vocab, hyperparameters, and logs |
| tpu-ip | NaN | No default TPU IP is provided. Start up a TPU from GCP and paste its IP address here |
| data | data | Data folder (you created this in the last step) |

### Install requirements
`cd` into the `Jade_V8` directory and run 
```
sudo -H pip3 install requirements.txt
```

### Running the training loop
IN THEORY, the training loop should automatically be able to split, sample, and create its own segmentations, and fit the entire model into TPU memory properly as well. All you really need to do is run 
```
python3 train.py
```
The model will run 1000 epochs at 200 steps per epoch
> sorry this is [hardcoded](https://github.com/JEF1056/Jade_V8/blob/c1a65390c5439d311bf6d51750bade928327934f/train.py#L120), i'll fix it later...

### Monitoring training
All training logging is done by tensorboard and saved to the `out-dir` config in your `config.py`. As such, you can run tensorboard as such:
```
tensorboard --port 6060 --logdir [your-out-dir-here]
```
Since you (most likely) will be running this on a GCP vm, you can use [this great tutorial by Monte Fischer](https://www.montefischer.com/2020/02/20/tensorboard-with-gcp.html) to port the 6060 port on your vm to your local computer's port 6060, where you can then view tensorboard on your local machine at http://localhost:6060.

I will be including my tensorboard logs with the pretrained models when they are complete, so that you may compare your results to mine!

## Sampling and evaluation
Sampling the model is possible at every 200 steps, because the model evaluates itself at those intervals.
> sorry this is [hardcoded](https://github.com/JEF1056/Jade_V8/blob/c1a65390c5439d311bf6d51750bade928327934f/train.py#L120), i'll fix it later.... 

> I recommend using [rclone](https://rclone.org/) to copy the model form your GCP instance to your google drive/other storage service, as it may take an eternity to download it directly from google.

### Configuring the script settings
> This section refers to the `eval` part of `config.json`.

Modify the parameters in the file, relative to Jade_V8:
| Setting | Default | Purpose |
|:-:|:-:|-|
| model-dir | train | Folder that contains the output model, tokenizer, vocab, hyperparameters, and logs |

### Running the eval script
To start an interactive session with the model, run
```
python3 evaluate.py
```
