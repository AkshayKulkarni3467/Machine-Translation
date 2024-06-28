import torch
import torch.nn as nn
from model import Transformer,build_transformer
from config import get_config,get_weights_file_path,latest_weights_file_path
from train import get_model, get_ds,greedy_decode
import altair as alt
import pandas as pd
import numpy as np
import warnings

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
warnings.filterwarnings('ignore')

device = torch.device('cpu')

config = get_config()
tokenizer_src = Tokenizer.from_file('tokenizer_en.json')
tokenizer_target = Tokenizer.from_file('tokenizer_it.json')
model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_target.get_vocab_size()).to(device)

#Loading pretrained model
model_filename = latest_weights_file_path(config)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])
print('Using model : {}'.format(model_filename))

def input_sentence(input_senc,tokenizer_src=tokenizer_src,tokenizer_target=tokenizer_target,seq_len=350):
    encoder_input_tokens = input_senc
    encoder_input_tokens = tokenizer_src.encode(encoder_input_tokens).ids
    # encoder_mask = batch['encoder_mask'].to(device)

    enc_num_padding_tokens = seq_len - len(encoder_input_tokens) - 2 
  
    
    sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)
    encoder_input = torch.cat(
            [
                sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                eos_token,
                torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
    
    model_out = greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_target,config['seq_len'],device)
    predicted = tokenizer_target.decode(model_out.cpu().detach().numpy())
    return f'Translation: {predicted}'

sentence = input('Enter a sentence in English: ')

print(input_sentence(sentence))