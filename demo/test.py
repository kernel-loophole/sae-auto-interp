from functools import partial
from safetensors import safe_open
import torch
from nnsight import LanguageModel
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.autoencoders.OpenAI import Autoencoder

tensors = {}
with safe_open("sae.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

path = "sae.safetensors" # Change this line to your weights location.
state_dict = tensors
ae = Autoencoder.from_state_dict(state_dict=state_dict)
ae.to("cuda:0")

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)