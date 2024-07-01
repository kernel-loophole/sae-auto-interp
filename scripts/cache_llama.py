from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.cache import FeatureCache
from sae_auto_interp.utils import get_samples
from nnsight import LanguageModel
import torch
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

# Load model and autoencoders
model = LanguageModel("meta-llama/Meta-Llama-3-8b", device_map="auto", dispatch=True,torch_dtype =torch.bfloat16)
print("Model loaded")
# Load autoencoders, submodule dict, and edits.
# Submodule dict is used in caching to save ae latents
# Edits are applied to the model
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    layers,
    "saved_autoencoders/Meta-Llama-3-8B/"
)
print("Autoencoders loaded")
# Set a default alteration on the model
with model.alter(" ", edits=edits):
    for layer_idx, _ in ae_dict.items():
        layer = model.model.layers[layer_idx]
        acts = layer.output[0]
        layer.ae(acts, hook=True)

# Get and sort samples
samples = get_samples()
samples = {
    int(layer) : features
    for layer, features in samples.items() 
    if int(layer) in ae_dict.keys()
}
print("Samples loaded")
# Cache and save features
cache = FeatureCache(model, submodule_dict)
cache.run()
print("Caching complete")
for layer in layers:
    feature_range = torch.tensor(samples[layer])
    cache.save_selected_features(feature_range, layer, save_dir="raw_features_llama")
print("Selected features saved")