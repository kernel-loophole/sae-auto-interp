import asyncio
from functools import partial

import orjson
import torch
from safetensors.torch import load_file  # Import for safetensors with PyTorch
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Local
from sae_auto_interp.config import FeatureConfig, ExperimentConfig
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.features import FeatureDataset, sample, pool_max_activation_windows
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)

### Set directories ###
raw_features = "raw_features/gpt2"
explanation_dir = "results/gpt2_explanations"
fuzz_dir = "results/gpt2_fuzz"

autoencoder_weights_path = "sae.safetensors"  # Make sure this is correct


### Define the SparseAutoencoder Model ###
class SparseAutoencoder(torch.nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()

        # Adjusted Encoder dimensions based on the error message
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 32768),  # Adjust the input size and layers
            torch.nn.ReLU(),
            torch.nn.Linear(32768, 32768),
            torch.nn.ReLU(),
            torch.nn.Linear(32768, 32768)   # Latent space based on saved model
        )
        
        # Adjusted Decoder dimensions to match the weight sizes in the checkpoint
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32768, 1024),  # Match the size from the saved checkpoint
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),  
            torch.nn.Sigmoid()          
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Initialize the autoencoder model
autoencoder = SparseAutoencoder()

### Load Weights from Safetensors and Map Keys ###
def load_safetensors_weights(model, filepath):
    state_dict = load_file(filepath)
    
    # Create a new state_dict with renamed keys for the model
    new_state_dict = {}
    for key in state_dict.keys():
        # Mapping safetensors keys to model's expected keys
        if key == "encoder.weight":
            new_state_dict["encoder.0.weight"] = state_dict[key]
        elif key == "encoder.bias":
            new_state_dict["encoder.0.bias"] = state_dict[key]
        elif key == "W_dec":
            new_state_dict["decoder.0.weight"] = state_dict[key]
        elif key == "b_dec":
            new_state_dict["decoder.0.bias"] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    
    # Load the renamed state_dict into the model, using strict=False to avoid issues with missing/unexpected keys
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

# Load the weights into the autoencoder model
load_safetensors_weights(autoencoder, autoencoder_weights_path)

# Set the model to evaluation mode
autoencoder.eval()

### Main Function to Run the Pipeline ###
def main(args):

    ### Load tokens ###
    tokenizer = load_tokenizer("gpt2")
    tokens = load_tokenized_data(
        args.feature.example_ctx_len,
        tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    modules = [f".transformer.h.{i}" for i in range(0, 12, 2)]
    features = {mod: torch.arange(50) for mod in modules}

    dataset = FeatureDataset(
        raw_dir=raw_features,
        cfg=args.feature,
        modules=modules,
        features=features,
    )

    loader = partial(
        dataset.load,
        constructor=partial(
            pool_max_activation_windows, tokens=tokens, cfg=args.feature
        ),
        sampler=partial(sample, cfg=args.experiment)
    )

    client = Local("casperhansen/llama-3-70b-instruct-awq")

    ### Build the Explainer Pipe ###
    
    def preprocess(record):
        test = []
        extra_examples = []
        for examples in record.test:
            test.append(examples[:5])
            extra_examples.extend(examples[5:])
        record.test = test
        record.extra_examples = extra_examples
        return record

    def explainer_postprocess(result):
        with open(f"{explanation_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))
        return result

    def apply_autoencoder(features):
        with torch.no_grad():
            encoded_features = autoencoder(features)
        return encoded_features

    explainer_pipe = process_wrapper(
        SimpleExplainer(
            client,
            tokenizer=tokenizer,
            activations=True,
            max_tokens=500,
            temperature=0.0,
        ),
        preprocess=lambda record: apply_autoencoder(preprocess(record)),
        postprocess=explainer_postprocess,
    )

    ### Build the Scorer Pipe ###
    
    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        return record

    def scorer_postprocess(result, score_dir):
        with open(f"{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = Pipe(
        process_wrapper(
            FuzzingScorer(
                client,
                tokenizer=tokenizer,
                verbose=True,
                max_tokens=50,
                temperature=0.0,
                batch_size=10,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=fuzz_dir),
        ),
    )

    ### Build and Run the Pipeline ###
    pipeline = Pipeline(loader, explainer_pipe, scorer_pipe)
    asyncio.run(pipeline.run(max_processes=5))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FeatureConfig, dest="feature")
    parser.add_arguments(ExperimentConfig, dest="experiment")
    args = parser.parse_args()
    main(args)
