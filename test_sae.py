import asyncio
from functools import partial

import orjson
import torch
from safetensors import safe_open  # New import for safetensors
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


raw_features = "raw_features/gpt2"
explanation_dir = "results/gpt2_explanations"
fuzz_dir = "results/gpt2_fuzz"


autoencoder_weights_path = "sae.safetensors"


class SparseAutoencoder(torch.nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(768, 512),  # Adjust the input size and layers
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)   # Latent space
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 768),  
            torch.nn.Sigmoid()          
        )
    
    def forward(self, x):
        # Forward pass through the encoder
        encoded = self.encoder(x)
        # Forward pass through the decoder
        decoded = self.decoder(encoded)
        return decoded

autoencoder = SparseAutoencoder()

def load_safetensors_weights(model, filepath):
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in model.state_dict():
                model.state_dict()[key].copy_(tensor)

load_safetensors_weights(autoencoder, autoencoder_weights_path)

# Put the model into evaluation mode if it's not being trained
autoencoder.eval()

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

        with torch.no_grad():  # Ensure no gradients are computed
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
        preprocess=lambda record: apply_autoencoder(preprocess(record)),  # Apply autoencoder to the data
        postprocess=explainer_postprocess,
    )

    ### Build Scorer pipe ###
    
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

    ### Build the pipeline ###
    
    pipeline = Pipeline(
        loader,
        explainer_pipe,
        scorer_pipe,
    )

    asyncio.run(pipeline.run(max_processes=5))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FeatureConfig, dest="feature")
    parser.add_arguments(ExperimentConfig, dest="experiment")

    args = parser.parse_args()

    main(args)
