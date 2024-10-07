model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

submodule_dict = load_oai_autoencoders(
    model,
    layer_list,
    "sae.safetensors",
)