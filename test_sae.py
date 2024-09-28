class SparseAutoencoder(torch.nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()

        # Encoder dimensions based on the error message
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 32768),  # Encoder layers are fine
            torch.nn.ReLU(),
            torch.nn.Linear(32768, 32768),
            torch.nn.ReLU(),
            torch.nn.Linear(32768, 32768)   # Latent space based on saved model
        )
        
        # Decoder dimensions adjusted to match the checkpoint
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32768, 32768),  # Use 32768 as both input and output size
            torch.nn.ReLU(),
            torch.nn.Linear(32768, 1024),  # Match the checkpoint's shape
            torch.nn.Sigmoid()          
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
