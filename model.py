import torch

class ConvAE(torch.nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = torch.nn.Sequential(
            #640x640x3
            torch.nn.Conv2d(1, 8, 64, stride=2, padding=1),
            torch.nn.ReLU(True),
            #290x290x8
            torch.nn.Conv2d(8, 16, 64, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            #115x115x16
            torch.nn.Conv2d(16, 32, 64, stride=2, padding=1),
            torch.nn.ReLU(True),
            #27x27x32
            torch.nn.Conv2d(32, 64, 16, stride=2, padding=1),
            torch.nn.ReLU(True)
            #7x7x64
        )

        self.flatten = torch.nn.Flatten(start_dim=1)
        
        self.encoder_lin = torch.nn.Sequential(
            torch.nn.Linear(3136, 2048),
            torch.nn.ReLU(True),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, encoded_space_dim)
        )

        self.decoder_lin = torch.nn.Sequential(
            torch.nn.Linear(encoded_space_dim, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(True),
            torch.nn.Linear(2048, 3136),
            torch.nn.ReLU(True)
        )

        self.unflatten = torch.nn.Unflatten(dim=1, 
        unflattened_size=(64, 7, 7))

        self.decoder_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 16, stride=2,
            padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 16, 64, stride=2, 
            padding=1, output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 8, 64, stride=2, 
            padding=1, output_padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(8, 1, 64, stride=2, 
            padding=1, output_padding=0)
        )
    
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x