import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()
 
# Download the MNIST Dataset
# dataset = "PLACEHOLDER"
dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = tensor_transform) #loads the MNIST dataset into loader
 
# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 32,
                                     shuffle = True)

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 784 ==> 9
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(28 * 28, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 9)
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 784
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(9, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 28 * 28),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


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
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.encoder_lin(x)
        print(x.shape)
        x = self.decoder_lin(x)
        print(x.shape)
        x = self.unflatten(x)
        print(x.shape)
        x = self.decoder_conv(x)
        print(x.shape)
        x = torch.sigmoid(x)
        print(x.shape)
        return x

# Model Initialization
model = AE()
 
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

epochs = 0 # takes 15 to 20 mins to execute, Initialize epoch = 1, for quick results
outputs = []
losses = []
for epoch in range(epochs):
	for (image, _) in loader:
            # Reshaping the image to (-1, 784)
            image = image.reshape(-1, 28*28)
        
            # Output of Autoencoder
            reconstructed = model(image)
        
            # Calculating the loss function
            loss = loss_function(reconstructed, image)
        
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Storing the losses in a list for plotting
            losses.append(loss)
            outputs.append((epochs, image, reconstructed))

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
#plt.plot(losses.detach().numpy()[-100:])

conv_model = ConvAE(512)
img = torch.Tensor(np.array(Image.open("images/canny/dirty/19.jpg")))
img = torch.unsqueeze(torch.unsqueeze(img, 0), 0)
out = conv_model(img).detach().numpy()
out = np.squeeze(out, (0,1))
plt.imshow(out)
plt.show()
"""for i, item in enumerate(image):
  # Reshape the array for plotting
  item = item.reshape(-1, 28, 28)
  plt.imshow(item[0])
  plt.show()

for i, item in enumerate(reconstructed):
  item = item.reshape(-1, 28, 28)
  with torch.no_grad():
    plt.imshow(item[0])
    plt.show()"""
