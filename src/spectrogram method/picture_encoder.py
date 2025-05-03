import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DATA_DIR = None
IMG_SIZE = (128, 128)  # Resize all spectrograms to this size
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    with open("src/config.txt", "r") as config:
        print("----------------------------------------------------------------")
        line = config.readline() # input_dir_clean
        line = config.readline() # input_dir_clean
        line = config.readline() # input_dir_clean
        line = config.readline() # input_dir_clean
        line = config.readline() # input_dir_clean
        # -------------------------------------------------
        line = config.readline() # input_dir_clean        
        line = config.readline() # noisy_dir
        args = line.strip().split("=")
        if len(args) > 1:
            DATA_DIR = str(args[1].strip())
            print(f"Using the following training directory: {DATA_DIR}")
        print("----------------------------------------------------------------")
except TypeError:
    print("Please use correct data paths or epoch numbers")

class SpectrogramDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".png"):
                    self.image_files.append(file)

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> 64x64
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 16x16
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # -> 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> 64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # -> 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    dataset = SpectrogramDataset(DATA_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images in tqdm(dataloader):
            images = images.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

    # Save full model and separately encoder/decoder
    torch.save(model.state_dict(), "src/spectrogram method/autoencoder.pth")
    torch.save(model.encoder.state_dict(), "src/spectrogram method/encoder.pth")
    torch.save(model.decoder.state_dict(), "src/spectrogram method/decoder.pth")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train()
