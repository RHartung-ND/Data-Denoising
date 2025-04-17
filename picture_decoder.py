import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIG ===
ENCODER_PATH = "./encoder.pth"
DECODER_PATH = "./decoder.pth"
INPUT_DIR = "./output/spectrographs/train-noisy"
OUTPUT_DIR = "./output/spectrographs/denoised"
IMG_SIZE = (128, 128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATASET ===
class SpectrogramDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = []
        self.transform = transform

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".png"):
                    self.image_files.append(file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

# === MODELS ===
def get_encoder():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.ReLU(True),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(True),
    )

def get_decoder():
    return nn.Sequential(
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # -> 32x32
        nn.ReLU(True),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> 64x64
        nn.ReLU(True),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # -> 128x128
        nn.Sigmoid()
    )


# === MAIN ===
def denoise_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    dataset = SpectrogramDataset(INPUT_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    encoder = get_encoder().to(DEVICE)
    decoder = get_decoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    encoder.eval()
    decoder.eval()

    print("Denoising images...")
    with torch.no_grad():
        for img_tensor, filename in dataloader:
            img_tensor = img_tensor.to(DEVICE)
            encoded = encoder(img_tensor)
            reconstructed = decoder(encoded)

            reconstructed_img = reconstructed.squeeze().cpu().numpy()
            save_path = os.path.join(OUTPUT_DIR, filename[0])
            plt.imsave(save_path, reconstructed_img, cmap='gray')

    print(f"Denoised images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    denoise_images()