from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FilenameLabeledDataset(Dataset):
    """Loads real images from input/ and labels by filename prefix."""

    def __init__(self, root: str = "input", resize_hw=(288, 384)):
        self.root = Path(root)
        all_paths = sorted(
            [p for p in self.root.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        
        # Filter out corrupted images
        self.paths = []
        for p in all_paths:
            try:
                with Image.open(p) as img:
                    img.verify()
                self.paths.append(p)
            except Exception:
                print(f"Skipping corrupted image: {p.name}")
        
        if not self.paths:
            raise ValueError(f"No valid images found in {self.root.resolve()}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_hw),  # force 4:3 aspect (288x384)
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def label_from_name(path: Path) -> int:
        name = path.name.lower()
        if name.startswith("roblox"):
            return 0
        if name.startswith("screenshot"):
            return 1
        return 2  # default to Other class if unmatched

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            label = self.label_from_name(path)
            return image, label
        except Exception as e:
            # If image fails to load, return a black placeholder
            print(f"Error loading {path.name}: {e}")
            image = torch.zeros(3, 288, 384)
            label = self.label_from_name(path)
            return image, label


def build_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )

def train(num_epochs: int = 50, batch_size: int = 16, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3  # 0: Roblox* files, 1: Screenshot* files, 2: Other

    dataset = FilenameLabeledDataset("input")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Quick class count so you know what you have
    counts = [0, 0, 0]
    for path in dataset.paths:
        counts[dataset.label_from_name(path)] += 1
    print(f"Found {len(dataset)} images -> class0(Roblox): {counts[0]}, class1(Screenshot): {counts[1]}, class2(Other): {counts[2]}")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("Training complete. Model saved to model.pt")


if __name__ == "__main__":
    train()
