import argparse 
from torchvision import transforms
import torch 
import torchvision
from torch import nn
from model import VisionTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Training of Vision Transformer")
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--forward_expansion", type=int, default=4)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()




def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.img_size, antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)



    criterion = nn.CrossEntropyLoss()
    model = VisionTransformer(args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in tqdm(train_loader, desc="Training Epoch: " + str(epoch)):
            images, labels = images.to(args.device), labels.to(args.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            


if __name__ == "__main__":
    main()