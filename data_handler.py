from utils import *

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

# Datasets
tr_dataset = datasets.ImageFolder(root='data/DataSet/train', transform=transform)
te_dataset = datasets.ImageFolder(root='data/DataSet/test', transform=transform)
va_dataset = datasets.ImageFolder(root='data/DataSet/val', transform=transform)

# Dataloaders
batch_size = 8
tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
te_loader = DataLoader(te_dataset, batch_size=batch_size, shuffle=True)
va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    print(tr_dataset.class_to_idx) # {'rest': 0, 'wall': 1}