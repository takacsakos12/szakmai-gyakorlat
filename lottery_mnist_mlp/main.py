import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# Beállítások
# --------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.002


# --------------------------------------------------
# Eszköz kiválasztása: GPU, ha elérhető, különben CPU
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Használt eszköz:", device)


# --------------------------------------------------
# MNIST adathalmaz előkészítése
# --------------------------------------------------
# ToTensor:
#   A képet PyTorch tenzorrá alakítja.
#
# Normalize:
#   Normalizálja a pixelértékeket.
#   Az MNIST ismert átlaga: 0.1307
#   Az MNIST ismert szórása: 0.3081
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))#
])


# --------------------------------------------------
# Tanító és teszt adathalmaz betöltése
# --------------------------------------------------

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# --------------------------------------------------
# DataLoaderek létrehozása
# --------------------------------------------------
# A DataLoader batchekre bontja az adatokat.
# BATCH_SIZE = 128 esetén egyszerre 128 képet dolgoz fel.
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# MLP modell definiálása
# --------------------------------------------------
# MNIST kép mérete: 28 x 28 = 784 pixel
#
# Architektúra:
#   784 -> 300 -> 100 -> 10
#
# 10 kimenet:
#   0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# --------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),             # 28x28 kép -> 784 hosszú vektor
            nn.Linear(28 * 28, 300),  # 784 bemenet -> 300 neuron
            nn.ReLU(),                # aktivációs függvény
            nn.Linear(300, 100),      # 300 neuron -> 100 neuron
            nn.ReLU(),                # aktivációs függvény
            nn.Linear(100, 10)        # 100 neuron -> 10 kimenet
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# Tanító függvény
# --------------------------------------------------

def train(model, train_loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Előző batch gradienseinek törlése
        optimizer.zero_grad()

        # Forward pass: képek átmennek a modellen
        outputs = model(images)

        # Loss / hiba kiszámítása
        loss = criterion(outputs, labels)

        # Backpropagation: gradiensek kiszámítása
        loss.backward()

        # Súlyok frissítése
        optimizer.step()

        # Loss gyűjtése
        total_loss += loss.item()

        # Predikció meghatározása
        _, predicted = torch.max(outputs, 1)

        # Pontosság számítása
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return average_loss, accuracy


# --------------------------------------------------
# Tesztelő / kiértékelő függvény
# --------------------------------------------------

def evaluate(model, test_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Tesztelésnél nem kell gradienst számolni
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Loss számítása
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Predikció
            _, predicted = torch.max(outputs, 1)

            # Pontosság számítása
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return average_loss, accuracy


# --------------------------------------------------
# Főprogram
# --------------------------------------------------

def main():
    # Modell létrehozása
    model = MLP().to(device)

    # Loss függvény
    criterion = nn.CrossEntropyLoss()

    # Optimalizáló
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nBaseline MLP modell tanítása MNIST adathalmazon...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochok száma: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 70)

    start_time = time.time()

    # Epoch ciklus
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test acc: {test_acc:.2f}%"
        )

    end_time = time.time()

    print("-" * 70)
    print(f"Tanítási idő: {end_time - start_time:.2f} másodperc")

    final_loss, final_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    print(f"Végső baseline tesztpontosság: {final_acc:.2f}%")

    # Modell mentése
    torch.save(model.state_dict(), "baseline_mlp_mnist.pth")
    print("Baseline modell elmentve: baseline_mlp_mnist.pth")


# --------------------------------------------------
# Program indítása
# --------------------------------------------------

if __name__ == "__main__":
    main()