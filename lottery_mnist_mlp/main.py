import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.002
SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Használt eszköz:", device)


transform = transforms.Compose([
    transforms.ToTensor()
])

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

def train(model, train_loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Előző gradiensek törlése
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Hiba kiszámítása
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Súlyok módosítása
        optimizer.step()

        total_loss += loss.item()

        # Legnagyobb kimenet
        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    average_loss = (
        total_loss / len(train_loader)
    )

    accuracy = (
        100 * correct / total
    )

    return average_loss, accuracy

def evaluate(model, test_loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    average_loss = (
        total_loss / len(test_loader)
    )

    accuracy = (
        100 * correct / total
    )

    return average_loss, accuracy

def main():

    model = MLP().to(device)

# A modell kezdeti, véletlenszerű súlyainak mentése
    torch.save(
    model.state_dict(),
    "results/initial_mlp_mnist.pth"
    )

    print("Kezdeti véletlenszerű súlyok elmentve: initial_mlp_mnist.pth")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(
        "\nBaseline MLP modell tanítása "
        "MNIST adathalmazon..."
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochok száma: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print("-" * 70)

    start_time = time.time()

    # Epochok

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

    print(
        f"Tanítási idő: "
        f"{end_time - start_time:.2f} másodperc"
    )

    final_loss, final_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        f"Végső baseline tesztpontosság: "
        f"{final_acc:.2f}%"
    )

    torch.save(
    model.state_dict(),
    "results/baseline_mlp_mnist.pth"
    )

    print(
        "Baseline modell elmentve: "
        "baseline_mlp_mnist.pth"
    )


if __name__ == "__main__":
    main()