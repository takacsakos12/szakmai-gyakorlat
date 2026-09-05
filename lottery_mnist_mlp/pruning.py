import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.002

# Minden IMP körben a még aktív súlyok 20%-át prune-oljuk
PRUNE_PER_ROUND = 0.20

# Hány pruning + retraining kör legyen
NUM_ROUNDS = 30

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

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

def evaluate(model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    return accuracy

def create_initial_masks(model):

    masks = {}

    for name, param in model.named_parameters():

        if "weight" in name:

            masks[name] = torch.ones_like(
                param,
                dtype=torch.bool
            )

    return masks

def prune_smallest_weights(model, masks, pruning_ratio):

    active_weights = []

    # Csak a jelenleg aktív súlyokat gyűjtjük össze
    for name, param in model.named_parameters():

        if "weight" in name:

            mask = masks[name]

            weights = param.data.abs()[mask]

            active_weights.append(weights)

    active_weights = torch.cat(active_weights)

    # Megkeressük az alsó 20% határértékét
    threshold = torch.quantile(
        active_weights,
        pruning_ratio
    )

    # Frissítjük a maskokat
    for name, param in model.named_parameters():

        if "weight" in name:

            old_mask = masks[name]

            new_mask = (
                param.data.abs() > threshold
            )

            # Ami már egyszer 0 lett,
            # az többé nem jöhet vissza
            masks[name] = old_mask & new_mask

    return threshold.item()

def reset_to_initial_weights(
    model,
    initial_state,
    masks
):

    for name, param in model.named_parameters():

        # Visszatöltjük az eredeti
        # tanítás előtti értéket
        param.data.copy_(
            initial_state[name]
        )

        # Weight esetén alkalmazzuk a maskot
        if "weight" in name:

            param.data.mul_(
                masks[name]
            )


def train_sparse_model(model, masks):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # A prune-olt súlyok gradientje is 0 legyen
            for name, param in model.named_parameters():

                if (
                    "weight" in name
                    and param.grad is not None
                ):

                    param.grad.mul_(
                        masks[name]
                    )

            optimizer.step()

            # Biztonságból minden update után
            # újra alkalmazzuk a maskot
            for name, param in model.named_parameters():

                if "weight" in name:

                    param.data.mul_(
                        masks[name]
                    )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_accuracy = (
            100 * correct / total
        )

        test_accuracy = evaluate(model)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train: {train_accuracy:.2f}% | "
            f"Test: {test_accuracy:.2f}%"
        )

def count_weights(masks):

    total_weights = 0
    active_weights = 0

    for mask in masks.values():

        total_weights += mask.numel()

        active_weights += (
            mask.sum().item()
        )

    zero_weights = (
        total_weights - active_weights
    )

    sparsity = (
        100
        * zero_weights
        / total_weights
    )

    return (
        total_weights,
        zero_weights,
        active_weights,
        sparsity
    )


def main():

    initial_state = torch.load(
    "results/initial_mlp_mnist.pth",
    map_location=device
    )


    model = MLP().to(device)

    model.load_state_dict(
    torch.load(
        "results/baseline_mlp_mnist.pth",
        map_location=device
        )
    )

    masks = create_initial_masks(model)

    baseline_accuracy = evaluate(model)

    print("\n--------------------------------")
    print("BASELINE")
    print("--------------------------------")

    print(
        f"Accuracy: {baseline_accuracy:.2f}%"
    )

    total, zero, active, sparsity = (
        count_weights(masks)
    )

    print(f"Összes súly: {total}")
    print(f"Aktív súly: {active}")
    print(f"Nulla súly: {zero}")
    print(f"Sparsity: {sparsity:.2f}%")

    for round_number in range(
        1,
        NUM_ROUNDS + 1
    ):

        print("\n================================")
        print(f"IMP KÖR {round_number}")
        print("================================")

        # Pruning a jelenleg betanított modellből
        

        threshold = prune_smallest_weights(
            model,
            masks,
            PRUNE_PER_ROUND
        )

        total, zero, active, sparsity = (
            count_weights(masks)
        )

        print(
            f"Pruning threshold: "
            f"{threshold:.6f}"
        )

        print(
            f"Sparsity pruning után: "
            f"{sparsity:.2f}%"
        )

        print(
            f"Aktív súlyok: {active}"
        )

        # Reset az eredeti random inicializációra

        reset_to_initial_weights(
            model,
            initial_state,
            masks
        )

        # Sparse háló újratanítása

        print("\nÚjratanítás...")

        train_sparse_model(
            model,
            masks
        )

        accuracy = evaluate(model)

        total, zero, active, sparsity = (
            count_weights(masks)
        )

        print("\n--- Kör eredménye ---")

        print(
            f"Sparsity: {sparsity:.2f}%"
        )

        print(
            f"Aktív súlyok: {active}"
        )

        print(
            f"Nulla súlyok: {zero}"
        )

        print(
            f"Test accuracy: "
            f"{accuracy:.2f}%"
        )

        torch.save({
        "model_state_dict": model.state_dict(),
        "masks": masks,
        "sparsity": sparsity,
        "accuracy": accuracy
        }, f"results/imp_round_{round_number}.pth")

        print(
        f"Modell elmentve: results/imp_round_{round_number}.pth"
        )


if __name__ == "__main__":
    main()