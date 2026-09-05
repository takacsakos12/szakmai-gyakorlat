import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from model import MLP


BATCH_SIZE = 128

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MNIST_C_PATH = "data/MNIST-C/mnist_c"


def remove_small_components(image_array):

    binary = (
        image_array > 30
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return image_array

    largest_label = 1
    largest_area = stats[
        1,
        cv2.CC_STAT_AREA
    ]

    for label in range(
        2,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area > largest_area:

            largest_area = area
            largest_label = label

    cleaned = np.zeros_like(
        image_array
    )

    cleaned[
        labels == largest_label
    ] = image_array[
        labels == largest_label
    ]

    return cleaned


def preprocess_image(image_array):

    image_array = image_array.copy()

    if image_array.ndim == 3:
        image_array = image_array.squeeze()

    image_array[
        image_array < 30
    ] = 0

    image_array = remove_small_components(
        image_array
    )

    coords = np.argwhere(
        image_array > 0
    )

    if len(coords) == 0:

        return np.zeros(
            (28, 28),
            dtype=np.uint8
        )

    y_min, x_min = coords.min(
        axis=0
    )

    y_max, x_max = coords.max(
        axis=0
    )

    cropped = image_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    height, width = cropped.shape

    max_dimension = max(
        height,
        width
    )

    scale = 20 / max_dimension

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    resized = cv2.resize(
        cropped,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (
        28 - new_width
    ) // 2

    y_offset = (
        28 - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


class MNISTCDataset(Dataset):

    def __init__(
        self,
        images_path,
        labels_path,
        preprocess=False
    ):

        self.images = np.load(
            images_path
        )

        self.labels = np.load(
            labels_path
        )

        self.preprocess = preprocess

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        if self.preprocess:

            image = preprocess_image(
                image
            )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        image = image / 255.0

        if image.ndim == 2:

            image = image.unsqueeze(0)

        elif image.ndim == 3:

            image = image.permute(
                2,
                0,
                1
            )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = (
        100 * correct / total
    )

    return accuracy


def main():

    model = MLP().to(
        device
    )

    model.load_state_dict(
        torch.load(
            "results/baseline_mlp_mnist.pth",
            map_location=device
        )
    )

    print("Device:", device)

    print(
        "\nMNIST-C BASELINE TESZT"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"{'Corruption':20s}"
        f"{'Nyers':>15s}"
        f"{'Feldolgozott':>20s}"
    )

    print(
        "------------------------------------------------------------"
    )

    raw_results = []
    processed_results = []

    for corruption in sorted(
        os.listdir(
            MNIST_C_PATH
        )
    ):

        corruption_path = os.path.join(
            MNIST_C_PATH,
            corruption
        )

        if not os.path.isdir(
            corruption_path
        ):
            continue

        images_path = os.path.join(
            corruption_path,
            "test_images.npy"
        )

        labels_path = os.path.join(
            corruption_path,
            "test_labels.npy"
        )

        if not (
            os.path.exists(images_path)
            and os.path.exists(labels_path)
        ):
            continue

        raw_dataset = MNISTCDataset(
            images_path,
            labels_path,
            preprocess=False
        )

        processed_dataset = MNISTCDataset(
            images_path,
            labels_path,
            preprocess=True
        )

        raw_loader = DataLoader(
            raw_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        processed_loader = DataLoader(
            processed_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        raw_accuracy = evaluate(
            model,
            raw_loader
        )

        processed_accuracy = evaluate(
            model,
            processed_loader
        )

        raw_results.append(
            raw_accuracy
        )

        processed_results.append(
            processed_accuracy
        )

        print(
            f"{corruption:20s}"
            f"{raw_accuracy:14.2f}%"
            f"{processed_accuracy:19.2f}%"
        )

    print(
        "------------------------------------------------------------"
    )

    if (
        raw_results
        and processed_results
    ):

        raw_average = (
            sum(raw_results)
            / len(raw_results)
        )

        processed_average = (
            sum(processed_results)
            / len(processed_results)
        )

        print(
            f"{'ÁTLAG':20s}"
            f"{raw_average:14.2f}%"
            f"{processed_average:19.2f}%"
        )


if __name__ == "__main__":
    main()