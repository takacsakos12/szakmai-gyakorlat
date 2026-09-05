import torch

from torchvision import transforms

from PIL import Image, ImageOps

import numpy as np
import cv2

from model import MLP

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Használt eszköz:", device)

def remove_small_components(image_array):

    # Bináris kép
    binary = (
        image_array > 30
    ).astype(np.uint8)


    # Összefüggő komponensek
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )


    # Ha nincs felismerhető objektum
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

def preprocess_image(image_path):
    image = Image.open(
        image_path
    ).convert("L")

    print(
        "Eredeti kép mérete:",
        image.size
    )

    image = ImageOps.invert(
        image
    )

    image_array = np.array(
        image
    )

    threshold = 30

    image_array[
        image_array < threshold
    ] = 0

    image_array = remove_small_components(
        image_array
    )

    coords = np.argwhere(
        image_array > 0
    )


    if coords.size == 0:

        raise ValueError(
            "Nem található számjegy a képen."
        )


    y_min, x_min = coords.min(
        axis=0
    )

    y_max, x_max = coords.max(
        axis=0
    )

    cropped = Image.fromarray(
        image_array
    ).crop(
        (
            x_min,
            y_min,
            x_max + 1,
            y_max + 1
        )
    )


    print(
        "Körbevágott számjegy mérete:",
        cropped.size
    )

    width, height = cropped.size

    max_digit_size = 20


    if width > height:

        new_width = max_digit_size

        new_height = round(
            height
            * max_digit_size
            / width
        )

    else:

        new_height = max_digit_size

        new_width = round(
            width
            * max_digit_size
            / height
        )


    new_width = max(
        1,
        new_width
    )

    new_height = max(
        1,
        new_height
    )


    cropped = cropped.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )


    print(
        "Átméretezett számjegy mérete:",
        cropped.size
    )

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )


    x_offset = (
        28 - new_width
    ) // 2

    y_offset = (
        28 - new_height
    ) // 2


    canvas.paste(
        cropped,
        (
            x_offset,
            y_offset
        )
    )


    canvas.save(
        "processed_input.png"
    )

    print(
        "Feldolgozott kép: "
        "processed_input.png"
    )

    tensor = transforms.ToTensor()(
        canvas
    )


    return tensor

def predict_png(
    model,
    image_path
):

    model.eval()


    image_tensor = preprocess_image(
        image_path
    )
    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )


        # Softmax
        probabilities = torch.softmax(
            output,
            dim=1
        )


        # Legnagyobb valószínűség
        predicted = torch.argmax(
            probabilities,
            dim=1
        ).item()

    print("\n" + "-" * 50)

    print(
        "Kép:",
        image_path
    )

    print(
        "Predikció:",
        predicted
    )

    print("-" * 50)


    print("\nValószínűségek:")


    for digit, probability in enumerate(
        probabilities[0]
    ):

        print(
            f"{digit}: "
            f"{probability.item() * 100:.2f}%"
        )


    confidence = (
        probabilities[
            0,
            predicted
        ].item()
        * 100
    )


    print(
        f"\nBizonyosság: "
        f"{confidence:.2f}%"
    )


    return predicted

def main():

    # Modell létrehozása
    model = MLP().to(
        device
    )

    model.load_state_dict(
        torch.load(
            "results/baseline_mlp_mnist.pth",
            map_location=device
        )
    )


    print(
        "Baseline modell betöltve."
    )

    image_path = "images/sajat_szam.png"


    # Predikció
    predict_png(
        model,
        image_path
    )


if __name__ == "__main__":
    main()