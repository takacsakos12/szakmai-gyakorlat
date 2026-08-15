import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import cv2


# --------------------------------------------------
# Eszköz kiválasztása
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Használt eszköz:", device)


# --------------------------------------------------
# Ugyanaz az MLP architektúra,
# mint a tanító programban
# --------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# Kis zajok eltávolítása
# --------------------------------------------------
# A legnagyobb összefüggő világos objektumot tartjuk meg.
# Feltételezzük, hogy ez maga a számjegy.
# --------------------------------------------------

def remove_small_components(image_array):

    # Bináris kép létrehozása:
    # 0 = háttér
    # 1 = számjegy / objektum
    binary = (image_array > 30).astype(np.uint8)

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

    # 0-ás label mindig a háttér
    # Ha nincs valódi objektum, visszaadjuk az eredetit
    if num_labels <= 1:
        return image_array

    # Megkeressük a legnagyobb objektumot
    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for label in range(2, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = label

    # Új fekete kép
    cleaned = np.zeros_like(image_array)

    # Csak a legnagyobb komponens marad
    cleaned[labels == largest_label] = \
        image_array[labels == largest_label]

    return cleaned


# --------------------------------------------------
# Külső PNG előfeldolgozása
# --------------------------------------------------

def preprocess_image(image_path):

    # 1. Kép betöltése szürkeárnyalatosan
    image = Image.open(image_path).convert("L")

    print("Eredeti kép mérete:", image.size)


    # --------------------------------------------------
    # 2. Inverzió
    # --------------------------------------------------
    # Tipikus külső kép:
    # fehér háttér + fekete szám
    #
    # MNIST:
    # fekete háttér + világos szám
    #
    # Ezért invertálunk.
    # --------------------------------------------------

    image = ImageOps.invert(image)


    # --------------------------------------------------
    # 3. NumPy tömbbé alakítás
    # --------------------------------------------------

    image_array = np.array(image)


    # --------------------------------------------------
    # 4. Küszöbölés
    # --------------------------------------------------

    threshold = 30

    image_array[image_array < threshold] = 0


    # --------------------------------------------------
    # 5. Kis zajok eltávolítása
    # --------------------------------------------------

    image_array = remove_small_components(
        image_array
    )


    # --------------------------------------------------
    # 6. A számjegy pozíciójának meghatározása
    # --------------------------------------------------

    coords = np.argwhere(
        image_array > 0
    )

    if coords.size == 0:
        raise ValueError(
            "Nem található számjegy a képen."
        )


    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)


    # --------------------------------------------------
    # 7. Körbevágás
    # --------------------------------------------------

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


    # --------------------------------------------------
    # 8. Aránytartó resize
    # --------------------------------------------------
    # A hosszabb oldal maximum 20 pixel lesz.
    # Így marad körülötte egy kis margó a 28x28-as képen.
    # --------------------------------------------------

    width, height = cropped.size

    max_digit_size = 20


    if width > height:

        new_width = max_digit_size

        new_height = round(
            height * max_digit_size / width
        )

    else:

        new_height = max_digit_size

        new_width = round(
            width * max_digit_size / height
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


    # --------------------------------------------------
    # 9. 28x28 fekete vászon
    # --------------------------------------------------

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )


    # --------------------------------------------------
    # 10. Geometriai középre igazítás
    # --------------------------------------------------

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


    # --------------------------------------------------
    # 11. Előfeldolgozott kép mentése
    # --------------------------------------------------
    # Ez nagyon hasznos debughoz:
    # pontosan látod, mit kap majd az MLP.
    # --------------------------------------------------

    canvas.save(
        "processed_input.png"
    )

    print(
        "Előfeldolgozott kép elmentve:"
        " processed_input.png"
    )


    # --------------------------------------------------
    # 12. Tensor
    # --------------------------------------------------
    # Mivel a jelenlegi baseline modellt
    # normalizálás nélkül tanítottuk,
    # itt sem normalizálunk.
    # --------------------------------------------------

    tensor = transforms.ToTensor()(
        canvas
    )

    return tensor


# --------------------------------------------------
# Predikció
# --------------------------------------------------

def predict_png(model, image_path):

    model.eval()


    # Előfeldolgozás
    image_tensor = preprocess_image(
        image_path
    )


    # --------------------------------------------------
    # [1, 28, 28]
    # ->
    # [1, 1, 28, 28]
    #
    # Első 1 = batch size
    # Második 1 = szürke csatorna
    # --------------------------------------------------

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )


    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )


        # Nyers kimenetekből valószínűségek
        probabilities = torch.softmax(
            output,
            dim=1
        )


        # Legnagyobb valószínűségű osztály
        predicted = torch.argmax(
            probabilities,
            dim=1
        ).item()


    # --------------------------------------------------
    # Eredmény kiírása
    # --------------------------------------------------

    print("\n" + "-" * 50)

    print(
        "Kép:",
        image_path
    )

    print(
        "A modell predikciója:",
        predicted
    )

    print("-" * 50)


    print(
        "\nValószínűségek:"
    )


    for digit, probability in enumerate(
        probabilities[0]
    ):

        print(
            f"{digit}: "
            f"{probability.item() * 100:.2f}%"
        )


    # Legnagyobb valószínűség
    confidence = probabilities[
        0,
        predicted
    ].item() * 100

    print(
        f"\nBizonyosság: "
        f"{confidence:.2f}%"
    )


    return predicted


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # 1. Modell létrehozása
    # --------------------------------------------------

    model = MLP().to(
        device
    )


    # --------------------------------------------------
    # 2. Betanított súlyok betöltése
    # --------------------------------------------------
    # A fájlnevet igazítsd ahhoz,
    # amit ténylegesen mentettél.
    # --------------------------------------------------

    model.load_state_dict(
        torch.load(
            "baseline_mlp_mnist.pth",
            map_location=device
        )
    )


    print(
        "Betanított modell sikeresen betöltve."
    )


    # --------------------------------------------------
    # 3. Tesztelendő PNG
    # --------------------------------------------------

    image_path = "images/sajat_8.png"


    # --------------------------------------------------
    # 4. Predikció
    # --------------------------------------------------

    predict_png(
        model,
        image_path
    )


if __name__ == "__main__":
    main()