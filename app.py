from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


NUM_CLASSES = 37
MODEL_PATH = Path(__file__).parent / "pet_detective_mobilenetv2.pth"
EXAMPLES_DIR = Path(__file__).parent / "examples"

# Matches Oxford-IIIT Pet class ordering used by torchvision.
BREED_NAMES = [
    "Abyssinian",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "boxer",
    "British_Shorthair",
    "chihuahua",
    "Egyptian_Mau",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "Maine_Coon",
    "miniature_pinscher",
    "newfoundland",
    "Persian",
    "pomeranian",
    "pug",
    "Ragdoll",
    "Russian_Blue",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "Siamese",
    "Sphynx",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
]


def build_model() -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


@st.cache_resource
def load_model() -> nn.Module:
    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def square_pad(image: Image.Image, size: int = 200) -> Image.Image:
    """Fit image inside a square canvas without cropping or distorting."""
    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (240, 240, 240))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def predict_top3(model: nn.Module, image: Image.Image) -> list[tuple[str, float]]:
    image_tensor = get_preprocess()(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    values, indices = torch.topk(probs, k=3)
    return [(BREED_NAMES[idx], float(conf)) for conf, idx in zip(values.tolist(), indices.tolist())]


def main() -> None:
    st.set_page_config(page_title="Pet Detective", page_icon="🐾")
    st.title("🐾 Pet Detective")
    st.write("Upload a pet photo (or pick an example) to predict the breed.")

    if not MODEL_PATH.exists():
        st.error(f"Model checkpoint not found: `{MODEL_PATH}`")
        st.info("Export the model first from `model.ipynb` to continue local testing.")
        return

    model = load_model()
    uploaded_file = st.file_uploader("Choose a pet image", type=["jpg", "jpeg", "png", "webp"])

    selected_example: Path | None = None
    if EXAMPLES_DIR.exists():
        example_paths = sorted(
            [p for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        )
        if example_paths:
            st.subheader("Try an example image")
            columns = st.columns(len(example_paths))
            for idx, example_path in enumerate(example_paths):
                with columns[idx]:
                    thumb = square_pad(Image.open(example_path).convert("RGB"))
                    st.image(thumb, use_container_width=True)
                    if st.button(f"Example {idx + 1}", key=f"example-{idx}", use_container_width=True):
                        selected_example = example_path

    image: Image.Image | None = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    elif selected_example is not None:
        image = Image.open(selected_example).convert("RGB")

    if image is None:
        st.caption("No image selected yet.")
        return

    st.image(image, caption="Input image", use_container_width=True)
    predictions = predict_top3(model, image)

    st.subheader("Top-3 predictions")
    for label, confidence in predictions:
        st.write(f"- **{label}**: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
