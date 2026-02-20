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


def get_crop(image: Image.Image) -> Image.Image:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])(image)


def predict_top3(model: nn.Module, image: Image.Image) -> list[tuple[str, float]]:
    image_tensor = get_preprocess()(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    values, indices = torch.topk(probs, k=3)
    return [(BREED_NAMES[idx], float(conf)) for conf, idx in zip(values.tolist(), indices.tolist())]


def main() -> None:
    st.set_page_config(page_title="Pet Detective", page_icon="🐾")
    st.markdown("""
<style>
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    img { max-width: 100%; height: auto; }
}
</style>
""", unsafe_allow_html=True)
    st.title("🐾 Pet Detective")
    st.write("Upload a pet photo (or pick an example) to predict the breed.")

    if not MODEL_PATH.exists():
        st.error(f"Model checkpoint not found: `{MODEL_PATH}`")
        st.info("Export the model first from `model.ipynb` to continue local testing.")
        return

    model = load_model()

    # ── Session state ──────────────────────────────────────────────────────
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = None
    if "last_input_id" not in st.session_state:
        st.session_state.last_input_id = None

    # ── 1. Input zone ──────────────────────────────────────────────────────
    col_up, col_cam = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload a photo", type=["jpg", "jpeg", "png", "webp"]
        )
    with col_cam:
        camera_file = st.camera_input("Take a photo")

    # Auto-select newly arrived user input
    if uploaded_file is not None:
        current_input_id = f"upload:{uploaded_file.name}:{uploaded_file.size}"
    elif camera_file is not None:
        current_input_id = f"camera:{camera_file.name}:{camera_file.size}"
    else:
        current_input_id = None

    if current_input_id != st.session_state.last_input_id:
        st.session_state.last_input_id = current_input_id
        if uploaded_file is not None:
            st.session_state.selected_source = "upload"
        elif camera_file is not None:
            st.session_state.selected_source = "camera"

    # ── Build tile pool ────────────────────────────────────────────────────
    tiles: list[dict] = []

    if uploaded_file is not None:
        tiles.append({
            "key": "upload",
            "image": Image.open(uploaded_file).convert("RGB"),
            "is_user": True,
        })
    elif camera_file is not None:
        tiles.append({
            "key": "camera",
            "image": Image.open(camera_file).convert("RGB"),
            "is_user": True,
        })

    if EXAMPLES_DIR.exists():
        example_paths = sorted(
            p for p in EXAMPLES_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for p in example_paths:
            tiles.append({
                "key": str(p),
                "image": Image.open(p).convert("RGB"),
                "is_user": False,
            })

    # If the saved selection no longer exists in the tile pool, reset it
    valid_keys = {t["key"] for t in tiles}
    if st.session_state.selected_source not in valid_keys:
        st.session_state.selected_source = tiles[0]["key"] if tiles else None

    # ── 2. Gallery ─────────────────────────────────────────────────────────
    COLS_PER_ROW = 3

    example_tiles = [t for t in tiles if not t["is_user"]]

    if example_tiles:
        with st.expander("🖼️ Browse examples", expanded=False):
            rows = [example_tiles[i:i + COLS_PER_ROW] for i in range(0, len(example_tiles), COLS_PER_ROW)]
            example_num = 0
            tile_index = 0
            for row_tiles in rows:
                cols = st.columns(len(row_tiles))
                for col, tile in zip(cols, row_tiles):
                    i = tile_index
                    tile_index += 1
                    with col:
                        st.image(get_crop(tile["image"]), use_container_width=True)
                        is_selected = st.session_state.selected_source == tile["key"]
                        example_num += 1
                        btn_label = "✓" if is_selected else f"Example {example_num}"
                        if st.button(btn_label, key=f"tile-{i}", use_container_width=True):
                            st.session_state.selected_source = tile["key"]
                            st.rerun()

    # ── 3. Results ─────────────────────────────────────────────────────────
    selected_tile = next(
        (t for t in tiles if t["key"] == st.session_state.selected_source), None
    )
    if selected_tile is None:
        st.caption("No image selected yet.")
        return

    predictions = predict_top3(model, selected_tile["image"])

    st.subheader("Results")
    medals = ["🥇", "🥈", "🥉"]
    for i, (label, confidence) in enumerate(predictions):
        display_name = label.replace("_", " ").title()
        pct = confidence * 100
        if i == 0:
            st.markdown(f"### {medals[i]} {display_name}")
            st.caption(f"{pct:.1f}% confidence")
        else:
            st.markdown(f"**{medals[i]} {display_name}**")
        st.progress(confidence, text=f"{pct:.1f}%")
        if i < len(predictions) - 1:
            st.write("")


if __name__ == "__main__":
    main()
