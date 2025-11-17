# app.py
# Skin lesion Streamlit app with Google Drive hard-coded IDs.
# You do NOT need to type file IDs in the UI.
#
# Drive files:
#   MODEL_FILE_ID        = "15rQdpHiHI9HGBlKpRmlBfOsTmxUCI9m3"
#   METADATA_FILE_ID     = "11RG8Wf2YOxnN5oGbVXgEv1D5Nh_dAYwi"
#   DATASET_ZIP_FILE_ID  = "1oypOScrmWuw3-vS8Mfg8nXj70suIawZt"
#
# REQUIREMENT: you must have st.secrets['gdrive_service_account'] configured
# with your service-account JSON (and that SA must have Viewer access
# to the three Drive files above).

import os
import io
import tempfile
import zipfile
from glob import glob
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import streamlit as st
import tensorflow as tf
from gradcam import make_gradcam_heatmap_tf2, overlay_heatmap_on_pil

# ---------------------------
# HARDCODED Google Drive IDs
# ---------------------------
MODEL_FILE_ID = "15rQdpHiHI9HGBlKpRmlBfOsTmxUCI9m3"
METADATA_FILE_ID = "11RG8Wf2YOxnN5oGbVXgEv1D5Nh_dAYwi"
DATASET_ZIP_FILE_ID = "1oypOScrmWuw3-vS8Mfg8nXj70suIawZt"
EVAL_ACC_FILE_ID = "1mmjRXdAZpMVQYbOAtGz9adiP55CkT1qN"
EVAL_CM_FILE_ID = "1NyqjQldbgzNHqf7n3NQNEdV7wHF9ONDi"
EVAL_LOSS_FILE_ID = "14Xz86Sn97uHRG_A5f5-xH3zKWGli2arH"

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Skin Lesion · Classifier + Grad-CAM", layout="wide")
st.title("Skin Lesion Classifier · Grad-CAM · Dataset Preview · Evaluation")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Settings")

    data_source = st.radio(
        "Dataset source",
        ["Local folder", "Google Drive (hard-coded IDs)"],
        index=1,
    )

    # Local path only used in local mode
    default_dir = "./ham10000"
    DATA_DIR = st.text_input(
        "Local dataset directory (root)",
        value=default_dir,
        help="Used only when 'Local folder' is selected.",
    )

    target_img_size = st.number_input(
        "Model input size (square)",
        min_value=64,
        max_value=768,
        value=128,
        step=32,
    )

    st.markdown("### Google Drive info (read-only)")
    st.caption(f"Model file ID: `{MODEL_FILE_ID}`")
    st.caption(f"Metadata file ID: `{METADATA_FILE_ID}`")
    st.caption(f"Dataset ZIP file ID: `{DATASET_ZIP_FILE_ID}`")

# ---------------------------
# Google Drive helpers
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_drive_service_from_secrets():
    """Create Drive API service using st.secrets['gdrive_service_account']."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except Exception:
        return None

    if "gdrive_service_account" not in st.secrets:
        return None

    creds_info = dict(st.secrets["gdrive_service_account"])
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=scopes
    )
    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception:
        return None


@st.cache_resource(show_spinner=True)
def get_dataset_zip_bytes():
    """Download dataset ZIP from Drive once and cache the raw bytes."""
    if not DATASET_ZIP_FILE_ID:
        raise RuntimeError("DATASET_ZIP_FILE_ID is empty.")
    b = gdrive_download_bytes(DATASET_ZIP_FILE_ID)
    if not b:
        raise RuntimeError("Failed to download dataset ZIP from Drive.")
    return b


@st.cache_resource(show_spinner=False)
def get_zip_name_map():
    """Build a mapping image_id -> ZIP member name (only JPG files)."""
    zip_bytes = get_dataset_zip_bytes()
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    name_map = {}
    for name in zf.namelist():
        if name.lower().endswith(".jpg"):
            key = os.path.splitext(os.path.basename(name))[0]
            name_map[key] = name
    return name_map


@st.cache_data(show_spinner=False)
def gdrive_download_bytes(file_id: str) -> bytes:
    """Download a file from Google Drive and return raw bytes."""
    service = get_drive_service_from_secrets()
    if service is None or not file_id:
        return b""

    from googleapiclient.http import MediaIoBaseDownload

    fh = io.BytesIO()
    request = service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()


@st.cache_data(show_spinner=True)
def load_image_from_drive(file_id: str):
    """Download an image file from Google Drive and return a PIL.Image object."""
    b = gdrive_download_bytes(file_id)
    if not b:
        return None
    try:
        return Image.open(io.BytesIO(b))
    except Exception:
        return None


def ensure_drive_available():
    """Check Google Drive connection and show a clear error if it's not ready."""
    service = get_drive_service_from_secrets()
    if service is None:
        st.error(
            "Google Drive is not configured or reachable.\n\n"
            "Please check:\n"
            "1. `.streamlit/secrets.toml` contains a [gdrive_service_account] section;\n"
            "2. The service account's client_email has Viewer access to all required Drive files;\n"
            "3. Python packages `google-api-python-client` and `google-auth` are installed."
        )
        st.stop()


@st.cache_resource(show_spinner=True)
def load_model_from_drive() -> tf.keras.Model | None:
    """Download .keras model from Drive and load it."""
    if not MODEL_FILE_ID:
        return None
    b = gdrive_download_bytes(MODEL_FILE_ID)
    if not b:
        raise RuntimeError(
            "Failed to download model bytes from Drive (check file ID & sharing)."
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    tmp.write(b)
    tmp.flush()
    tmp.close()

    # If model is from AutoKeras, try to add custom objects.
    custom_objects = None
    try:
        import autokeras as ak  # type: ignore

        custom_objects = ak.CUSTOM_OBJECTS
    except Exception:
        custom_objects = None

    model = tf.keras.models.load_model(
        tmp.name, custom_objects=custom_objects, compile=False
    )
    return model


@st.cache_resource(show_spinner=True)
def prepare_dataset_dir_from_drive_zip() -> str:
    """Download dataset ZIP from Drive, unzip to a temp directory, and return that directory."""
    if not DATASET_ZIP_FILE_ID:
        raise RuntimeError("DATASET_ZIP_FILE_ID is empty.")
    b = gdrive_download_bytes(DATASET_ZIP_FILE_ID)
    if not b:
        raise RuntimeError("Failed to download dataset ZIP from Drive.")

    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_zip.write(b)
    tmp_zip.flush()
    tmp_zip.close()

    extract_dir = tempfile.mkdtemp(prefix="ham10000_")
    with zipfile.ZipFile(tmp_zip.name, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


# ---------------------------
# Image / Grad-CAM utilities
# ---------------------------
def preprocess_input(img_pil, size):
    """Resize and scale image to [0,1] range."""
    img = img_pil.convert("RGB").resize((size, size))
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def find_last_conv_like_layer(keras_model):
    """Return the name of the last Conv-like layer in the model.

    This searches from the end and supports Conv2D, SeparableConv2D, DepthwiseConv2D.
    If your architecture is special, you can replace this with a fixed layer name.
    """
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    for layer in keras_model.layers[::-1]:
        if isinstance(layer, conv_types):
            return layer.name
        # Support nested models (e.g. Sequential or Functional blocks)
        if hasattr(layer, "layers"):
            for sub in layer.layers[::-1]:
                if isinstance(sub, conv_types):
                    return sub.name
    raise ValueError("No Conv/Separable/Depthwise conv layer found in the model.")
def list_conv_like_layers(keras_model):
    """Return a list of names of all Conv-like layers (in order)."""
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    names = []
    for layer in keras_model.layers:
        if isinstance(layer, conv_types):
            names.append(layer.name)
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, conv_types):
                    names.append(sub.name)
    return names

def plot_prob_bar(labels, probs):
    """Bar plot of class probabilities."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    order = np.argsort(-probs)
    ax.bar([labels[i] for i in order], probs[order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Class")
    ax.set_title("Predicted probabilities")
    fig.tight_layout()
    return fig


# ---------------------------
# Tabs
# ---------------------------
TAB1, TAB2, TAB3 = st.tabs(
    ["1) Upload · Classify · Grad-CAM", "2) Dataset Preview", "3) Evaluation Overview"]
)

# ---------------------------
# Tab 1 – Upload · Classify · Grad-CAM
# ---------------------------
with TAB1:
    st.subheader("Upload an image, load your model, run prediction, and visualize Grad-CAM")

    model = st.session_state.get("_loaded_model")
    class_names = st.session_state.get("_class_names")
    # Optional: choose which conv layer to use for Grad-CAM
    gradcam_layer_name = None
    if model is not None:
        if "_conv_layer_names" not in st.session_state:
            st.session_state["_conv_layer_names"] = list_conv_like_layers(model)
        conv_layers = st.session_state["_conv_layer_names"]
        if conv_layers:
            gradcam_layer_name = st.selectbox(
                "Grad-CAM convolution layer (from early to late)",
                conv_layers,
                index=len(conv_layers) - 1,  # default = last conv layer
            )

    col1, col2 = st.columns([1, 1])
    with col1:
        model_source = st.radio(
            "Model source",
            ["Local file / path", "Google Drive (hard-coded ID)"],
            index=1,
        )

        model_file = None
        model_path_text = None
        if model_source == "Local file / path":
            model_file = st.file_uploader(
                "Upload Keras model file (*.keras)",
                type=["keras"],
                help="Upload your trained .keras model, or use local path below.",
            )
            model_path_text = st.text_input(
                "…or provide a local path to your model",
                value="./skin_xception_augmented_from_scratch.keras",
            )

        load_btn = st.button("Load / reload model")

    # Auto-load from Drive once if using Drive mode
    if model is None and model_source == "Google Drive (hard-coded ID)":
        ensure_drive_available()
        try:
            with st.spinner("Downloading model from Google Drive ..."):
                model = load_model_from_drive()
            st.session_state["_loaded_model"] = model
            st.success("Model auto-loaded from Google Drive ✅")
        except Exception as e:
            st.error(f"Auto-load model from Drive failed: {e}")
            st.error(f"Grad-CAM generation failed: {e}")
            st.text("Traceback:")
            st.text(traceback.format_exc())

    if load_btn:
        try:
            if model_source == "Google Drive (hard-coded ID)":
                ensure_drive_available()
                with st.spinner("Downloading model from Google Drive ..."):
                    model = load_model_from_drive()
            else:
                if model_file is not None:
                    tmp_path = "uploaded_model.keras"
                    with open(tmp_path, "wb") as f:
                        f.write(model_file.read())
                    model = tf.keras.models.load_model(tmp_path)
                else:
                    model = tf.keras.models.load_model(model_path_text)
            st.session_state["_loaded_model"] = model
            st.success("Model loaded ✅")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            model = None

    up_img = st.file_uploader(
        "Upload a lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="image_upl"
    )

    if model is not None and up_img is not None:
        img_pil = Image.open(up_img)
        st.image(img_pil, caption="Uploaded image", width=320)

        arr = preprocess_input(img_pil, target_img_size)
        probs = model.predict(arr[None, ...], verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        if class_names is None:
            labels = np.array([str(i) for i in range(len(probs))])
        else:
            labels = np.array(class_names)

        st.markdown(f"**Predicted class:** {labels[pred_idx]}")
        st.pyplot(plot_prob_bar(labels, probs))

        try:
            last_conv_name = find_last_conv_like_layer(model)

            heatmap, _ = make_gradcam_heatmap_tf2(
                arr, model, last_conv_layer_name=last_conv_name, class_index=pred_idx
            )

            overlay = overlay_heatmap_on_pil(img_pil, heatmap, alpha=0.35)

            cA, cB, cC = st.columns(3)
            with cA:
                st.image(img_pil, caption="Input (original)", use_container_width=True)
            with cB:
                st.image(
                    heatmap,
                    clamp=True,
                    caption=f"Grad-CAM heatmap ({last_conv_name})",
                    use_container_width=True,
                )
            with cC:
                st.image(overlay, caption="Overlay (jet × original)", use_container_width=True)

        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")
            st.text(traceback.format_exc())

    else:
        st.info("Load a model and upload an image to run prediction + Grad-CAM.")

# ---------------------------
# Tab 2 – Dataset Preview
# ---------------------------
with TAB2:
    st.subheader("Preview the dataset and class distribution")

    # Remember whether we have already loaded metadata
    if "meta_loaded" not in st.session_state:
        st.session_state["meta_loaded"] = False

    load_clicked = st.button("Load dataset metadata")

    # Once clicked, keep the flag = True so later reruns still execute the preview code
    if load_clicked:
        st.session_state["meta_loaded"] = True

    if st.session_state["meta_loaded"]:
        try:
            if data_source != "Google Drive (hard-coded IDs)":
                st.warning("Right now the preview only supports the Google Drive mode.")
                st.stop()

            ensure_drive_available()

            # ---------- Step 1: FULL metadata (for pie chart) ----------
            st.info("Step 1/3: Downloading FULL metadata CSV from Google Drive ...")
            with st.spinner("Downloading FULL metadata CSV from Drive ..."):
                csv_bytes_full = gdrive_download_bytes(METADATA_FILE_ID)

            if not csv_bytes_full:
                raise RuntimeError(
                    "gdrive_download_bytes returned empty bytes for METADATA_FILE_ID"
                )

            df_full = pd.read_csv(io.BytesIO(csv_bytes_full))
            st.success(f"Loaded FULL metadata CSV with {len(df_full)} rows.")
            st.write("FULL metadata head:", df_full.head())

            # Cache class names for Tab 1 (use FULL metadata)
            class_names = sorted(df_full["dx"].unique().tolist())
            st.session_state["_class_names"] = class_names
            st.write("Classes found in FULL metadata:", class_names)
            # Add explanation of diagnostic abbreviations
            st.markdown("**Class abbreviation meanings:**")
            class_expl = {
                "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen’s disease",
                "bcc": "Basal cell carcinoma",
                "bkl": "Benign keratosis-like lesions (solar lentigines / seborrheic keratoses / lichen-planus like keratoses)",
                "df": "Dermatofibroma",
                "mel": "Melanoma",
                "nv": "Melanocytic nevi",
                "vasc": "Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)"
            }
            # Display nicely formatted as a Markdown table
            st.markdown("| Abbreviation | Full Diagnosis |")
            st.markdown("|--------------|----------------|")
            for k, v in class_expl.items():
                st.markdown(f"| `{k}` | {v} |")

            # ---------- Step 2: pie chart using FULL metadata ----------
            st.info("Rendering class distribution pie chart (FULL metadata) ...")
            counts = df_full["dx"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            ax.set_title("Class distribution (dx) – FULL metadata")
            st.pyplot(fig)
            st.success("Metadata preview finished ✅")

            # ---------- Step 3: automatically load SMALL dataset for image preview ----------
            st.markdown("---")
            st.subheader("Sample images from SMALL dataset in ZIP")

            st.info(
                "Step 2/3: Downloading SMALL dataset ZIP (metadata_small + images) ..."
            )
            with st.spinner("Downloading SMALL dataset ZIP from Drive ..."):
                zip_bytes = get_dataset_zip_bytes()

            zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
            namelist = zf.namelist()

            # Find the small metadata CSV inside the ZIP:
            # Prefer a .csv whose filename contains 'small'; otherwise use the first .csv.
            small_csv_candidates = [n for n in namelist if n.lower().endswith(".csv")]
            if not small_csv_candidates:
                raise RuntimeError("No CSV file found inside SMALL dataset ZIP.")

            small_csv_name = None
            for n in small_csv_candidates:
                if "small" in os.path.basename(n).lower():
                    small_csv_name = n
                    break
            if small_csv_name is None:
                small_csv_name = small_csv_candidates[0]

            df_small = pd.read_csv(zf.open(small_csv_name))
            st.success(
                f"Loaded SMALL metadata from '{small_csv_name}' with {len(df_small)} rows."
            )

            # Number of examples per class from the SMALL dataset
            n_per_cls = st.slider(
                "Samples per class to display (SMALL dataset)",
                1,
                5,
                3,
            )

            st.info("Step 3/3: Reading sample images from SMALL dataset ZIP ...")

            # Build image_id -> filename map from the ZIP
            with st.spinner("Building image_id → ZIP member map (cached) ..."):
                name_map = get_zip_name_map()
            st.write(f"DEBUG: found {len(name_map)} JPG files in SMALL ZIP.")

            available_ids = set(name_map.keys())

            with st.spinner("Loading sample images per class from SMALL dataset ..."):
                for cls in class_names:
                    # Select rows in SMALL metadata where this class exists and the image is present in the ZIP
                    subset = df_small[
                        (df_small["dx"] == cls)
                        & (df_small["image_id"].isin(available_ids))
                    ].copy()
                    total_cls = len(subset)
                    if total_cls == 0:
                        continue

                    subset = subset.sample(min(n_per_cls, total_cls), random_state=42)

                    with st.expander(
                        f"Class: {cls} — {total_cls} images in SMALL metadata"
                    ):
                        cols = st.columns(n_per_cls)
                        for i, (_, row) in enumerate(subset.iterrows()):
                            img_id = row["image_id"]
                            member_name = name_map.get(img_id)
                            with cols[i % n_per_cls]:
                                if member_name is None:
                                    st.write(f"(image {img_id} not found in ZIP)")
                                    continue
                                try:
                                    img_bytes = zf.read(member_name)
                                    img = Image.open(io.BytesIO(img_bytes))
                                    st.image(img, caption=f"{img_id}.jpg")
                                except Exception as img_err:
                                    st.write(
                                        f"(image {img_id} not displayable: {img_err})"
                                    )

            st.success("Sample images from SMALL dataset loaded ✅")

        except Exception as e:
            st.error(f"Failed to load dataset preview: {e}")
            st.text("Traceback:")
            st.text(traceback.format_exc())
    else:
        st.info('Click "Load dataset metadata" to preview.')

# ---------------------------
# Tab 3 – Evaluation Overview
# ---------------------------
with TAB3:
    st.subheader("Model Evaluation – fixed plots from Google Drive")

    # Make sure Drive is available (service account + packages)
    ensure_drive_available()

    with st.spinner("Downloading evaluation plots from Google Drive ..."):
        loss_img = load_image_from_drive(EVAL_LOSS_FILE_ID)
        acc_img = load_image_from_drive(EVAL_ACC_FILE_ID)
        cm_img = load_image_from_drive(EVAL_CM_FILE_ID)

    c1, c2, c3 = st.columns(3)

    with c1:
        if loss_img is not None:
            st.image(
                loss_img,
                caption="Training / Validation Loss",
                use_container_width=True,
            )
        else:
            st.error("Failed to load Loss curve image from Google Drive.")

    with c2:
        if acc_img is not None:
            st.image(
                acc_img,
                caption="Training / Validation Accuracy",
                use_container_width=True,
            )
        else:
            st.error("Failed to load Accuracy curve image from Google Drive.")

    with c3:
        if cm_img is not None:
            st.image(
                cm_img,
                caption="Confusion Matrix",
                use_container_width=True,
            )
        else:
            st.error("Failed to load Confusion Matrix image from Google Drive.")

st.caption(
    "Hard-coded Google Drive IDs · Model + metadata + dataset ZIP are all fetched via Drive API."
)
