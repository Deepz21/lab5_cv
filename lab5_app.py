from hashlib import sha256
from io import BytesIO

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models
import pandas as pd

st.set_page_config(
    page_title="Image Classifier | PyTorch",
    page_icon="🖼️",
    layout="wide",
)

MODEL_OPTIONS = {
    "ResNet-18": {
        "builder": models.resnet18,
        "weights": models.ResNet18_Weights.DEFAULT,
        "description": "A compact residual network that is a good starting point.",
    },
    "ResNet-50": {
        "builder": models.resnet50,
        "weights": models.ResNet50_Weights.DEFAULT,
        "description": "A deeper residual network for comparison with ResNet-18.",
    },
    "MobileNet V3 Large": {
        "builder": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.DEFAULT,
        "description": "A model designed for efficient inference on resource-limited devices.",
    },
    "EfficientNet-B0": {
        "builder": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.DEFAULT,
        "description": "A compact model that balances network depth, width, and resolution.",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device_name: str):
    config = MODEL_OPTIONS[model_name]
    weights = config["weights"]
    model = config["builder"](weights=weights)
    model.to(device_name)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]


def classify_image(image: Image.Image, model_names: list[str], top_k: int):
    predictions = {}
    errors = []

    for model_name in model_names:
        try:
            model, preprocess, labels = load_model(model_name, str(DEVICE))
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                probabilities = torch.softmax(model(input_tensor)[0], dim=0)

            scores, class_ids = torch.topk(probabilities, top_k)
            predictions[model_name] = [
                {"Rank": rank, "Label": labels[class_id], "Confidence": score}
                for rank, (score, class_id) in enumerate(
                    zip(scores.tolist(), class_ids.tolist()), start=1
                )
            ]
        except (OSError, RuntimeError) as exc:
            errors.append(f"{model_name}: {exc}")

    return predictions, errors


st.title("Interactive Image Classifier")
st.write(
    "Compare four pretrained PyTorch models on an uploaded image or a webcam photo. "
    "All models classify images into ImageNet's 1,000 categories."
)

with st.sidebar:
    st.header("Classification settings")
    with st.form("classification_settings"):
        selected_models = st.multiselect(
            "Models to compare",
            options=list(MODEL_OPTIONS),
            default=["ResNet-18", "MobileNet V3 Large"],
            help="The first selected model is not given special treatment; every model runs on the same image.",
        )
        top_k = st.slider("Predictions per model", min_value=1, max_value=10, value=5)
        run_classification = st.form_submit_button(
            "Classify image",
            type="primary",
            use_container_width=True,
        )

    st.caption(f"Inference device: **{DEVICE.type.upper()}**")
    with st.expander("About the models"):
        for name, config in MODEL_OPTIONS.items():
            st.markdown(f"**{name}** — {config['description']}")
        st.caption(
            "Confidence scores are each model's ImageNet class probabilities; "
            "they are not calibrated for direct accuracy comparisons."
        )

image_source = st.radio(
    "Choose an image source",
    ["Upload an image", "Take a webcam photo"],
    horizontal=True,
)

if image_source == "Upload an image":
    image_data = st.file_uploader(
        "Choose a JPG, PNG, or WebP image",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    image_data = st.camera_input("Take a photo")

if image_data is None:
    st.info("Upload an image or take a webcam photo to get started.")
    st.stop()

image_bytes = image_data.getvalue()
try:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
except (UnidentifiedImageError, OSError) as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

st.image(image, caption="Image to classify", width="stretch")
image_signature = sha256(image_bytes).hexdigest()

if run_classification:
    if not selected_models:
        st.error("Select at least one model before starting classification.")
    else:
        with st.spinner("Running the selected models..."):
            predictions, errors = classify_image(image, selected_models, top_k)
        st.session_state["last_classification"] = {
            "image_signature": image_signature,
            "models": selected_models,
            "top_k": top_k,
            "predictions": predictions,
            "errors": errors,
        }

result = st.session_state.get("last_classification")
if result is not None:
    settings_match = (
        result["image_signature"] == image_signature
        and result["models"] == selected_models
        and result["top_k"] == top_k
    )
    if settings_match:
        for error in result["errors"]:
            st.error(
                "Could not run a model. Check your connection and available memory. "
                f"{error}"
            )

        predictions = result["predictions"]
        if predictions:
            summary_rows = [
                {
                    "Model": model_name,
                    "Top prediction": model_predictions[0]["Label"],
                    "Confidence": model_predictions[0]["Confidence"],
                }
                for model_name, model_predictions in predictions.items()
            ]
            df_summary = pd.dataframe(summary_rows)
            df_summary["Confidence"] = df_summary["Confidence"].map(lambda x: f"{x:.2%}")

            st.subheader("Model comparison")
            st.dataframe(summary_rows, hide_index=True, width="stretch")
            st.bar_chart(
                pd.dataframe(summary_rows),
                x="Model",
                y="Confidence",
                color="Model",
                horizontal=True,
            )

            result_tabs = st.tabs(list(predictions))
            for tab, (model_name, model_predictions) in zip(
                result_tabs, predictions.items()
            ):
                with tab:
                    best_prediction = model_predictions[0]
                    st.metric(
                        "Top prediction",
                        best_prediction["Label"],
                        f"{best_prediction['Confidence']:.1%}",
                    )
                    st.markdown("#### Other likely labels")
                    for prediction in model_predictions:
                        st.progress(
                            prediction["Confidence"],
                            text=(
                                f"{prediction['Rank']}. {prediction['Label']} — "
                                f"{prediction['Confidence']:.1%}"
                            ),
                        )
        elif not result["errors"]:
            st.warning("No predictions were returned. Try selecting another model.")
    else:
        st.info("Settings changed. Click **Classify image** to run the selected models.")
