from __future__ import annotations

import base64
import json

import pandas as pd
import streamlit as st

from src.sdp_pipeline import DefectPredictor


FIELD_LABELS = {
    "sd": "Short Description",
    "bs": "Bug Severity",
    "bsr": "Bug Severity Rating",
    "pr": "Priority",
    "sev": "Severity",
    "os": "Operating System",
    "component": "Component",
    "product": "Product",
    "version": "Version",
    "platform": "Platform",
}


def format_field_label(column: str) -> str:
    return FIELD_LABELS.get(column, column.replace("_", " ").strip().title())


def build_history_csv(history_df: pd.DataFrame) -> bytes:
    return history_df.to_csv(index=False).encode("utf-8")


def build_history_download_link(history_df: pd.DataFrame) -> str:
    csv_bytes = build_history_csv(history_df)
    encoded = base64.b64encode(csv_bytes).decode("utf-8")
    return (
        f'<a href="data:text/csv;base64,{encoded}" '
        f'download="session_predictions.csv">Download session predictions as CSV</a>'
    )


@st.cache_resource
def load_predictor(artifact_dir: str) -> DefectPredictor:
    return DefectPredictor(artifact_dir)


def reset_form_state() -> None:
    st.session_state["prediction_form"] = {}


st.set_page_config(page_title="SDP Defect Predictor", page_icon=":clipboard:", layout="wide")
st.title("Software Defect Prediction")
st.caption("Run the trained CIFPD pipeline on new issue fields.")

artifact_dir = st.sidebar.text_input("Artifact directory", value="models")

try:
    predictor = load_predictor(artifact_dir)
except FileNotFoundError:
    st.error("Model artifacts were not found. Train the model first with `train_model.py`.")
    st.stop()

st.sidebar.subheader("Model metadata")
st.sidebar.text(json.dumps(predictor.metadata, indent=2))
st.sidebar.caption(
    f"PyTorch device: {predictor.metadata.get('torch_device', 'unknown')} | "
    f"XGBoost train device: {predictor.metadata.get('xgboost_device', 'unknown')} | "
    f"XGBoost inference device: {getattr(predictor, 'inference_xgboost_device', 'unknown')}"
)
decision_threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction_done" not in st.session_state:
    st.session_state.last_prediction_done = False

st.subheader("Issue Input")
with st.form("prediction_form", clear_on_submit=True):
    values = {}
    for column in predictor.selected_columns:
        if column.lower() in {"sd", "summary", "description"}:
            values[column] = st.text_area(format_field_label(column), height=140)
        else:
            values[column] = st.text_input(format_field_label(column))
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([values])
    result = predictor.predict(input_df).iloc[0]
    probability = float(result["defect_probability"])
    predicted_label = "Defective" if probability >= decision_threshold else "Non-defective"
    st.session_state.last_prediction_done = True

    st.subheader("Prediction")
    st.write(f"Predicted label: **{predicted_label}**")
    st.write(f"Defect probability: **{probability:.2%}**")
    st.caption(f"Threshold used for this prediction: {decision_threshold:.2f}")
    st.success("Prediction completed. You can enter the next issue now.")

    st.subheader("Constructed intent_text")
    st.text_area("Constructed intent text", value=result["intent_text"], height=120, disabled=True)

    st.subheader("Submitted fields")
    submitted_fields = pd.DataFrame(
        {
            "Field": [format_field_label(column) for column in input_df.columns],
            "Value": [input_df.iloc[0][column] for column in input_df.columns],
        }
    )
    st.table(submitted_fields)

    session_record = input_df.copy()
    session_record["intent_text"] = result["intent_text"]
    session_record["predicted_label"] = predicted_label
    session_record["defect_probability"] = probability
    session_record["decision_threshold"] = decision_threshold
    st.session_state.prediction_history.append(session_record.iloc[0].to_dict())

if st.session_state.last_prediction_done:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict next issue"):
            st.session_state.last_prediction_done = False
    with col2:
        if st.button("Reset form"):
            st.session_state.last_prediction_done = False
            reset_form_state()

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.subheader("Session Prediction History")
    st.table(history_df)
    if st.button("Clear session history"):
        st.session_state.prediction_history = []
    st.markdown(build_history_download_link(history_df), unsafe_allow_html=True)
