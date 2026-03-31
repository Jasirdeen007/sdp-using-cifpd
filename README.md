# SDP Using CIFPD

This project refactors the original `SDP_using_CIFPD.ipynb` notebook into a repeatable training, testing, and inference workflow.

## What changed

- The original notebook logic is preserved, but moved into reusable Python code in [`src/sdp_pipeline.py`](/home/jasir/sdp/src/sdp_pipeline.py).
- A testing phase is now part of the project:
  - holdout evaluation with saved metrics
  - cross-validation summary
  - saved confusion matrix and ROC curve plots
  - lightweight unit tests for core preprocessing logic
- A Streamlit UI is available in [`app.py`](/home/jasir/sdp/app.py) to run the trained model on new inputs.
- The refactored training flow explicitly removes `rs` from candidate features because `defective` is created from `rs`. Keeping `rs` as an input would leak the target into the model.

## Project structure

- [`SDP_using_CIFPD.ipynb`](/home/jasir/sdp/SDP_using_CIFPD.ipynb): original notebook
- [`src/sdp_pipeline.py`](/home/jasir/sdp/src/sdp_pipeline.py): shared training and inference pipeline
- [`train_model.py`](/home/jasir/sdp/train_model.py): model training entry point
- [`evaluate_model.py`](/home/jasir/sdp/evaluate_model.py): print saved evaluation summary
- [`check_gpu.py`](/home/jasir/sdp/check_gpu.py): confirm whether PyTorch can see CUDA
- [`app.py`](/home/jasir/sdp/app.py): Streamlit UI for inference
- [`tests/test_sdp_pipeline.py`](/home/jasir/sdp/tests/test_sdp_pipeline.py): smoke tests for preprocessing logic

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want GPU-enabled PyTorch, install a CUDA wheel instead of the CPU-only default. On March 31, 2026, PyTorch's official install page shows CUDA-specific pip wheels for Linux, including a `cu118` example:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Pick the CUDA build that matches your NVIDIA driver support from the official PyTorch selector:

- https://pytorch.org/get-started/locally/

After installation, verify GPU visibility:

```bash
python3 check_gpu.py
```

Train the model:

```bash
python3 train_model.py --csv /path/to/Eclipse.csv --artifacts models --reports reports
```

Training writes:

- `models/classifier.joblib`
- `models/metadata.json`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/*_feature_ranking.csv`

## Testing phase

1. Unit tests:

```bash
pytest -q
```

2. Review saved evaluation:

```bash
python3 evaluate_model.py --artifacts models
```

3. Inspect charts in `reports/`.

## Run the UI

```bash
streamlit run app.py
```

The UI reads the selected feature columns from `models/metadata.json`, so it automatically matches the trained model.

## GPU notes

- RoBERTa embeddings use CUDA automatically when `torch.cuda.is_available()` is `True`.
- XGBoost now uses `device="cuda"` automatically when CUDA is available, and the pipeline can keep the embedding matrix on GPU when `cupy-cuda12x` is installed.
- You can force XGBoost device selection with `SDP_XGB_DEVICE`, for example:

```bash
SDP_XGB_DEVICE=cpu python3 train_model.py --csv /path/to/Eclipse.csv
```

Official references:

- PyTorch install selector: https://pytorch.org/get-started/locally/
- XGBoost GPU support: https://xgboost.readthedocs.io/en/stable/gpu/
- XGBoost recommends GPU data structures with `device="cuda"` and `QuantileDMatrix`-backed training for best GPU execution: https://xgboost.readthedocs.io/en/stable/gpu/

## Recommended next improvements

- Replace the current label definition if you want true defect prediction earlier in the lifecycle. Deriving the label from resolution status is convenient, but it limits how realistic the prediction task is.
- Add a separate validation dataset from another project or time period. A single random split can still overestimate generalization.
- If training time becomes an issue, cache embeddings to disk so you do not recompute RoBERTa vectors on every run.
