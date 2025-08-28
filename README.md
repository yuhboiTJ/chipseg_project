# Merchant Lab Microscope Automation

End-to-end system for microscope image capture, segmentation, and chip area calculation.

## Structure
- `scripts/` — training, TFLite conversion, and inference
- `data/` — folder structure with manifest.csv (datasets excluded)
- `models/` — model definitions (weights excluded by .gitignore)
- `requirements.txt` — Python dependencies

## Note
Large model files (`*.h5`, `*.tflite`) and full datasets are excluded from GitHub.
