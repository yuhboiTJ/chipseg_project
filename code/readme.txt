Microchip Segmentation - code
=============================

This folder contains the code that trains a U-Net to segment the chip
region in the microchip images. A small classical step (HSV-based shadow
removal) refines the U-Net's mask to drop the chip's shadow on the
background paper, and the resulting seedable mask is converted to mm^2
using a pixel-to-mm^2 factor derived from the known ground truth areas.
The per-chip predictions are then compared to the ground truth.

Setup
-----
Tested with Python 3.14 on Windows. CPU PyTorch is fine, GPU is faster.

    python -m pip install -r requirements.txt

Folder layout
-------------
    code/
      src/                  the package code
        dataset.py          PyTorch dataset + albumentations transforms
        model.py            U-Net implementation
        train.py            training loop (BCE + Dice loss)
        calibration.py      derives mm^2 per pixel from labeled masks
                            and the 24 ground truth area values
        stage2_classical.py shadow refinement (HSV) and optional defect
                            removal inside the predicted chip mask
        eval.py             IoU / Dice on the val split, plus per-chip
                            area MAE / R^2 across all 168 images
        visualize.py        plotting helpers
      scripts/              command line entry points
        train_model.py            train the U-Net
        evaluate.py               IoU/Dice + per-chip area metrics
        predict_image.py          single-image prediction
        save_sample_overlays.py   make qualitative overlay PNGs
        predict_test_dataset_1.py inference on the no-ground-truth test set
        finalize_report.py        substitute metrics into REPORT.txt
        live_demo.py              live webcam/USB-capture demo
        build_notebook.py         rebuild the notebook from source
      notebooks/
        01_full_pipeline.ipynb    end-to-end run with figures inline
      outputs/              generated artifacts
        checkpoints/best.pt       trained U-Net weights
        figures/                  training_curves, area_scatter, overlays
        metrics.json              final numbers used in REPORT.txt
        training_history.csv      per-epoch loss/dice/iou
        test1_predictions.csv     per-image predicted mm^2 on Test_1

How to train and evaluate
-------------------------
Each manual mask goes into ../Training_dataset_2_ground_truth_masks/
with a filename containing the chip number and background number. The
matcher accepts .png/.jpg/.tif and is forgiving about case and the
common 'co' typo for 'c0'. Example: c01_bg1_mask.tif works.

From the code/ folder:

    python scripts/train_model.py --epochs 40 --base-channels 16
    python scripts/evaluate.py
    python scripts/finalize_report.py

If you only want to see results without retraining, evaluate.py loads
whatever best.pt is already in outputs/checkpoints/.

To run on a single image and print the predicted area:

    python scripts/predict_image.py --image ../Training_dataset_2/C03/C03_Bg2_z1.png

Live demo
---------
The live_demo.py script opens a video device and shows the predicted
seedable area in mm^2 on each frame:

    python scripts/live_demo.py
    python scripts/live_demo.py --device 1     # second camera

Hotkeys: q quit, s save current frame, d toggle defect removal,
1/2/3 switch between 1.0x / 0.75x / 2.0x zoom calibration. The 0.75x
and 2.0x factors are scaled mathematically from the 1.0x calibration;
for real accuracy capture a calibration image at each zoom and update
the factors in scripts/live_demo.py (load_zoom_calibration).

Raspberry Pi 4 deployment
-------------------------
The project is portable Python. The same code that trains and evaluates
on a desktop runs on a Pi with no changes other than installing the
dependencies.

  # on the Pi (one-time setup)
  sudo apt update
  sudo apt install -y python3-pip python3-opencv libopenblas-dev
  pip3 install --user torch numpy matplotlib scikit-image
  pip3 install --user pillow pandas scikit-learn

Then copy this code/ folder plus outputs/checkpoints/best.pt to the Pi
(USB stick, scp, or git clone if you push the repo to GitHub) and run:

  python3 scripts/live_demo.py

Notes for the Pi:
  - The trained model is small (about 8 MB on disk, 1.94 million params)
    and runs on the Pi's CPU at a few frames per second at 720x576.
  - For faster inference, the model can be converted to TensorFlow Lite
    via ONNX. We have not done that conversion in this code base; the
    script that would do it is left as future work.
  - The microscope camera should be plugged into the Pi as a USB capture
    device. Pi Camera v2/v3 also works through libcamera if you adjust
    the cv2.VideoCapture call.
  - For different zoom levels you must tell the live_demo which zoom is
    set, since the microscope cannot tell the Pi this information.

Notes on design choices
-----------------------
- One U-Net for chip detection plus a classical HSV shadow-refinement
  step. The original plan was two U-Nets (chip then seedable); we
  replaced the second one with classical shadow handling because the
  user's training masks already encode what is seedable, so no second
  learned stage is needed.

- The labeled pairs are split chip-level (entire chips held out for
  validation), not random per-mask, so the validation metric measures
  generalization to truly unseen chips. Default holdout is C07, C12,
  C18, C20, chosen to span the area range.

- Heavy augmentation (flips, affine, brightness, noise) compensates for
  the small labeled set. RandomRotate90 is intentionally omitted because
  the input is rectangular and a 90-degree rotation breaks batching.

- Input is resized to 288x384 for the network (native is 576x720). The
  predicted mask is upsampled back to native resolution before measuring
  pixel area for calibration.

- Pixel-to-mm^2 calibration is fit per labeled image (chip ground-truth
  area divided by labeled mask pixel count). The mean across all labeled
  chips is the global calibration factor.
