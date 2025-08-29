# Branch Segmentation (UAV Tree Branches)

A simple, **single-file** pipeline for semantic segmentation on a branches dataset.  
It supports multiple architectures via `segmentation_models_pytorch` and includes:
- 80/10/10 split or K-Fold CV utilities
- Rich validation metrics (IoU, Dice, Precision/Recall, Thin-Structure IoU, Boundary-F1, Skeleton Similarity, CPR)
- Result saving to JSON/CSV/Excel (+ charts)
- Optional visualizations

> Data folders expected: `images/` (RGB inputs) and `marks/` (binary masks, same basenames as images, `.png`)

---

## Folder Layout

```
your-repo/
├─ The semantic segmentation on branches dataset.py
├─ images/                # your input images (jpg/jpeg/png)
├─ marks/                 # your masks (binary, 0/255), filenames match images but use .png
├─ results/               # auto-created for logs/exports
├─ README.md
├─ requirements.txt
└─ .gitignore
```

Mask file naming rule: for `images/ABC.jpg` expect `marks/ABC.png`.

---

## Quickstart

### 1) Create & activate an environment
- **Conda (recommended)**
  ```bash
  conda create -n branchseg python=3.10 -y
  conda activate branchseg
  ```

### 2) Install PyTorch
Install a build that matches **your CUDA**. See https://pytorch.org/get-started/locally/.  
For CPU-only, you can use:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3) Install the rest
```bash
pip install -r requirements.txt
```

### 4) Put your data
```
images/  -> RGB inputs
marks/   -> binary masks (0/255), same filenames with .png extension
```

### 5) Run training (single run, 80/10/10)
> The filename has spaces, so remember to quote it.

- **Windows (PowerShell/CMD)**
  ```bash
  python "The semantic segmentation on branches dataset.py"
  ```

- **macOS/Linux**
  ```bash
  python "The semantic segmentation on branches dataset.py"
  ```

By default the script:
- looks for `images/` and `marks/` in the repo root
- trains a model (default `unet`), batch size 128, image size 1024 (change inside the file if needed)
- writes results into `results/` (JSON/CSV/Excel) and saves the best model: `results/best_<model>.pth`

If you prefer to call functions explicitly, you can do:
```python
from importlib import util
spec = util.spec_from_file_location("branchseg", "The semantic segmentation on branches dataset.py")
module = util.module_from_spec(spec)
spec.loader.exec_module(module)

module.main(model_name="unet", save_dir="results", split_type="three_way")
# or: module.run_kfold_cross_validation(model_name="unet", save_dir="results", k=5)
```

### 6) Switch model backbones
Inside the script there is a `model_configs` dictionary you can change to select:
`unet`, `unet++`, `deeplabv3+`, `fpn`, `linknet`, `manet`, `pan`, and several encoder backbones (`resnet34/50`, `mobilenet_v2`, `densenet121`, `efficientnet-b0`, `convnext_base`, `vit_b_16`, `swin_base`, `mit_b0/b1/b2/...`).

### 7) Outputs
- `results/<model>_results.json / .csv / .xlsx`
- `results/<model>_training_curves.png`
- `results/<model>_prediction_results.png` (if enabled)
- `results/best_<model>_model.pth`

---

## Tips & Troubleshooting

- **CUDA OOM**: the code auto-reduces `BATCH_SIZE` on OOM. You can also lower `image_size` or manually reduce the batch.
- **Windows `num_workers`**: it's set conservatively; if you see DataLoader issues, keep `num_workers=0` or `2`.
- **Torch install for GPUs**: install the correct CUDA build first, *then* other packages.
- **Excel export**: requires either `xlsxwriter` or `openpyxl` (both included in `requirements.txt`).

---

## GitHub Desktop — step-by-step

1. **New repo**  
   - Open GitHub Desktop → *File* → *New repository…*  
   - Name: `branch-segmentation` (or any name)  
   - Local path: choose your folder containing this script + `images/` & `marks/`  
   - Click **Create repository**

2. **Add your files**  
   - Ensure these exist in the repo folder:  
     - `The semantic segmentation on branches dataset.py`  
     - `README.md`, `requirements.txt`, `.gitignore`  
     - (Optionally) `results/` (will be created automatically on first run)
     - `images/` and `marks/` (they are large—**DO NOT** commit big datasets; see below)

3. **.gitignore for datasets**  
   - We ignore large raw data by default (see `.gitignore`).  
   - Keep a small sample (e.g., `images/sample_*`, `marks/sample_*`) if you want demo data in the repo.

4. **First commit**  
   - In GitHub Desktop, check the files you want to commit.  
   - Write a summary: `Initial commit: training script + README`  
   - Click **Commit to main**

5. **Publish to GitHub**  
   - Click **Publish repository** (top bar) → choose visibility (Public/Private) → **Publish**.

6. **Push subsequent changes**  
   - Any time you edit the script / add results: *Commit to main* → **Push origin**.

> **Note** on datasets: for large datasets, consider Git LFS or keep data out of the repo and document how to obtain them.

---

## Citation / Acknowledgements

- Uses [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) backbones.
- Metrics and utilities adapted around the single-file training script.  
- Your uploaded script is referenced as the primary source in this repo.