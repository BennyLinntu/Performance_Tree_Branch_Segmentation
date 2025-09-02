# Branch Segmentation (UAV Tree Branches)

A simple, **single-file** pipeline for semantic segmentation on a branches dataset.  
It supports multiple architectures via `segmentation_models_pytorch` and includes:

- 80/10/10 split or K-Fold CV utilities  
- Rich validation metrics (IoU, Dice, Precision/Recall, Thin-Structure IoU, Boundary-F1, Skeleton Similarity, CPR)  
- Result saving to JSON/CSV/Excel (+ charts)  
- Optional visualizations  

> Data folders expected: `images/` (RGB inputs) and `marks/` (binary masks, same basenames as images, `.png`)

---

## 📂 Folder Layout

```
your-repo/
├─ The semantic segmentation on branches dataset.py
├─ images/                # input images (jpg/jpeg/png)
├─ marks/                 # binary masks (0/255), filenames match images but use .png
├─ results/               # auto-created for logs/exports
├─ README.md
├─ requirements.txt
└─ .gitignore
```

Mask file naming rule: for `images/ABC.jpg` expect `marks/ABC.png`.

---

## 🚀 Quickstart

### 1) Create & activate environment
```bash
conda create -n branchseg python=3.10 -y
conda activate branchseg
```

### 2) Install PyTorch
Install a build that matches **your CUDA**: [PyTorch install guide](https://pytorch.org/get-started/locally/).  
For CPU-only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3) Install requirements
```bash
pip install -r requirements.txt
```

### 4) Put your data
```
images/  -> RGB inputs
marks/   -> binary masks (0/255), same filenames with .png extension
```

### 5) Run training (single run, 80/10/10)
```bash
python "The semantic segmentation on branches dataset.py"
```

By default:
- Looks for `images/` and `marks/` in the repo root
- Trains `unet` (default), batch size 128, image size 1024
- Saves results to `results/` and best model as `results/best_<model>.pth`

---

## 🔄 Switch Models / Backbones

Edit the `model_configs` dictionary inside the script to choose:
- Architectures: `unet`, `unet++`, `deeplabv3+`, `fpn`, `linknet`, `manet`, `pan`
- Backbones: `resnet34/50`, `mobilenet_v2`, `densenet121`, `efficientnet-b0`, `convnext_base`, `vit_b_16`, `swin_base`, `mit_b0/b1/b2/...`

---

## 📊 Outputs

- `results/<model>_results.json / .csv / .xlsx`  
- `results/<model>_training_curves.png`  
- `results/<model>_prediction_results.png` (if enabled)  
- `results/best_<model>_model.pth`  

---

## 🖼 Example Prediction Results

Here are some qualitative results from **U-Net++** on the branches dataset:

![U-Net++ Results](results/unet++_prediction_results.png)

---

## ⚠️ Tips & Troubleshooting

- **CUDA OOM**: auto-reduces `BATCH_SIZE`. You can also lower `image_size` or batch manually.  
- **Windows DataLoader**: set `num_workers=0` or `2` if issues.  
- **Excel export**: needs `xlsxwriter` or `openpyxl` (already in `requirements.txt`).  
- **Datasets**: Large raw data should not be committed. Keep only small samples if needed.  

---

## 📌 Citation / Acknowledgements

- Uses [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) backbones.  
- Metrics/utilities adapted around this single-file training script.  
