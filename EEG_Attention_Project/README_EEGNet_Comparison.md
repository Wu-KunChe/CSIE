# EEGNet Model Comparison — 附加實驗使用說明

## 目的與原則

本套件只**新增**檔案，不會覆蓋或修改 `main.py`、`src/`、`experiments/E1_region.py`，也不會修改 `data/`、`results/baseline/` 或 `results/E1/`。

以**已完成的 RT5–35%** 標籤、相同 Subject-wise 5-fold（含固定的 train / val / test 受試者）、相同 2 秒 EEG 與 B0（1–30 Hz、250 Hz）為基礎，比較 **EEGNet** 與既有的 **1D-CNN**。

* EEGNet 先測三組：`Full30`、`Forehead2`（FP1/FP2）、`Frontal2`（F3/F4）。
* 五折每折重新訓練一個 EEGNet，共 **15 次訓練**。
* 不從新受試者的測試訊號計算任何標準化參數；訓練資料專屬統計量套用到 validation/test。
* `configs/baseline.yaml` 的**訓練參數**沿用（AdamW、batch size、60 epochs、Early Stopping、固定 0.5 threshold），`configs/eegnet.yaml` 另外定義 EEGNet 的模型架構。舊 baseline.yaml 的 labeling 文字若仍描述原 RT30，不會影響本程式：本程式讀取實際 `data/labels/rt30_labels.csv` 的**新標籤內容**。
* EEGNet 是參考 Lawhern et al. 2018 的 EEGNet-8,2 架構，使用 Temporal、Spatial Depthwise、Separable Convolution；這份是新的、隨機初始化的 PyTorch 實作，不是官方已訓練權重。
* 1D-CNN 讀取 `results/E1` 現有 Fold 數據作為對照，**不重新訓練**。

## 安裝檔案

把 `EEGNet_Standalone_Additions_20261104.zip` 解壓縮至：

```text
F:\CSIE\EEG_Attention_Project
```

解壓後只會**新增**：

```text
EEG_Attention_Project/
├── configs/
│   └── eegnet.yaml
├── experiments/
│   └── model_comparison/
│       ├── __init__.py
│       └── eegnet/
│           ├── __init__.py
│           ├── model.py
│           └── runner.py
├── run_eegnet_comparison.py
└── README_EEGNet_Comparison.md
```

`results/model_comparison/EEGNet` 會在執行後自動建立。**不需新增、修改或刪除**原有 `results/E1`、`data/labels`、`data/splits`、`src` 任何檔案。

## PowerShell 指令

```powershell
cd F:\CSIE\EEG_Attention_Project

# 若已安裝過依賴可跳過
python -m pip install numpy pandas scipy scikit-learn PyYAML torch matplotlib

# 1. 只核對設定、B0 Epoch、當前 RT Label、Subject Folds、既有 E1 對照資料
python run_eegnet_comparison.py --check

# 2. 極小型測試（僅第一折 Fp1/Fp2、每人少量 Trial、2 epochs）
python run_eegnet_comparison.py --smoke-test

# 3. 正式：三組 × 五折，共 15 次新 EEGNet 訓練
python run_eegnet_comparison.py
```

如果想先看第一折（**僅局部正式結果，不代表完整五折**）：

```powershell
python run_eegnet_comparison.py --group Forehead2 --fold 1
```

之後執行完整指令會自動略過已完成且設定一致的 Fold。

只重新產生整合報表（不重訓）：

```powershell
python run_eegnet_comparison.py --summarize-only
```

需要明確重訓某一個 EEGNet fold 時（**只覆蓋新實驗的該 Fold**）：

```powershell
python run_eegnet_comparison.py --group Forehead2 --fold 1 --force
```

指定使用 CPU 或 CUDA：

```powershell
python run_eegnet_comparison.py --device cpu
python run_eegnet_comparison.py --device cuda
```

如果要求 CUDA 但 `torch.cuda.is_available()` 是 `False`，會報錯；改用 `--device auto`（預設）可自動選用 CUDA 或 CPU。

## 結果結構

```text
results/
└── model_comparison/
    ├── _smoke_test/EEGNet/...             # 小型測試（不納入正式結果）
    └── EEGNet/
        ├── Full30/
        │   ├── fold_1/ ... fold_5/
        │   │   ├── best_model.pt
        │   │   ├── metrics.json
        │   │   ├── predictions.csv
        │   │   ├── per_subject_metrics.csv
        │   │   ├── training_history.csv
        │   │   ├── scaler.json
        │   │   └── run_config.json
        │   ├── summary.csv
        │   ├── fold_summary.csv
        │   ├── combined_predictions.csv
        │   ├── combined_metrics.json
        │   ├── per_subject_metrics.csv
        │   ├── confusion_matrix.csv
        │   └── report.txt
        ├── Forehead2/                      # 同上
        ├── Frontal2/                       # 同上
        ├── summary.csv                     # 三組五折結果
        ├── fold_summary.csv                # 所有個別 fold
        ├── model_comparison.csv            # EEGNet vs 舊 1D-CNN
        ├── comparison.png                  # 兩模型各通道 BA，含數值標籤
        ├── eegnet_channels.png             # EEGNet 三組，含數值標籤
        └── report.txt
```

* **`mean_BA`**：五折 BA 的算術平均；每折等權。
* **`std_BA`**：五折 BA 的樣本標準差，不是信賴區間。
* **`pooled_BA`**：合併所有 held-out 測試預測後重新計算的 BA；樣本等權。
* **`model_comparison.csv`** 的 `delta_BA_EEGNet_minus_CNN`：相同通道及 Fold 下，EEGNet 減去 1D-CNN 的平均 BA。
* 是否「更接近 Full30」請比較完整執行後輸出的 `[GAP]` 數據，而不是只看哪組 BA 較高。

## 硬體與資料完整性

* `Full30` 的暫存數列約需 0.9 GB 磁碟空間；逐 Session 讀取、用 NPY Memmap 取樣，**不把整組 30ch EEG 常駐 RAM**。暫存檔正常結束會自動清理。
* 只讀 `nm000275_epochs` 中現有的 `.npz`（`X_B0_uV`）；不會重新 Bandpass、重採樣或 RT Labeling。
* 程式會核對 `results/E1` 既有預測與現在 Label CSV 的 Trial ID、標籤、Fold；若不符則報錯，以免比較錯誤版本。
* 完整跑完後確認 `summary.csv` 中的 `all_folds_complete` 全部為 `True`。
* 因仍只有單一固定 Random Seed，模型差距很小時不宜直接宣稱顯著優劣。RT 標籤是相對快/慢的行為代理標籤，不能視為絕對注意力真值。

## 依據與測試範圍

EEGNet 基本概念：Lawhern et al. (2018), doi:10.1088/1741-2552/aace8c。

已在合成 EEG 資料上測試：資料與 E1 標籤對照、2ch/30ch forward、正式/Smoke 訓練、五折結果匯總、重跑跳過、合併混淆矩陣及數值標籤比較圖。也使用你之前的 RT5–35% Label CSV 與 E1 預測檔檢查所有 15,628 筆對應，**但本環境沒有原始 36 GB EEG，尚未以真實 EEG 訓練 EEGNet**。
