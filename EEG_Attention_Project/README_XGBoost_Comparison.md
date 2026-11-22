# XGBoost + Spectral Features：獨立模型比較（七組 E1 配置）

## 這份套件會做什麼？

- **只新增檔案**；不修改 `main.py`、`configs/baseline.yaml`、`src/`、`experiments/E1_region.py`、原本 EEGNet 或任何已訓練結果。
- 使用你已經保存的 `X_B0_uV`：2 秒、250 Hz、1–30 Hz B0 預處理 EEG，不重新濾波、不重新抽取事件。
- 使用 `configs/paths.yaml` 指向既有 EEG `.npz`、現有 `data/labels/rt30_labels.csv`（目前內容應為 RT5–35%）與 `data/splits/subject_folds.csv`。**實際讀取的是 CSV 內容，並非根據檔名推斷標籤，且會核對 `labeling_method=session_RT_rank_5to35_vs_65to95`。**
- **完整七組**：Full30、Forehead2、Frontal2、Central2、Temporal2、Parietal2、Occipital2，每組五折，正式執行共 **35 個 XGBoost 分類器**。
- 對所有 30 個 EEG 通道只計算一次特徵，快取在 `results/model_comparison/XGBoost/_cache/`；後續選擇不同組別直接取相應通道的特徵，無須重複讀 62 個大型 `.npz`。
- 以相同的 Train／Validation／Test 受試者分組訓練各模型；Early Stopping **只用 Validation Subjects**，Test Subjects 只在訓練完成後評估。
- 若 `results/E1/` 存在且使用相同標籤與測試受試者，額外輸出 XGBoost vs 既有 1D-CNN 比較；如果 E1 對照結果不存在，XGBoost 仍可獨立運行。

## 解壓縮後會新增

```text
EEG_Attention_Project/
├── configs/
│   └── xgboost.yaml
├── experiments/
│   └── model_comparison/
│       └── xgboost/
│           ├── __init__.py
│           ├── features.py
│           ├── model.py
│           └── runner.py
├── run_xgboost_comparison.py
└── README_XGBoost_Comparison.md
```

**原本的 `experiments/model_comparison/__init__.py` 已在 EEGNet 套件內；本套件不會覆蓋它。**

## 安裝

Windows PowerShell：

```powershell
cd F:\CSIE\EEG_Attention_Project
python -m pip install "xgboost>=1.7" numpy pandas scipy scikit-learn PyYAML matplotlib
```

原本的 PyTorch 不必移除；共用 `src/evaluation.py` 會 import `torch`，因此仍需維持原先環境。

## 執行順序

**1. 資料檢查（不訓練）**

```powershell
python run_xgboost_comparison.py --check
```

會印出受試者數量、Label 數量、五折設定，並抽取三個 EEG Epoch，確認頻譜特徵可計算。

**2. Smoke Test（不列入正式結果）**

```powershell
python run_xgboost_comparison.py --smoke-test
```

預設只跑 Forehead2 的第一折，從每位受試者每個類別挑少量 Epoch，最大 20 棵樹。結果寫到 `results/model_comparison/_smoke_test/XGBoost/`。

**3. 正式完整七組 × 五折**

```powershell
python run_xgboost_comparison.py
```

每組五折，不自行刪除組別。任何 Fold 已完成且設定一致，重新執行會自動跳過；未完成的 Fold 可以續跑。

**只執行指定組別或 Fold（可選）**

```powershell
python run_xgboost_comparison.py --group Forehead2
python run_xgboost_comparison.py --group Forehead2 --fold 1
```

之後執行完整指令即可把其餘組別與 Fold 補齊。

**只重建報表（可選）**

```powershell
python run_xgboost_comparison.py --summarize-only
```

設定或標籤改變後請備份 XGBoost 結果再重訓，不可以靠 `--summarize-only` 重新評估不同 Label。`--force` 只會清除並重跑所選 XGBoost Fold，**不會修改 E1/E2/E3/EEGNet**。

## 特徵定義與資料處理

每個 EEG channel 提取 **14 個固定特徵**：

- Welch PSD 對應的 Theta (4–8 Hz)、Alpha (8–13 Hz)、Beta (13–30 Hz) **log10 absolute band power**：3 個
- 這三個頻帶除以總 1–30 Hz power 的 **relative power**：3 個
- Theta/Alpha、Theta/Beta、Alpha/Beta 的 log10 power ratios：3 個
- 總 1–30 Hz power：1 個
- 時間域 RMS、Variance（取 log10）：2 個
- Hjorth Mobility、Hjorth Complexity：2 個

Welch 使用 250 Hz，`nperseg=250`、`noverlap=125`，故頻率解析度為 1 Hz；2 秒訊號內 1–4 Hz 的極低頻估計不夠穩定，本版不單獨輸出 Delta Power。所有特徵都是**每筆 Epoch 自行計算**，未以測試受試者資料學習任何全域轉換參數。XGBoost 不需要 CNN 的 Channel-wise Z-score；不額外標準化表格特徵。

完整 30ch 對應 30 × 14 = **420 個特徵**；每組 2ch 對應 2 × 14 = **28 個特徵**。不同通道數的特徵數自然不同，實驗比較的是各配置實際可使用的資訊與分類能力。

XGBoost 使用固定配置（見 `configs/xgboost.yaml`），不會在測試資料上調參或選擇分類閾值。Model selection / Early Stopping 只使用 `val` Subjects。

## 輸出

```text
results/model_comparison/XGBoost/
├── _cache/
│   ├── features.npy            # 七組共用的 30ch 特徵快取
│   ├── metadata.csv
│   └── cache_config.json
├── Full30/
│   ├── fold_1/ ... fold_5/
│   │   ├── best_model.json
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   ├── per_subject_metrics.csv
│   │   ├── feature_importance.csv
│   │   └── run_config.json
│   ├── summary.csv
│   ├── fold_summary.csv
│   ├── combined_predictions.csv
│   ├── combined_metrics.json
│   ├── per_subject_metrics.csv
│   ├── feature_importance_summary.csv
│   ├── confusion_matrix.csv
│   └── report.txt
├── Forehead2/                  # 其餘五組格式相同
├── Frontal2/
├── Central2/
├── Temporal2/
├── Parietal2/
├── Occipital2/
├── summary.csv                 # 七組一列一組，五折 mean BA / pooled BA
├── fold_summary.csv
├── model_comparison.csv        # 與原 E1 1D-CNN 相同 fold 的比較
├── comparison.png             # 七組 XGBoost / 1D-CNN 對照，數值標籤
├── xgboost_channels.png       # XGBoost 七組圖，數值標籤
└── report.txt
```

注意 `feature_importance_summary.csv` 是每個 Fold 內 feature gain 正規化後的平均，屬描述性重要度，**不能解釋成因果效果**。`mean_BA` 是每折等權平均，`pooled_BA` 是把所有測試 Trial 合併再計算，兩者可能不同。

## 實驗方法邊界

- 這個任務仍是 **RT 相對快／慢的 proxy label**，不等同絕對專注/不專注真值。
- 本程式不更換現有 RT5–35% 標籤，也不排除已保存的更多 Trial；如果標籤 CSV 仍為舊 RT30，必須先換成預期版。
- XGBoost **不是 E1–E3 的正式模型替代品**，只是獨立比較，不覆蓋任何現有訓練結果。
- 合成資料的測試準確率不是 nm000275 真實數據結果；真正 BA 需在你本機完成訓練後才能取得。
- 此版本針對 Python / XGBoost 的 scikit-learn estimator API。建議安裝 XGBoost 1.7+，實作在 XGBoost 3.1.3 執行過整合測試。
