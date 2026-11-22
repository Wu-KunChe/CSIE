# 4 秒 EEG — 1D-CNN Random Search 與七組通道比較

## 1. 程式用途

這是一個**同一支主程式跑完的實驗**：

1. 只用 **4 秒 Full30**（`X_B0_uV`, 250 Hz × 4 = 1000 samples），建立 **50 個不同組合**，其中 **1 組是原始 Baseline、49 組從指定分布隨機抽取**；
2. 每組使用 **3-fold Subject-wise Scoring**（27 位受試者分三個輪流評分的群組），共 **50 × 3 = 150 次獨立 CNN 訓練**；
3. 按 **3 折平均 Balanced Accuracy** 選出一組超參數，若剛好相同用 Macro-F1 決定；
4. **固定完全相同的超參數**，使用原本 `data/splits/subject_folds.csv` 的外層 Subject-wise 五折分組，重新訓練 Full30 與六組 2-channel，共 **7 × 5 = 35 次**；
5. 最終合計 **185 次 CNN 訓練**（不計額外的 Smoke Test）。不沿用先前模型 checkpoint，也不改動 E1、Window Length 或 EEGNet 的既有結果。

**注意**：Random Search 的「3-fold」是用來選超參數；正式「5-fold」才是做通道縮減比較。兩者用途不同。

## 2. 新增檔案與位置

將 ZIP **解壓縮到** `F:\CSIE\EEG_Attention_Project\`；不要把程式放到上一層 CSIE，因為這次是模型實驗，不是原始 EEG 切分。

```text
EEG_Attention_Project/
  configs/hyperparameter_search.yaml
  experiments/hyperparameter_search/__init__.py
  experiments/hyperparameter_search/runner.py
  scripts/run_hyperparameter_search.py
  README_Random_Search_4sec.md
```

依賴你專案中**已存在**的 `src/model.py`、`src/trainer.py`、`src/data_loader.py`、`src/preprocessing.py`、`src/evaluation.py`、`src/config.py`、`configs/baseline.yaml`、`configs/channels.yaml`。上述檔案完全不修改。

資料路徑**固定為**：

```text
EEG_Attention_Project/epochs/4sec/sub-XX_ses-YY_unlabeled.npz
EEG_Attention_Project/data/labels/window_length/4sec/rt30_labels.csv
EEG_Attention_Project/data/splits/subject_folds.csv
```

原本 `data/labels/rt30_labels.csv` 是舊 RT5–35，**不會讀取，也不會寫入**。

## 3. 安裝與使用

```powershell
cd F:\CSIE\EEG_Attention_Project
python -m pip install numpy pandas scipy scikit-learn pyyaml torch matplotlib

# 僅檢查路徑、Labels、原有五折、第一份 NPZ 形狀；不訓練
python scripts\run_hyperparameter_search.py --check

# 強烈建議先做小規模測試：1 組參數 × 3 折 + 2 組通道 × 1 折、每次 2 Epoch
# 產生獨立的 random_search_4sec_SMOKE，不影響正式結果
python scripts\run_hyperparameter_search.py --smoke-test

# 正式：50 × 3 + 7 × 5 = 185 次訓練
python scripts\run_hyperparameter_search.py
```

程式會顯示每組 Candidate、每折 Train／Early-stop／Scoring 筆數、BA、Macro-F1、Best Epoch 等。可用 `--device cuda` 強制 NVIDIA GPU，或 `--device cpu` 強制 CPU；`auto` 會自動選用可用的 CUDA。

中途停止後，再執行**相同命令**，已完成且來源指紋相同的任務會自動 `[SKIP]`；缺少完成標記的折會重新執行。若來源程式／設定／標籤／NPZ 發生改變，會拒絕混用舊結果，請另存舊結果資料夾後重新執行。

可分開跑：

```powershell
# 先只做 150 次搜尋
python scripts\run_hyperparameter_search.py --phase search

# 日後再做 35 次正式比較（要求前 150 次已全部完成）
python scripts\run_hyperparameter_search.py --phase final
```

**注意**：`--smoke-test` 的結果只是確認程式可運作，不能視為研究成績。

## 4. 七個調整項目

| 超參數 | 分布 | 範圍或候選 |
|---|---|---|
| learning_rate | log-uniform | 1e-4 ～ 3e-3 |
| weight_decay | log-uniform | 1e-6 ～ 1e-2 |
| dropout | uniform | 0.10 ～ 0.50 |
| conv_filters | categorical | `[16,32]`、`[32,64]`、`[64,128]` |
| kernels | categorical（兩個卷積核分別抽樣） | K1 `[25,51,101,201]`；K2 `[7,15,31]` |
| batch_size | categorical | 32、64、128 |
| early_stopping_patience | 離散整數 | 6 ～ 12（含） |

`kernels` 是**一個架構參數（含兩層的 kernel）**，合計前述七項。預設採用 `include_baseline_as_one_candidate: true`：`trial_001` 是原始 Baseline，`trial_002`～`trial_050` 由固定 `search_seed=20261109` 隨機抽取。Baseline **算在 50 組裡，不增加訓練次數**，且能用同一套 3-fold 直接比較調參後的模型是否有改善。若要改成 50 組全部隨機抽取，可在第一次正式訓練前設為 `false`。

固定不變：模型仍是原本 `CNN1D` 兩層卷積 + ReLU + MaxPool + AdaptiveAvgPool + Dropout + Linear；Optimizer = AdamW；Max Epochs = 60；決策閾值 0.5；Class Weight 僅由 Training Fold 計算；每個通道的 Z-score 僅使用當折 Training Subjects 計算；B0 預處理 1–30 Hz 已存在 NPZ 中，**不再重新濾波**；RT 不增加額外排除閾值。

## 5. 為何搜尋一次仍要分 Train／Early-stop／Score？

每組參數有 3 個 Search Fold，每個 Fold **只訓練一個新 CNN**：

- **Score Subjects**：約三分之一受試者，完全不參與梯度更新、Scaler 或 Early Stopping，用於替這組參數計算 BA；3 個 Fold 輪流涵蓋所有 27 位受試者；
- **Early-stop Subjects**：從剩下的受試者中抽約 20%（約 4 位），只決定最佳 Epoch；
- **Training Subjects**：其餘約 14～15 位，負責權重更新與 Scaler Fit。

這不是多訓練一次；**一個 Fold = 一次 CNN 訓練**。跟單純拿同一批 Validation Subjects 同時決定最佳 Epoch 和參數分數相比，這種方式較能區分「挑 Epoch」與「挑超參數」。

正式五折完全使用**原本的 `subject_folds.csv` 的 train/val/test roles**，保留舊版 E1 的評估流程（含 Early Stopping）。不會重新隨機切分 Epoch，也不會讓同一受試者同時出現在同折 Train、Val、Test。

## 6. 輸出內容

存到：`results/diagnostics/random_search_4sec/`。

```text
random_search_4sec/
  experiment_manifest.json        # 輸入與程式的 fingerprint；中斷續跑用
  search_candidates.csv           # 50 組參數（標記 baseline 或 random）
  search_subject_roles.csv        # 3-fold 對應的 train/early_stop/score 受試者
  search_summary.csv              # 各組參數的 3-fold 平均 BA、MacroF1、完成進度
  search_ranked.csv               # 完整搜尋完成後的成績排序
  best_params.json                # 選中且固定的唯一一組超參數
  search/
    trial_001/
      fold_1/{metrics.json,training_history.csv,scaler.json,run_config.json,...}
      fold_2/...
      fold_3/...
    ...
    trial_050/...
  final/
    Full30/fold_1/{best_model.pt,metrics.json,predictions.csv,training_history.csv,...}
    ...
    Forehead2/fold_1/...
    Frontal2/...
    Central2/...
    Temporal2/...
    Parietal2/...
    Occipital2/...
    summary.csv
    fold_summary.csv
    subject_summary.csv
    channel_comparison.csv
    channel_comparison.png
  report.txt
```

`channel_comparison.csv` 最值得關注：

- `mean_BA`：各組 **原本五折平均的測試 BA**；
- `BA_delta_vs_Full30_pp`：相較 Full30 相差幾個百分點；
- `BA_retained_vs_Full30_pct`：BA 直接相除（例如 0.66／0.69）；
- `chance_corrected_retention_pct`：以超過隨機基準 50% 的 BA 增益做比率 `(BA_group-.5)/(BA_Full30-.5)*100`，更適合量化鑑別能力保留比例。Full30 BA 不高於 0.5 時不定義此比例。

## 7. 研究限制（重要）

**這是開發階段的 Random Search + 原有五折通道比較，不是完整 Nested CV。**

由於 50 組參數是用這 27 位受試者的 3-fold score 選出，之後正式五折仍測試這 27 位，雖然每一次訓練都符合 Subject-wise 分離，但**正式五折 BA 不能視為對超參數選擇完全獨立的無偏性能估計**。你目前目標是建立基礎 CNN 再比較通道縮減，這樣可作開發階段研究，但不能直接宣稱對新受試者的最終泛化成績；正式論文結論最好以未用於搜尋的獨立資料集或 Nested CV 補強。

此外，4 秒也曾依先前測試表現被選定，該選擇同樣會對後續報告的最終效能估計帶來潛在樂觀偏差。
