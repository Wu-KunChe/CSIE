# E3 P2 — 官方 Preprocessed EEG：驗證、4 秒切窗與固定 CNN 訓練

## 0. 先看：你現在提供的 NEMAR 目錄**還不是官方 P2**

`F:\CSIE\nm000275_v1.0.0` 目前是 **Raw/BIDS**，包含 `sub-XX/ses-XX/eeg/*.set`、`sourcedata/raw/*.set` 及 `sourcedata/figshare_preproc_7666055.json`。最後一個 JSON **僅是官方預處理資料的來源資訊，不是 EEG 訊號**。

官方 P2 是另外發布的 Figshare preprocessed dataset：
https://doi.org/10.6084/m9.figshare.7666055.v3

先從此處取得真正的 EEG 檔案（一般為原始錄製檔名的 `.set.zip`），放在例如：

```text
F:\CSIE\
├── EEG_Attention_Project\
├── nm000275_v1.0.0\                 # 目前已有的 Raw/BIDS
└── nm000275_preprocessed\           # 需另外下載官方 P2
    ├── s01_051017m.set.zip
    ├── s01_060227n.set.zip
    └── ... 其餘錄製檔 ...
```

可讓官方檔案分散於子資料夾中；程式會遞迴尋找 `.set` 或 `.set.zip`，並使用 NEMAR `sub-XX_sessions.tsv` 的 `original_filename` 對應各 Session。不要把 Raw 的 `.set` 複製到這個資料夾冒充 P2。

**如果官方 P2 不存在、或事件對齊無法驗證，程式會停止。** 不會自動下載數十 GB，不會把 Raw 當 P2，也不會偷偷將新標籤重新排序。

## 1. 安裝

將本 ZIP 解壓至 `F:\CSIE\EEG_Attention_Project`。只會新增檔案，不會改動既有 `src/`、`epochs/4sec/`、`data/labels/window_length/4sec/`。

```text
EEG_Attention_Project/
├── README_E3_P2_Official.md
├── configs/e3_p2_official.yaml
├── experiments/E3_P2/
│   ├── __init__.py
│   ├── prepare.py
│   └── train.py
└── scripts/
    ├── prepare_e3_p2_official.py
    ├── run_e3_p2_train.py
    └── relocate_e2_results.py
```

依賴：`numpy, pandas, scipy, mne`（資料準備）；`torch, scikit-learn, PyYAML`（後續訓練；與原 E1 相同）。

## 2. P2 檢查（只檢查，不建立 EEG Epoch）

```powershell
cd F:\CSIE\EEG_Attention_Project
python scripts\prepare_e3_p2_official.py --check
```

如果官方資料放在其他位置，例如 `F:\CSIE\P2_downloads`：

```powershell
python scripts\prepare_e3_p2_official.py --preproc-root "F:\CSIE\P2_downloads" --check
```

報告輸出至：

```text
results/E3/P2_preflight/
├── preflight.json
├── session_check.csv
└── README.txt
```

檢查內容：

1. B0 4 秒標籤檔、62 個 Session 的對應關係、五折受試者分組，以及 4 秒 NPZ/Manifest 的 Trial ID 順序。
2. 用 NEMAR 的 `original_filename` **精確**對應官方 `*.set(.zip)`，不靠目錄排序猜測。
3. 原 Raw/BIDS 與官方 P2 均為 **500 Hz**，連續錄製 sample 數相同；官方至少包含完整 30 個標準 EEG 通道且名稱可對應。
4. `251/252` 事件數量、事件方向、順序及觸發時間（容許最大 2 個 500Hz samples 的誤差）與原始 RT30 Trial 一一對應。
5. 原 B0 的每個有效 4 秒視窗，官方訊號都有對應樣本；不能跨越 P2 記錄中的 `boundary`。
6. 代表性訊號的數值有限、單位尺度合理，且不能與 Raw 完全相同（防止誤用 Raw）。

**如果官方前處理刪除時間片段、改變事件索引或取樣率，程式會 FAIL 並保留失敗報告，不會硬套原始標籤。** 如有失敗，請先提供 `session_check.csv` 與 `preflight.json`；需要根據實際變動建立可靠事件對應策略，不能簡單用相同秒數切窗。

## 3. 通過後產生 P2 4 秒 Epoch

保留所有 30 通道（利於未來 Full30 作為次要對照）：

```powershell
python scripts\prepare_e3_p2_official.py --run
```

**省空間的選擇：只保存 FP1、FP2（推薦目前 Forehead2 主實驗）**：

```powershell
python scripts\prepare_e3_p2_official.py --run --forehead-only
```

這兩個模式不可混在同一個輸出資料夾。如果需要兩個版本，應事先安排不同路徑或保存獨立副本；程式不會覆蓋已發布的 P2 輸出。

無論哪種模式，P2 採用原本 B0 的 **4 秒事件前視窗、事件限制 warmup 的 causal Butterworth 1–30 Hz、500→250 Hz 重採樣**，因此維持每通道 1000 點。官方本身的 FIR/去偽跡處理已存在於輸入，不會用此程式重新執行官方 AAR。

**重要：保留原 4 秒 RT0–30 標籤**，原先已排除的事件不會被補回；不根據官方檔案的剩餘 Trial 再次排名。P2 和 P0 的檔案名稱及 x_row 順序相同，但 P2 放在新資料夾。

成功時：

```text
EEG_Attention_Project/
├── epochs/
│   ├── 4sec/                              # P0，原檔不動
│   └── P2_official/
│       └── 4sec/
│           ├── sub-01_ses-01_unlabeled.npz
│           ├── sub-01_ses-01_manifest.csv
│           ├── sub-01_ses-01_p2_done.json
│           ├── ... 其他 61 個 Session ...
│           ├── all_trials_UNLABELED.csv
│           ├── P2_provenance.json
│           └── extraction_report.txt
├── data/labels/
│   ├── window_length/4sec/                # P0，原檔不動
│   └── E3/P2_official/4sec/
│       ├── rt30_labels.csv               # 原標籤逐位元複製
│       ├── rt30_summary.csv
│       └── P2_provenance.json
└── results/E3/P2_preflight/              # P2 資料查驗報告
```

P2 NPZ 的核心 key：`X_P2_uV`、`channel_names`、`trial_index`、`onset_s`、`rt_s`、`sfreq_P2`。原 `src.data_loader.iter_session_arrays(..., source_key='X_P2_uV', expected_samples=1000)` 可直接讀取。

**空間估算：**原先約 22,777 筆有效 4 秒 Epoch，30ch × 1000 float32 純資料約 2.73 GB；2ch 約 182 MB（不含 NPZ metadata/壓縮差異）。處理 `.set.zip` 時還需暫存解壓空間。若已有 P2 輸出，程式拒絕覆蓋；中途中斷的 Session 可依 `*_p2_done.json` 在暫存區續做。

## 4. 準備好之後，才進行 P2 固定 1D-CNN 五折訓練（可選）

```powershell
python scripts\run_e3_p2_train.py --check
python scripts\run_e3_p2_train.py --smoke-test
python scripts\run_e3_p2_train.py
```

若要指定 Fold，例如：

```powershell
python scripts\run_e3_p2_train.py --fold 1 --fold 2
```

- 完全沿用 `configs/baseline.yaml` 的**原始** CNN、原始訓練參數、五折分組。
- 只讀 P2 的 FP1/FP2、4s×250 Hz；不改 Label，不使用 Random Search。
- 每 Fold 只用 TRAIN 擬合標準化；VAL 作 Early Stopping；TEST 只負責評估。
- `results/E3/P2_official/fold_1...fold_5/` 保存模型、Scaler、Loss、預測、每受試者指標；`results/E3/P2_official/summary.json`、`fold_summary.csv`、`subject_summary.csv` 等匯總。
- 支援完成 Fold 的續跑；Smoke 單獨存 `results/E3/P2_official_SMOKE/`。
- 需要與 P0 比較時，P0 必須對應**相同的** 13,604 筆測試 Trial；此工具不會重新訓練 P0。

## 5. 修正 E2 結果目錄（與 P2 完全獨立）

原先 E2 產生在 `results/diagnostics/E2_frequency_4sec`；你要的結構應是 `results/E2/E2_frequency_4sec`。

此 ZIP 另附一支**可選的安全遷移腳本**，先預覽：

```powershell
python scripts\relocate_e2_results.py
```

確認無誤後：

```powershell
python scripts\relocate_e2_results.py --apply
```

它會移動 **已完成的 E2 結果**並將 `configs/e2_frequency.yaml` 的 `results_root` 改成 `results/E2/E2_frequency_4sec`。它不修改任何 E1／E2 訓練結果或 P2 檔案；若目標已存在會拒絕覆寫。

注意：E2 程式將 YAML 檔案雜湊當作續跑識別碼，因此搬移和修改路徑後**不要直接重新跑原 E2**；這次既然已完成，只將檔案歸檔到正確資料夾即可。

## 6. 研究方法學限制

- P2 是**官方完整前處理**，不是官方 ICA；P0/P2 的差別不全是去偽跡，還包含原始 FIR、人工編修與 AAR 等影響。
- 這裡假設官方資料與 Raw 保留同一連續時間軸，且會逐 Session 嚴格驗證。若資料被裁剪或 marker 移動，此版本刻意拒絕自動對齊。
- 使用作者已處理的整段測試 Session 訊號屬於**離線**比較；不等於即時可部署的前處理流程。
- P1（自行多通道 ICA 或 ICA-guided 2-channel）**尚未實作**；這包只準備 P2，比較時仍需共同試次、固定模型與五折。
- 本套程式已在**模擬 EEGLAB `.set`／`.set.zip`** 上通過單 Session、62 Session 資料生成和 1D-CNN Smoke 測試，尚未對你電腦上的真實官方 P2 訊號執行檢查或訓練。
