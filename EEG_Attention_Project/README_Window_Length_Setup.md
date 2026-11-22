# Window Length — 建立設定與空資料夾（第一階段）

這個包**只新增** `configs/window_length.yaml` 和必要的空資料夾；沒有資料擷取或訓練程式，不會更動現有 `epochs/2sec`、`data/labels/rt30_labels.csv`、`data/splits`、`src/` 或 `results/`。

## 建議專案分工

- `src/`：可重用的低階資料處理與訓練函式（既有檔案不更動）。
- `scripts/data_preparation/`：**未來**新建的資料擷取、事件重疊排除、RT 重新排序、品質檢查之獨立工具。
- `experiments/window_length/`：**未來**新建的 1D-CNN 2／3／4／5 秒比較 Runner，只負責實驗調度與彙整。
- `run_window_length.py`：**未來資料準備完成後**再建立，預設一次訓練四種長度、七組通道、五折；2 秒也重新訓練。

## 本包新增目錄

- `epochs/3sec/`, `epochs/4sec/`, `epochs/5sec/`
- `data/labels/window_length/{2,3,4,5}sec/`
- `scripts/data_preparation/`
- `results/diagnostics/window_length/`

**注意：**新建的 labels 目錄目前是空的；不可將原本 RT5–35% 的標籤直接複製作為新實驗 RT0–30% 的標籤。`epochs/2sec` 保留現有 NPZ；後續需對它重新產生此次實驗專用標籤。

## 自動路徑規則（未來程式將遵守）

輸入長度 `T`：

- epochs：`epochs/{T}sec/`
- manifest：`epochs/{T}sec/all_trials_UNLABELED.csv`
- labels：`data/labels/window_length/{T}sec/rt30_labels.csv`
- summary：`data/labels/window_length/{T}sec/rt30_summary.csv`
- 結果：`results/diagnostics/window_length/{T}sec/`
- B0 shape：`(N,30,T*250)`，例如 3 秒是 `(N,30,750)`
- raw forehead shape：`(N,2,T*500)`

受試者拆分永遠使用既有 `data/splits/subject_folds.csv`，不依秒數重新抽 Fold。

## 操作

將 ZIP 解壓至 `F:\CSIE\EEG_Attention_Project`，選擇**保留原有檔案**。再打開 `configs/window_length.yaml` 確認 `paths.raw_dataset_dir` 所填的原始連續 EEG 位置正確。`configs/paths.yaml` 保持指向 `epochs/2sec`，不要修改。

這份 YAML 是下一階段新程式將採用的規格；**尚未有程式會自動讀取它**，所以單憑此包不會擷取資料或訓練模型。

## 後續的嚴格資料規則

1. 對每個 T 擷取 `[t_i-T,t_i)`（只用偏移 onset 前 EEG）。
2. 依 Session 的前次 254 回復時間，排除跨入前次事件、或超出錄製起點的 Trial。
3. 對每個 Session **排除之後**保留的 Trial 重新排名 RT，最快／最慢各 30%；中間 40% 不訓練。
4. 新 NPZ 中所有 `x_row` 與 manifest 對齊；Trial ID 保持穩定。檢查所有 Fold 無受試者重疊。
5. 2 秒保留現有原始 Epoch 檔但重建專用 RT0–30 標籤。3、4、5 秒從連續 EEG 擷取。濾波暖機與其他未定義事件需記錄以供審核。
6. 2、3、4、5 秒重新訓練全部七通道組合，各自五折，總計 140 個模型訓練，**不覆蓋舊 E1 結果**。

跨視窗長度的資料量與標籤組成不同；正式解釋結果時建議另外做固定 Trial、固定 Label 的配對比較。
