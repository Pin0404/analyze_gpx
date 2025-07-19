# analyze_gpx

GPX 檔案批次清理、分析與視覺化

## 專案特色

| 功能 | 說明 |
|------|------|
| 檔名／資料夾清理 | 內建關鍵字白名單／黑名單，可一次重新命名或搬移檔案。 |
| 批次轉檔 | 將資料夾內所有 .gpx 檔匯出為乾淨的 CSV（ lat lon ele time 欄位）。 |
| 路線統計 | 自動計算水平距離、總爬升、總下降、累積花費時間、海拔高度差等指標。 |
| **路線過濾** | **根據地理位置條件過濾出符合特定路線的GPX檔案（如桃山單攻路線）。** |

## 專案結構
analyze_gpx/
│
├── FileClean/

│   │
│   ├── file_clean.ipynb          # 檔名篩選、資料夾重命名、搬移工具
│   │
│   ├── file_list.csv             # 統計並列出所.gpx
│   ├── normalized_counts.csv     # 用來選擇 保留 & 剔除 檔名的關鍵字
│   ├── gpx_kept_list.csv         # 列出保留的檔案清單
│   ├── gpx_deleted_list.csv      # 列出剔除的檔案清單
│   │
│   ├── Clean_HikingNote_gpx/     # HikingNote 篩選後剩下的 .gpx 資料夾
│   │   └── xxxx.gpx

│   ├── Clean_HikingBook_gpx/     # HikingBook 篩選後剩下的 .gpx 資料夾
│   │   └── xxxx.gpx

│   │
│   ├── 分辨有效路線.py             # 桃山步道路線過濾系統
│   ├── Clean_MustPass_HBHN_gpx/   # 符合桃山單攻條件的 .gpx 資料夾 (自動生成)
│   │   └── xxxx.gpx              # 過濾後的有效路線檔案
│   └── route_analysis_results.txt # 路線分析結果報告 (自動生成)
│

├── GpxClean/

│   │
│   ├── gpx_clean.ipynb           # GPX → CSV & 統計相關腳本
│   │
│   ├── gpx_to_csv/               # 選取 GPX [lat,lon,ele,time] 欄位並轉檔儲存為 CSV
│   │   └── xxxx.csv              # 避免檔名重複 HikingNote 前墜為 HN，HikingBook 前墜為 HB
│   └── add_cum_csv/              # 增加 [cum_dist,cum_up,cum_down,cum_time] 統計欄位
│       └── xxxx.csv

│
└── README.md
