analyze_gpx
GPX 檔案批次清理、分析與視覺化工具

📋 專案簡介
這是一個強大的 GPX 檔案處理工具集，專為戶外活動愛好者和數據分析師設計。提供從檔案清理到路線分析的完整解決方案，特別針對台灣登山路線（如桃山單攻）進行優化。

✨ 主要功能
功能模組	說明	輸出結果
檔名清理	內建關鍵字白名單／黑名單，批次重新命名或搬移檔案	整理後的檔案結構
批次轉檔	將 .gpx 檔案匯出為標準化 CSV 格式	lat, lon, ele, time 欄位
路線統計	自動計算距離、爬升、時間等登山指標	詳細統計報告
智能過濾	根據地理位置條件篩選特定路線的 GPX 檔案	符合條件的路線集合
🗂️ 專案結構
analyze_gpx/
│
├── 📁 FileClean/                    # 檔案清理模組
│   ├── 📓 file_clean.ipynb          # 檔名篩選、資料夾重命名工具
│   │
│   ├── 📊 統計檔案
│   │   ├── file_list.csv            # 所有 .gpx 檔案統計
│   │   ├── normalized_counts.csv    # 關鍵字篩選依據
│   │   ├── gpx_kept_list.csv        # 保留檔案清單
│   │   └── gpx_deleted_list.csv     # 剔除檔案清單
│   │
│   ├── 🧹 清理後資料夾
│   │   ├── Clean_HikingNote_gpx/    # HikingNote 篩選結果
│   │   └── Clean_HikingBook_gpx/    # HikingBook 篩選結果
│   │
│   ├── 🎯 路線過濾系統
│   │   ├── 分辨有效路線.py           # 桃山步道路線過濾主程式
│   │   ├── Clean_MustPass_HBHN_gpx/ # 符合桃山單攻條件的路線
│   │   └── route_analysis_results.txt # 路線分析結果報告
│   │
├── 📁 GpxClean/                     # GPX 處理模組
│   ├── 📓 gpx_clean.ipynb           # GPX → CSV 轉檔及統計腳本
│   │
│   ├── 📈 轉檔輸出
│   │   ├── gpx_to_csv/              # 基本 CSV 檔案
│   │   │   └── [HN/HB]_xxxx.csv     # 帶前綴的 CSV 檔案
│   │   └── add_cum_csv/             # 增強統計 CSV
│   │       └── xxxx.csv             # 含累積統計欄位
│   │
└── 📖 README.md                     # 專案說明文件
