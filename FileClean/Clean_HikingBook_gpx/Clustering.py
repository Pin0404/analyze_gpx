import glob
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ====== 增強版數據預處理部分 ======
class GPSTrajectoryProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.processed_files = []
        self.failed_files = []

    def load_and_clean_data(self, folder_path):
        """
        載入並清理GPS數據，支援資料夾內所有.gpx檔案
        """
        print("=== 開始批量處理GPX檔案 ===")

        # 尋找資料夾內所有GPX檔案
        gpx_files = self._find_gpx_files(folder_path)

        if not gpx_files:
            print("在指定資料夾中沒有找到GPX檔案")
            return None

        print(f"找到 {len(gpx_files)} 個GPX檔案")

        # 處理所有GPX檔案
        all_data = []
        for i, gpx_file in enumerate(gpx_files):
            print(f"處理檔案 {i+1}/{len(gpx_files)}: {os.path.basename(gpx_file)}")

            try:
                df = self._parse_gpx_file(gpx_file)
                if df is not None and len(df) > 0:
                    # 加入檔案來源標識
                    df["source_file"] = os.path.basename(gpx_file)
                    df["file_id"] = i
                    all_data.append(df)
                    self.processed_files.append(gpx_file)
                else:
                    print(f"  警告: {os.path.basename(gpx_file)} 沒有有效數據")
                    self.failed_files.append(gpx_file)
            except Exception as e:
                print(f"  錯誤: 處理 {os.path.basename(gpx_file)} 時發生錯誤: {e}")
                self.failed_files.append(gpx_file)

        if not all_data:
            print("沒有成功處理任何GPX檔案")
            return None

        # 合併所有數據
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n成功處理: {len(self.processed_files)} 個檔案")
        print(f"處理失敗: {len(self.failed_files)} 個檔案")
        print(f"總數據點: {len(combined_df)}")

        # 詳細的NaN值檢查和報告
        self._report_nan_status(combined_df)

        # 基本數據清理
        df_clean = self._comprehensive_nan_cleaning(combined_df)

        print(f"清理後數據點數量: {len(df_clean)}")
        return df_clean

    def _find_gpx_files(self, folder_path):
        """
        在資料夾中尋找所有GPX檔案
        """
        gpx_files = []

        # 支援多種路徑格式
        folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"錯誤: 資料夾 {folder_path} 不存在")
            return []

        # 尋找.gpx檔案（不區分大小寫）
        patterns = ["*.gpx", "*.GPX"]
        for pattern in patterns:
            gpx_files.extend(glob.glob(str(folder_path / pattern)))

        # 遞歸搜尋子資料夾
        for pattern in patterns:
            gpx_files.extend(
                glob.glob(str(folder_path / "**" / pattern), recursive=True)
            )

        # 去除重複並排序
        gpx_files = sorted(list(set(gpx_files)))

        return gpx_files

    def _parse_gpx_file(self, file_path):
        """
        解析GPX文件，處理常見的NaN問題
        """
        try:
            import gpxpy
        except ImportError:
            print("需要安裝gpxpy: pip install gpxpy")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as gpx_file:
                gpx = gpxpy.parse(gpx_file)
        except UnicodeDecodeError:
            # 嘗試其他編碼
            try:
                with open(file_path, "r", encoding="latin-1") as gpx_file:
                    gpx = gpxpy.parse(gpx_file)
            except Exception as e:
                print(f"    編碼錯誤: {e}")
                return None
        except Exception as e:
            print(f"    解析錯誤: {e}")
            return None

        data = []
        track_count = 0

        # 解析軌跡
        for track in gpx.tracks:
            track_count += 1
            for segment_idx, segment in enumerate(track.segments):
                for point_idx, point in enumerate(segment.points):
                    data.append(
                        {
                            "timestamp": point.time,
                            "latitude": (
                                point.latitude if point.latitude is not None else np.nan
                            ),
                            "longitude": (
                                point.longitude
                                if point.longitude is not None
                                else np.nan
                            ),
                            "elevation": (
                                point.elevation
                                if point.elevation is not None
                                else np.nan
                            ),
                            "track_number": track_count,
                            "segment_number": segment_idx + 1,
                            "point_number": point_idx + 1,
                        }
                    )

        # 解析航點（如果有的話）
        for waypoint in gpx.waypoints:
            data.append(
                {
                    "timestamp": waypoint.time,
                    "latitude": (
                        waypoint.latitude if waypoint.latitude is not None else np.nan
                    ),
                    "longitude": (
                        waypoint.longitude if waypoint.longitude is not None else np.nan
                    ),
                    "elevation": (
                        waypoint.elevation if waypoint.elevation is not None else np.nan
                    ),
                    "track_number": 0,  # 航點標記為track 0
                    "segment_number": 0,
                    "point_number": 0,
                }
            )

        if not data:
            return None

        df = pd.DataFrame(data)
        print(f"    提取到 {len(df)} 個GPS點")

        return df

    def _report_nan_status(self, df):
        """
        詳細報告NaN值狀況
        """
        print("\n=== NaN值檢查報告 ===")
        total_rows = len(df)

        for column in df.columns:
            nan_count = df[column].isna().sum()
            nan_percentage = (nan_count / total_rows) * 100
            print(f"{column}: {nan_count} NaN值 ({nan_percentage:.1f}%)")

        # 檢查完全為空的行
        completely_empty = df.isna().all(axis=1).sum()
        print(f"完全為空的行: {completely_empty}")

        # 檢查關鍵欄位的NaN情況
        critical_columns = ["latitude", "longitude"]
        if all(col in df.columns for col in critical_columns):
            critical_nan = df[critical_columns].isna().any(axis=1).sum()
            print(f"關鍵欄位(lat/lon)有NaN的行: {critical_nan}")

        # 按檔案檢查數據質量
        if "source_file" in df.columns:
            print("\n=== 各檔案數據質量 ===")
            for file_name in df["source_file"].unique():
                file_data = df[df["source_file"] == file_name]
                total_points = len(file_data)
                lat_nan = file_data["latitude"].isna().sum()
                lon_nan = file_data["longitude"].isna().sum()
                print(
                    f"{file_name}: {total_points} 點, 緯度NaN: {lat_nan}, 經度NaN: {lon_nan}"
                )

    def _comprehensive_nan_cleaning(self, df):
        """
        全面的NaN值清理策略
        """
        df_clean = df.copy()

        # 1. 移除完全為空的行
        df_clean = df_clean.dropna(how="all")
        print(f"移除完全為空的行後: {len(df_clean)} 行")

        # 2. 處理關鍵欄位的NaN值
        required_columns = ["latitude", "longitude"]
        for col in required_columns:
            if col not in df_clean.columns:
                print(f"警告: 缺少必要欄位 {col}")
                return pd.DataFrame()

        # 移除緯經度為NaN的行
        before_critical = len(df_clean)
        df_clean = df_clean.dropna(subset=required_columns)
        print(
            f"移除緯經度NaN後: {len(df_clean)} 行 (移除了 {before_critical - len(df_clean)} 行)"
        )

        # 3. 處理時間戳NaN值
        if "timestamp" in df_clean.columns:
            if df_clean["timestamp"].isna().any():
                print("發現時間戳NaN值，進行處理...")

                # 按檔案分組處理時間戳
                if "source_file" in df_clean.columns:
                    for file_name in df_clean["source_file"].unique():
                        file_mask = df_clean["source_file"] == file_name
                        file_data = df_clean.loc[file_mask, "timestamp"]

                        if file_data.isna().all():
                            # 如果整個檔案的時間戳都是NaN，創建順序時間戳
                            print(f"  為 {file_name} 創建順序時間戳")
                            start_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                                hours=len(df_clean[df_clean["source_file"] < file_name])
                            )
                            df_clean.loc[file_mask, "timestamp"] = pd.date_range(
                                start=start_time, periods=file_mask.sum(), freq="1min"
                            )
                        else:
                            # 嘗試插值
                            df_clean.loc[file_mask, "timestamp"] = pd.to_datetime(
                                file_data, errors="coerce"
                            )
                            df_clean.loc[file_mask, "timestamp"] = df_clean.loc[
                                file_mask, "timestamp"
                            ].interpolate()
                else:
                    # 單檔案處理邏輯
                    if df_clean["timestamp"].isna().all():
                        print("所有時間戳都是NaN，創建順序時間戳...")
                        df_clean["timestamp"] = pd.date_range(
                            start="2023-01-01", periods=len(df_clean), freq="1min"
                        )
                    else:
                        df_clean["timestamp"] = pd.to_datetime(
                            df_clean["timestamp"], errors="coerce"
                        )
                        df_clean = df_clean.dropna(subset=["timestamp"])

        # 4. 處理海拔NaN值
        if "elevation" in df_clean.columns:
            elevation_nan_count = df_clean["elevation"].isna().sum()
            if elevation_nan_count > 0:
                print(f"發現 {elevation_nan_count} 個海拔NaN值")

                # 按檔案分組處理海拔
                if "source_file" in df_clean.columns:
                    for file_name in df_clean["source_file"].unique():
                        file_mask = df_clean["source_file"] == file_name
                        file_elevation = df_clean.loc[file_mask, "elevation"]

                        if not file_elevation.isna().all():
                            # 使用該檔案的平均海拔填補
                            mean_elevation = file_elevation.mean()
                            df_clean.loc[
                                file_mask & df_clean["elevation"].isna(), "elevation"
                            ] = mean_elevation
                        else:
                            # 使用全局平均海拔
                            global_mean = df_clean["elevation"].mean()
                            if not pd.isna(global_mean):
                                df_clean.loc[file_mask, "elevation"] = global_mean
                            else:
                                df_clean.loc[file_mask, "elevation"] = 0
                else:
                    # 單檔案處理
                    if elevation_nan_count < len(df_clean) * 0.5:
                        df_clean["elevation"] = df_clean["elevation"].interpolate(
                            method="linear"
                        )
                        print("使用線性插值填補海拔NaN值")
                    else:
                        mean_elevation = df_clean["elevation"].mean()
                        if not pd.isna(mean_elevation):
                            df_clean["elevation"] = df_clean["elevation"].fillna(
                                mean_elevation
                            )
                            print(f"使用平均值 {mean_elevation:.1f} 填補海拔NaN值")
                        else:
                            df_clean["elevation"] = df_clean["elevation"].fillna(0)
        else:
            df_clean["elevation"] = 0
            print("沒有海拔數據，設置默認值為0")

        # 5. 座標範圍檢查（移除異常值）
        before_range = len(df_clean)
        df_clean = df_clean[
            (df_clean["latitude"] >= -90)
            & (df_clean["latitude"] <= 90)
            & (df_clean["longitude"] >= -180)
            & (df_clean["longitude"] <= 180)
        ]
        after_range = len(df_clean)
        if before_range != after_range:
            print(f"移除座標範圍異常的點: {before_range - after_range} 個")

        # 6. 最終NaN檢查和填補
        df_clean = self._final_nan_cleanup(df_clean)

        # 7. 保持重要欄位
        essential_columns = ["timestamp", "latitude", "longitude", "elevation"]
        optional_columns = ["source_file", "file_id", "track_number", "segment_number"]

        available_columns = [
            col for col in essential_columns if col in df_clean.columns
        ]
        available_columns.extend(
            [col for col in optional_columns if col in df_clean.columns]
        )

        return df_clean[available_columns].reset_index(drop=True)

    def _final_nan_cleanup(self, df):
        """
        最終的NaN清理，確保沒有遺漏
        """
        print("\n=== 最終NaN清理 ===")

        # 檢查是否還有NaN值
        remaining_nan = df.isna().sum()
        if remaining_nan.any():
            print("剩餘NaN值:")
            for col, nan_count in remaining_nan.items():
                if nan_count > 0:
                    print(f"  {col}: {nan_count}")

        # 對數值型欄位填充0，對類別型欄位填充適當值
        for column in df.columns:
            if df[column].dtype in ["float64", "int64"]:
                df[column] = df[column].fillna(0)
            elif df[column].dtype == "object":
                df[column] = df[column].fillna("unknown")

        return df

    def remove_gps_noise(self, df, speed_threshold=50):
        """
        移除GPS噪聲點，按檔案分組處理
        """
        if len(df) < 2:
            print("數據點太少，跳過噪聲移除")
            return df

        clean_data = []

        # 按檔案分組處理
        if "source_file" in df.columns:
            for file_name in df["source_file"].unique():
                file_data = df[df["source_file"] == file_name].copy()
                cleaned_file_data = self._remove_noise_single_file(
                    file_data, speed_threshold, file_name
                )
                if len(cleaned_file_data) > 0:
                    clean_data.append(cleaned_file_data)
        else:
            # 單檔案處理
            cleaned_data = self._remove_noise_single_file(
                df, speed_threshold, "單一軌跡"
            )
            if len(cleaned_data) > 0:
                clean_data.append(cleaned_data)

        # 合併所有清理後的數據
        if clean_data:
            result = pd.concat(clean_data, ignore_index=True)
            # 返回清理後的基本欄位
            essential_columns = ["timestamp", "latitude", "longitude", "elevation"]
            optional_columns = ["source_file", "file_id", "track_number"]

            available_columns = [
                col for col in essential_columns if col in result.columns
            ]
            available_columns.extend(
                [col for col in optional_columns if col in result.columns]
            )

            return result[available_columns].reset_index(drop=True)
        else:
            return pd.DataFrame()

    def _remove_noise_single_file(self, file_data, speed_threshold, file_name):
        """
        處理單一檔案的噪聲移除
        """
        if len(file_data) < 2:
            return file_data

        print(f"處理 {file_name} 的噪聲...")

        file_data = file_data.sort_values("timestamp").reset_index(drop=True)
        file_data["timestamp"] = pd.to_datetime(file_data["timestamp"])

        # 安全地計算相鄰點差異
        file_data["prev_lat"] = file_data["latitude"].shift(1)
        file_data["prev_lon"] = file_data["longitude"].shift(1)
        file_data["prev_time"] = file_data["timestamp"].shift(1)

        # 初始化列表
        distances = [0]  # 第一個點距離為0
        speeds = [0]  # 第一個點速度為0

        # 安全地計算速度
        for i in range(1, len(file_data)):
            try:
                if (
                    pd.notna(file_data.loc[i, "prev_lat"])
                    and pd.notna(file_data.loc[i, "prev_lon"])
                    and pd.notna(file_data.loc[i, "prev_time"])
                ):

                    # 計算距離
                    dist = geodesic(
                        (file_data.loc[i, "prev_lat"], file_data.loc[i, "prev_lon"]),
                        (file_data.loc[i, "latitude"], file_data.loc[i, "longitude"]),
                    ).meters

                    # 計算時間差
                    time_diff = (
                        file_data.loc[i, "timestamp"] - file_data.loc[i, "prev_time"]
                    ).total_seconds() / 3600

                    # 計算速度
                    if time_diff > 0 and not np.isnan(time_diff):
                        speed = (dist / 1000) / time_diff
                    else:
                        speed = 0

                    distances.append(dist)
                    speeds.append(speed)
                else:
                    distances.append(0)
                    speeds.append(0)
            except Exception as e:
                print(f"  計算速度時出錯 (行 {i}): {e}")
                distances.append(0)
                speeds.append(0)

        file_data["distance"] = distances
        file_data["speed"] = speeds

        # 安全地移除異常速度點
        before_count = len(file_data)
        file_data = file_data[
            (file_data["speed"] <= speed_threshold) & (file_data["speed"] >= 0)
        ]
        after_count = len(file_data)

        print(f"  {file_name}: 移除 {before_count - after_count} 個異常點")

        return file_data


# ====== 增強版特徵工程部分 ======
class FeatureEngineering:
    def __init__(self):
        pass

    def create_grid_features(self, df, grid_size=0.001):
        """
        擴展座標點網格特徵，安全處理NaN
        """
        df_copy = df.copy()

        # 安全地創建網格座標
        try:
            df_copy["grid_lat"] = (df_copy["latitude"] / grid_size).round() * grid_size
            df_copy["grid_lon"] = (df_copy["longitude"] / grid_size).round() * grid_size
            df_copy["grid_id"] = (
                df_copy["grid_lat"].astype(str) + "_" + df_copy["grid_lon"].astype(str)
            )
        except Exception as e:
            print(f"創建網格特徵時出錯: {e}")
            # 使用原始座標作為備用
            df_copy["grid_lat"] = df_copy["latitude"]
            df_copy["grid_lon"] = df_copy["longitude"]
            df_copy["grid_id"] = (
                df_copy["latitude"].astype(str) + "_" + df_copy["longitude"].astype(str)
            )

        return df_copy

    def calculate_movement_features(self, df):
        """
        計算移動特徵，安全處理NaN
        """
        df_copy = df.sort_values("timestamp").reset_index(drop=True)

        try:
            # 安全地計算方向變化
            df_copy["lat_diff"] = df_copy["latitude"].diff().fillna(0)
            df_copy["lon_diff"] = df_copy["longitude"].diff().fillna(0)

            # 安全地計算方位角
            df_copy["bearing"] = np.arctan2(df_copy["lon_diff"], df_copy["lat_diff"])
            df_copy["bearing"] = df_copy["bearing"].fillna(0)
            df_copy["bearing_change"] = df_copy["bearing"].diff().abs().fillna(0)

            # 安全地計算網格變化
            if "grid_id" in df_copy.columns:
                df_copy["grid_change"] = (
                    df_copy["grid_id"] != df_copy["grid_id"].shift(1)
                ).astype(int)
            else:
                df_copy["grid_change"] = 0

        except Exception as e:
            print(f"計算移動特徵時出錯: {e}")
            # 提供默認值
            df_copy["lat_diff"] = 0
            df_copy["lon_diff"] = 0
            df_copy["bearing"] = 0
            df_copy["bearing_change"] = 0
            df_copy["grid_change"] = 0

        return df_copy

    def create_multi_features(self, df):
        """
        創建多元特徵，全面防護NaN
        """
        features_df = df.copy()

        # 1. 基本座標特徵（確保不為NaN）
        features_df["lat_scaled"] = features_df["latitude"].fillna(0)
        features_df["lon_scaled"] = features_df["longitude"].fillna(0)

        # 2. 海拔相關特徵
        if "elevation" in features_df.columns:
            features_df["elevation_scaled"] = features_df["elevation"].fillna(0)
            features_df["elevation_change"] = features_df["elevation"].diff().fillna(0)
        else:
            features_df["elevation_scaled"] = 0
            features_df["elevation_change"] = 0

        # 3. 時間特徵（安全處理）
        try:
            features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
            features_df["hour"] = features_df["timestamp"].dt.hour
            features_df["day_of_week"] = features_df["timestamp"].dt.dayofweek
        except Exception as e:
            print(f"處理時間特徵時出錯: {e}")
            features_df["hour"] = 12  # 默認中午
            features_df["day_of_week"] = 0  # 默認週一

        # 4. 移動模式特徵（安全獲取）
        features_df["speed"] = features_df.get(
            "speed", pd.Series([0] * len(features_df))
        ).fillna(0)
        features_df["bearing_change"] = features_df.get(
            "bearing_change", pd.Series([0] * len(features_df))
        ).fillna(0)

        # 5. 密度特徵（安全計算）
        try:
            if "grid_id" in features_df.columns:
                grid_density = features_df["grid_id"].value_counts().to_dict()
                features_df["grid_density"] = (
                    features_df["grid_id"].map(grid_density).fillna(1)
                )
            else:
                features_df["grid_density"] = 1
        except Exception as e:
            print(f"計算密度特徵時出錯: {e}")
            features_df["grid_density"] = 1

        # 最終檢查：確保所有特徵都沒有NaN
        for col in features_df.columns:
            if features_df[col].dtype in ["float64", "int64"]:
                features_df[col] = features_df[col].fillna(0)

        return features_df


# ====== 增強版DBSCAN聚類 ======
class TrajectoryDBSCAN:
    def __init__(self):
        self.dbscan = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def prepare_features(self, df):
        """
        準備聚類特徵，強化NaN防護
        """
        # 選擇用於聚類的特徵
        feature_cols = [
            "lat_scaled",
            "lon_scaled",
            "elevation_scaled",
            "hour",
            "speed",
            "bearing_change",
            "grid_density",
        ]

        # 檢查哪些特徵存在且非全NaN
        available_features = []
        for col in feature_cols:
            if col in df.columns and not df[col].isna().all():
                available_features.append(col)

        if not available_features:
            print("警告: 沒有可用的特徵進行聚類")
            return None

        self.feature_columns = available_features
        print(f"使用特徵: {available_features}")

        # 提取特徵矩陣並確保沒有NaN
        X = df[available_features].fillna(0)

        # 檢查是否還有NaN或無窮值
        if X.isna().any().any():
            print("警告: 特徵矩陣中仍有NaN值，進行最終清理")
            X = X.fillna(0)

        if np.isinf(X.values).any():
            print("警告: 特徵矩陣中有無窮值，進行處理")
            X = X.replace([np.inf, -np.inf], 0)

        # 標準化特徵
        try:
            X_scaled = self.scaler.fit_transform(X)
        except Exception as e:
            print(f"特徵標準化時出錯: {e}")
            return X.values  # 返回未標準化的數據

        return X_scaled

    def find_optimal_parameters(self, X, eps_range=None, min_samples_range=None):
        """
        尋找最佳DBSCAN參數
        """
        if X is None or len(X) == 0:
            print("沒有有效數據進行參數優化")
            return {"eps": 0.5, "min_samples": 5}, pd.DataFrame()

        if eps_range is None:
            eps_range = np.arange(0.1, 2.0, 0.2)
        if min_samples_range is None:
            min_samples_range = range(3, min(15, max(5, len(X) // 10)))

        print("正在尋找最佳DBSCAN參數...")

        results = []
        best_score = -1
        best_params = {"eps": 0.5, "min_samples": 5}

        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X)

                    # 計算聚類評估指標
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)

                    if n_clusters > 1:
                        from sklearn.metrics import silhouette_score

                        silhouette_avg = silhouette_score(X, labels)

                        # 綜合評分 (考慮輪廓係數和聚類數量)
                        score = silhouette_avg * (1 - n_noise / len(labels))

                        results.append(
                            {
                                "eps": eps,
                                "min_samples": min_samples,
                                "n_clusters": n_clusters,
                                "n_noise": n_noise,
                                "silhouette_score": silhouette_avg,
                                "combined_score": score,
                            }
                        )

                        if score > best_score:
                            best_score = score
                            best_params = {"eps": eps, "min_samples": min_samples}

                except Exception as e:
                    continue

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            print(
                f"最佳參數: eps={best_params['eps']}, min_samples={best_params['min_samples']}"
            )
            print(f"最佳評分: {best_score:.3f}")
        else:
            print("未找到有效參數組合，使用默認參數")

        return best_params, results_df

    def perform_clustering(self, X, eps=0.5, min_samples=5):
        """
        執行DBSCAN聚類
        """
        if X is None or len(X) == 0:
            print("沒有有效數據進行聚類")
            return []

        try:
            print(f"執行DBSCAN聚類 (eps={eps}, min_samples={min_samples})...")
            self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = self.dbscan.fit_predict(X)

            # 聚類結果統計
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            print(f"發現 {n_clusters} 個聚類")
            print(f"噪聲點數量: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

            return labels

        except Exception as e:
            print(f"聚類過程中發生錯誤: {e}")
            return []


# ====== 結果驗證和視覺化 ======
class ResultValidator:
    def __init__(self):
        pass

    def plot_clusters(self, df, labels):
        """
        繪製聚類結果
        """
        try:
            plt.figure(figsize=(15, 10))

            # 創建包含標籤的數據框
            df_plot = df.copy()
            df_plot["cluster"] = labels

            # 子圖1: 地理分佈
            plt.subplot(2, 2, 1)
            unique_labels = set(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # 噪聲點用黑色
                    col = "black"
                    marker = "x"
                    label = "Noise"
                else:
                    marker = "o"
                    label = f"Cluster {k}"

                class_member_mask = labels == k
                xy = df_plot[class_member_mask]

                if len(xy) > 0:
                    plt.scatter(
                        xy["longitude"],
                        xy["latitude"],
                        c=[col],
                        marker=marker,
                        s=50,
                        alpha=0.7,
                        label=label,
                    )

            plt.xlabel("經度")
            plt.ylabel("緯度")
            plt.title("GPS軌跡聚類結果 - 地理分佈")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 子圖2: 時間分佈
            plt.subplot(2, 2, 2)
            if "timestamp" in df_plot.columns:
                df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

                for k in unique_labels:
                    if k != -1:
                        cluster_data = df_plot[df_plot["cluster"] == k]
                        if len(cluster_data) > 0:
                            plt.scatter(
                                cluster_data["timestamp"],
                                range(len(cluster_data)),
                                label=f"Cluster {k}",
                                alpha=0.7,
                            )

                plt.xlabel("時間")
                plt.ylabel("點序號")
                plt.title("聚類時間分佈")
                plt.legend()
                plt.xticks(rotation=45)

            # 子圖3: 速度分佈
            plt.subplot(2, 2, 3)
            if "speed" in df_plot.columns:
                for k in unique_labels:
                    if k != -1:
                        cluster_data = df_plot[df_plot["cluster"] == k]
                        if len(cluster_data) > 0:
                            plt.hist(
                                cluster_data["speed"],
                                bins=20,
                                alpha=0.5,
                                label=f"Cluster {k}",
                            )

                plt.xlabel("速度 (km/h)")
                plt.ylabel("頻次")
                plt.title("各聚類速度分佈")
                plt.legend()

            # 子圖4: 海拔分佈
            plt.subplot(2, 2, 4)
            if "elevation" in df_plot.columns:
                for k in unique_labels:
                    if k != -1:
                        cluster_data = df_plot[df_plot["cluster"] == k]
                        if len(cluster_data) > 0:
                            plt.scatter(
                                cluster_data["longitude"],
                                cluster_data["elevation"],
                                label=f"Cluster {k}",
                                alpha=0.7,
                            )

                plt.xlabel("經度")
                plt.ylabel("海拔 (m)")
                plt.title("經度-海拔分佈")
                plt.legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"繪圖時發生錯誤: {e}")

    def analyze_clusters(self, df, labels):
        """
        分析聚類結果統計
        """
        try:
            df_analysis = df.copy()
            df_analysis["cluster"] = labels

            cluster_stats = {}
            unique_labels = set(labels)

            print("\n=== 聚類分析結果 ===")

            for cluster_id in unique_labels:
                if cluster_id == -1:
                    continue

                cluster_data = df_analysis[df_analysis["cluster"] == cluster_id]

                stats = {
                    "size": len(cluster_data),
                    "lat_center": cluster_data["latitude"].mean(),
                    "lon_center": cluster_data["longitude"].mean(),
                    "lat_std": cluster_data["latitude"].std(),
                    "lon_std": cluster_data["longitude"].std(),
                }

                # 時間範圍分析
                if "timestamp" in cluster_data.columns:
                    cluster_data["timestamp"] = pd.to_datetime(
                        cluster_data["timestamp"]
                    )
                    stats["time_span"] = (
                        cluster_data["timestamp"].max()
                        - cluster_data["timestamp"].min()
                    ).total_seconds() / 3600
                    stats["avg_hour"] = cluster_data["timestamp"].dt.hour.mean()

                # 速度分析
                if "speed" in cluster_data.columns:
                    stats["avg_speed"] = cluster_data["speed"].mean()
                    stats["max_speed"] = cluster_data["speed"].max()

                # 海拔分析
                if "elevation" in cluster_data.columns:
                    stats["avg_elevation"] = cluster_data["elevation"].mean()
                    stats["elevation_range"] = (
                        cluster_data["elevation"].max()
                        - cluster_data["elevation"].min()
                    )

                # 檔案來源分析
                if "source_file" in cluster_data.columns:
                    stats["source_files"] = cluster_data["source_file"].nunique()
                    stats["main_source"] = (
                        cluster_data["source_file"].mode().iloc[0]
                        if len(cluster_data["source_file"].mode()) > 0
                        else "unknown"
                    )

                cluster_stats[cluster_id] = stats

                # 打印聚類資訊
                print(f"\n聚類 {cluster_id}:")
                print(f"  點數量: {stats['size']}")
                print(
                    f"  中心座標: ({stats['lat_center']:.6f}, {stats['lon_center']:.6f})"
                )
                print(f"  座標標準差: ({stats['lat_std']:.6f}, {stats['lon_std']:.6f})")

                if "time_span" in stats:
                    print(f"  時間跨度: {stats['time_span']:.1f} 小時")
                    print(f"  平均小時: {stats['avg_hour']:.1f}")

                if "avg_speed" in stats:
                    print(f"  平均速度: {stats['avg_speed']:.1f} km/h")
                    print(f"  最大速度: {stats['max_speed']:.1f} km/h")

                if "avg_elevation" in stats:
                    print(f"  平均海拔: {stats['avg_elevation']:.1f} m")
                    print(f"  海拔範圍: {stats['elevation_range']:.1f} m")

                if "source_files" in stats:
                    print(f"  涉及檔案數: {stats['source_files']}")
                    print(f"  主要來源: {stats['main_source']}")

            # 噪聲分析
            noise_count = list(labels).count(-1)
            if noise_count > 0:
                print(f"\n噪聲點:")
                print(f"  數量: {noise_count}")
                print(f"  比例: {noise_count/len(labels)*100:.1f}%")

            return cluster_stats

        except Exception as e:
            print(f"聚類分析時發生錯誤: {e}")
            return {}


# ====== 主要分析函數 ======
def main_analysis(folder_path):
    """
    主要分析流程
    """
    try:
        print("=== GPS軌跡聚類分析系統 ===")
        print(f"分析資料夾: {folder_path}")

        # 步驟1: 數據載入和清理
        print("\n1. 數據載入和清理...")
        processor = GPSTrajectoryProcessor()
        df_clean = processor.load_and_clean_data(folder_path)

        if df_clean is None or len(df_clean) == 0:
            print("沒有可用的數據進行分析")
            return None, None, None

        # 記錄處理結果
        processed_files = processor.processed_files
        failed_files = processor.failed_files

        # 步驟2: 特徵工程
        print("\n2. 特徵工程...")
        feature_eng = FeatureEngineering()

        # 創建網格特徵
        df_grid = feature_eng.create_grid_features(df_clean)

        # 移除GPS噪聲
        df_denoised = processor.remove_gps_noise(df_grid)

        # 計算移動特徵
        df_movement = feature_eng.calculate_movement_features(df_denoised)

        # 創建多元特徵
        df_features = feature_eng.create_multi_features(df_movement)

        print(f"特徵工程完成，最終數據點: {len(df_features)}")

        # 步驟3: DBSCAN聚類
        print("\n3. DBSCAN聚類分析...")
        clusterer = TrajectoryDBSCAN()
        X = clusterer.prepare_features(df_features)

        if X is None:
            print("特徵準備失敗")
            return df_features, None, None

        # 尋找最佳參數
        best_params, param_results = clusterer.find_optimal_parameters(X)

        # 執行聚類
        labels = clusterer.perform_clustering(X, **best_params)

        if len(labels) == 0:
            print("聚類執行失敗")
            return df_features, None, None

        # 步驟4: 結果驗證
        print("\n4. 結果驗證...")
        validator = ResultValidator()
        validator.plot_clusters(df_features, labels)
        cluster_stats = validator.analyze_clusters(df_features, labels)

        # 處理結果總結
        print(f"\n=== 處理總結 ===")
        print(f"成功處理檔案: {len(processed_files)}")
        print(f"失敗檔案: {len(failed_files)}")
        print(f"總數據點: {len(df_features)}")
        print(f"發現聚類: {len(set(labels)) - (1 if -1 in labels else 0)}")

        if failed_files:
            print(f"\n失敗檔案列表:")
            for failed_file in failed_files:
                print(f"  - {os.path.basename(failed_file)}")

        return df_features, labels, cluster_stats

    except Exception as e:
        print(f"分析過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


# ====== 使用範例 ======
if __name__ == "__main__":
    folder_path = os.getcwd()  # 現在的資料夾
    df_result, cluster_labels, stats = main_analysis(folder_path)
    print("GPX資料夾批量分析代碼準備完成！")
    print("\n主要功能:")
    print("✓ 自動掃描資料夾內所有GPX檔案")
    print("✓ 批量解析和合併GPX軌跡數據")
    print("✓ 保留檔案來源標識")
    print("✓ 按檔案分組進行數據清理")
    print("✓ 檔案級別的錯誤處理")
    print("✓ 詳細的處理報告")
    print("✓ DBSCAN聚類分析")
    print("✓ 視覺化結果展示")
    print("✓ 聚類統計分析")

    print("\n使用方法:")
    print("folder_path = 'path/to/your/gpx/folder'")
    print("df_result, cluster_labels, stats = main_analysis(folder_path)")

    print("\n範例:")
    print("# 分析當前目錄下的gpx_data資料夾")
    print("# df_result, cluster_labels, stats = main_analysis('./gpx_data')")
