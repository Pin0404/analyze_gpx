import glob
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import folium
import gpxpy
import gpxpy.gpx
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")


# ===== 全域參數 =====
class DenoiseConfig:
    # Savitzky-Golay
    SG_WINDOW_LEN = 7  # 原 11，先縮小會顯著減弱平滑
    SG_POLY_ORDER = 2  # 原 3

    # Hampel
    HAMPEL_WINDOW = 5  # 原 7
    HAMPEL_ZTHRESH = 3.5  # 原 3.0，略放寬 → 減少被替換的點

    # 速度異常
    BASE_SPEED_THRESHOLD = 6.0  # 原 5.0 (m/s)，放寬 20%

    # 重新取樣
    TIME_INTERVAL_LIGHT = 5  # 秒
    TIME_INTERVAL_STRONG = 10  # 秒
    DIST_RESAMPLE = 20.0  # m，原 15.0

    # Savitzky-Golay 是否啟用
    ENABLE_SG_SMOOTH = True


# ===== 數據結構定義 =====
@dataclass
class GPXPoint:
    """GPS點數據結構"""

    latitude: float
    longitude: float
    elevation: Optional[float]
    time: datetime

    @property
    def lat(self) -> float:
        return self.latitude

    @property
    def lon(self) -> float:
        return self.longitude


@dataclass
class Cluster:
    """靜止點叢集"""

    indices: List[int]
    center_lat: float
    center_lon: float
    duration: float

    def __init__(self, indices: List[int]):
        self.indices = indices


@dataclass
class ProcessedGPX:
    """處理結果數據結構"""

    original: List[GPXPoint]
    standardized: List[GPXPoint]
    filtered: List[GPXPoint]
    smoothed: List[GPXPoint]
    statistics: Dict
    processing_log: List[str]
    filename: str
    folder_name: str


# ===== 混合時空標準化策略 =====
class HybridSpaceTimeStandardizer:
    """混合時空標準化處理器"""

    def __init__(self):
        self.processing_log = []

    def standardize_gps_track(self, points: List[GPXPoint]) -> List[GPXPoint]:
        """智能GPS軌跡標準化"""
        if len(points) < 2:
            return points

        # 分析採樣模式
        analysis = self._analyze_sampling_pattern(points)
        self.processing_log.append(
            f"📊 GPS採樣分析: 平均間隔{analysis['avg_interval']:.1f}秒, 變異係數{analysis['cv_interval']:.2f}"
        )

        # 根據分析結果選擇標準化策略
        if analysis["cv_interval"] < 0.3:
            # 低變異：輕度時間標準化
            self.processing_log.append("🎯 選擇策略: 輕度時間標準化")
            return self._light_time_standardization(points, analysis["avg_interval"])

        elif analysis["cv_interval"] < 0.8:
            # 中等變異：距離+時間混合標準化
            self.processing_log.append("🎯 選擇策略: 混合時空標準化")
            return self._hybrid_standardization(points)

        else:
            # 高變異：強力時間標準化
            self.processing_log.append("🎯 選擇策略: 強力時間標準化")
            return self._strong_time_standardization(points)

    def _analyze_sampling_pattern(self, points: List[GPXPoint]) -> Dict:
        """分析GPS採樣模式"""
        intervals = []
        for i in range(1, len(points)):
            interval = (points[i].time - points[i - 1].time).total_seconds()
            if interval > 0:
                intervals.append(interval)

        if not intervals:
            return {"avg_interval": 5, "cv_interval": 0}

        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv_interval = std_interval / avg_interval if avg_interval > 0 else 0

        return {
            "avg_interval": avg_interval,
            "std_interval": std_interval,
            "cv_interval": cv_interval,
            "total_points": len(points),
        }

    def _light_time_standardization(
        self, points: List[GPXPoint], avg_interval: float
    ) -> List[GPXPoint]:
        """輕度時間標準化"""
        target_interval = max(5, int(avg_interval))
        return self._interpolate_to_fixed_interval(points, target_interval)

    def _strong_time_standardization(self, points: List[GPXPoint]) -> List[GPXPoint]:
        """強力時間標準化"""
        return self._interpolate_to_fixed_interval(points, 10)

    def _hybrid_standardization(self, points: List[GPXPoint]) -> List[GPXPoint]:
        """混合時空標準化"""
        # 第一步：距離標準化
        distance_standardized = self._distance_based_resampling(points, 15.0)
        # 第二步：時間微調
        return self._interpolate_to_fixed_interval(distance_standardized, 10)

    def _interpolate_to_fixed_interval(
        self, points: List[GPXPoint], target_interval: int
    ) -> List[GPXPoint]:
        """插值到固定時間間隔"""
        if len(points) < 2:
            return points

        start_time = points[0].time
        end_time = points[-1].time
        total_duration = (end_time - start_time).total_seconds()

        if total_duration <= 0:
            return points

        num_points = int(total_duration / target_interval) + 1
        interpolated_points = []

        for i in range(num_points):
            target_time = start_time + timedelta(seconds=i * target_interval)
            interpolated_point = self._interpolate_point_at_time(points, target_time)
            if interpolated_point:
                interpolated_points.append(interpolated_point)

        # 確保包含最後一個點
        if interpolated_points and interpolated_points[-1].time < points[-1].time:
            interpolated_points.append(points[-1])

        return interpolated_points

    def _interpolate_point_at_time(
        self, points: List[GPXPoint], target_time: datetime
    ) -> Optional[GPXPoint]:
        """在指定時間插值GPS點"""
        before_point = None
        after_point = None

        for point in points:
            if point.time <= target_time:
                before_point = point
            elif point.time > target_time:
                after_point = point
                break

        if before_point is None:
            return points[0]
        if after_point is None:
            return points[-1]
        if before_point.time == target_time:
            return before_point

        # 線性插值
        time_diff = (after_point.time - before_point.time).total_seconds()
        if time_diff == 0:
            return before_point

        ratio = (target_time - before_point.time).total_seconds() / time_diff

        interpolated_lat = (
            before_point.latitude
            + (after_point.latitude - before_point.latitude) * ratio
        )
        interpolated_lon = (
            before_point.longitude
            + (after_point.longitude - before_point.longitude) * ratio
        )

        # 高度插值
        if before_point.elevation is not None and after_point.elevation is not None:
            interpolated_ele = (
                before_point.elevation
                + (after_point.elevation - before_point.elevation) * ratio
            )
        else:
            interpolated_ele = before_point.elevation or after_point.elevation

        return GPXPoint(
            latitude=interpolated_lat,
            longitude=interpolated_lon,
            elevation=interpolated_ele,
            time=target_time,
        )

    def _distance_based_resampling(
        self, points: List[GPXPoint], target_distance: float
    ) -> List[GPXPoint]:
        """基於距離的重採樣"""
        if len(points) < 2:
            return points

        resampled_points = [points[0]]
        current_distance = 0.0

        for i in range(1, len(points)):
            segment_distance = self._haversine_distance(points[i - 1], points[i])
            current_distance += segment_distance

            if current_distance >= target_distance:
                resampled_points.append(points[i])
                current_distance = 0.0

        # 確保包含終點
        if len(resampled_points) == 0 or resampled_points[-1] != points[-1]:
            resampled_points.append(points[-1])

        return resampled_points

    def _haversine_distance(self, point1: GPXPoint, point2: GPXPoint) -> float:
        """計算兩點間的Haversine距離(米)"""
        R = 6371000  # 地球半徑(米)

        lat1_rad = math.radians(point1.latitude)
        lat2_rad = math.radians(point2.latitude)
        delta_lat = math.radians(point2.latitude - point1.latitude)
        delta_lon = math.radians(point2.longitude - point1.longitude)

        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(
            lat1_rad
        ) * math.cos(lat2_rad) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


# ===== 主GPX處理器 =====
class GPXProcessor:
    """GPX檔案處理器"""

    def __init__(self):
        self.standardizer = HybridSpaceTimeStandardizer()
        self.processing_log = []

    def find_highest_elevation_point(self, points: List[GPXPoint]) -> Optional[int]:
        """
        找到海拔最高的GPS點的索引
        返回: 最高點的索引，如果沒有海拔數據則返回None
        """
        if not points:
            return None

        max_elevation = None
        highest_index = None

        for i, point in enumerate(points):
            if point.elevation is not None:
                if max_elevation is None or point.elevation > max_elevation:
                    max_elevation = point.elevation
                    highest_index = i

        if highest_index is not None:
            self.processing_log.append(
                f"🏔️ 找到最高點: 索引 {highest_index}, 海拔 {max_elevation:.1f}m"
            )

        return highest_index

    def protect_highest_point(
        self, points: List[GPXPoint], outlier_mask: List[bool]
    ) -> List[bool]:
        """
        保護海拔最高的點不被修正

        Args:
            points: GPS點列表
            outlier_mask: 異常點遮罩（True表示異常）

        Returns:
            修正後的遮罩（最高點設為False，即不修正）
        """
        highest_index = self.find_highest_elevation_point(points)

        if highest_index is not None:
            # 創建遮罩的副本以避免修改原始數據
            protected_mask = outlier_mask.copy()

            # 如果最高點被標記為異常，取消標記
            if protected_mask[highest_index]:
                protected_mask[highest_index] = False
                self.processing_log.append(
                    f"🛡️ 保護最高點: 索引 {highest_index} 已從異常修正中排除"
                )

            return protected_mask

        return outlier_mask

    def protect_highest_from_hampel(
        self, points: List[GPXPoint], lat_mask: np.ndarray, lon_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        保護海拔最高的點不被Hampel濾波修正

        Args:
            points: GPS點列表
            lat_mask: 緯度異常遮罩
            lon_mask: 經度異常遮罩

        Returns:
            修正後的緯度和經度遮罩
        """
        highest_index = self.find_highest_elevation_point(points)

        if highest_index is not None:
            # 創建遮罩的副本
            protected_lat_mask = lat_mask.copy()
            protected_lon_mask = lon_mask.copy()

            # 保護最高點
            if protected_lat_mask[highest_index] or protected_lon_mask[highest_index]:
                protected_lat_mask[highest_index] = False
                protected_lon_mask[highest_index] = False
                self.processing_log.append(
                    f"🛡️ 保護最高點: 索引 {highest_index} 已從Hampel濾波修正中排除"
                )

            return protected_lat_mask, protected_lon_mask

        return lat_mask, lon_mask

    def protect_highest_from_clustering(
        self, points: List[GPXPoint], clusters: List[Cluster]
    ) -> List[Cluster]:
        """
        保護海拔最高的點不被靜止點叢集處理

        Args:
            points: GPS點列表
            clusters: 靜止點叢集列表

        Returns:
            修正後的叢集列表（移除包含最高點的叢集）
        """
        highest_index = self.find_highest_elevation_point(points)

        if highest_index is not None:
            protected_clusters = []
            removed_clusters = 0

            for cluster in clusters:
                if highest_index not in cluster.indices:
                    protected_clusters.append(cluster)
                else:
                    removed_clusters += 1
                    self.processing_log.append(
                        f"🛡️ 保護最高點: 移除包含最高點的靜止叢集（索引 {highest_index}）"
                    )

            if removed_clusters > 0:
                self.processing_log.append(
                    f"🛡️ 總共移除 {removed_clusters} 個包含最高點的叢集"
                )

            return protected_clusters

        return clusters

    def load_gpx_files_from_folders(self, folder_names: List[str]) -> List[str]:
        """載入指定資料夾中的GPX檔案"""
        all_gpx_files = []
        current_directory = os.getcwd()

        for folder_name in folder_names:
            folder_path = os.path.join(current_directory, folder_name)

            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                gpx_files = glob.glob(os.path.join(folder_path, "*.gpx"))
                all_gpx_files.extend(gpx_files)
                print(f"✅ 找到 {len(gpx_files)} 個GPX檔案 in {folder_name}")
            else:
                print(f"⚠️ 警告: 找不到資料夾 {folder_name}")

        return all_gpx_files

    def load_gpx_file(self, filepath: str) -> List[GPXPoint]:
        """載入單個GPX檔案"""
        try:
            with open(filepath, "r", encoding="utf-8") as gpx_file:
                gpx = gpxpy.parse(gpx_file)

            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time:  # 確保有時間戳
                            points.append(
                                GPXPoint(
                                    latitude=point.latitude,
                                    longitude=point.longitude,
                                    elevation=point.elevation,
                                    time=point.time,
                                )
                            )

            # 按時間排序
            points.sort(key=lambda p: p.time)
            return points

        except Exception as e:
            print(f"❌ 載入GPX檔案失敗: {filepath}, 錯誤: {e}")
            return []

    def detect_speed_outliers(
        self, points: List[GPXPoint], speed_threshold=DenoiseConfig.BASE_SPEED_THRESHOLD
    ) -> List[bool]:
        """檢測速度異常值"""
        outlier_mask = [False] * len(points)
        outlier_count = 0

        for i in range(1, len(points)):
            distance = self._haversine_distance(points[i - 1], points[i])
            time_diff = (points[i].time - points[i - 1].time).total_seconds()

            if time_diff > 0:
                speed = distance / time_diff

                # 根據時間間隔調整閾值
                adjusted_threshold = self._get_adaptive_speed_threshold(
                    time_diff, speed_threshold
                )

                if speed > adjusted_threshold:
                    outlier_mask[i] = True
                    outlier_count += 1

        self.processing_log.append(f"🚨 檢測到 {outlier_count} 個速度異常點")
        return outlier_mask

    def _get_adaptive_speed_threshold(
        self, time_interval: float, base_threshold: float
    ) -> float:
        """根據時間間隔調整速度閾值"""
        if time_interval <= 5:
            return base_threshold
        elif time_interval <= 15:
            return base_threshold * 1.2
        elif time_interval <= 30:
            return base_threshold * 1.5
        else:
            return base_threshold * 2.0

    # ===== 只回傳「離群 index」的 Hampel 濾波 =====
    def hampel_mask(
        self,
        data: np.ndarray,
        window_size: int = DenoiseConfig.HAMPEL_WINDOW,
        threshold: float = DenoiseConfig.HAMPEL_ZTHRESH,
    ) -> np.ndarray:
        """
        回傳一個 bool 陣列，True = 該點被判定為離群值
        """
        half = window_size // 2
        mask = np.zeros(len(data), dtype=bool)

        for i in range(len(data)):
            s, e = max(0, i - half), min(len(data), i + half + 1)
            med = np.median(data[s:e])
            mad = np.median(np.abs(data[s:e] - med))
            if mad > 0 and np.abs(data[i] - med) / (mad * 1.4826) > threshold:
                mask[i] = True
        return mask

    def detect_stationary_clusters(
        self,
        points: List[GPXPoint],
        distance_threshold: float = 10.0,
        time_threshold: int = 60,
    ) -> List[Cluster]:
        """檢測靜止點叢集"""
        if len(points) < 2:
            return []

        clusters = []
        current_cluster = []

        for i, point in enumerate(points):
            if not current_cluster:
                current_cluster.append(i)
            else:
                # 計算與叢集中心的距離
                cluster_center = self._calculate_cluster_center(points, current_cluster)
                distance = self._haversine_distance(point, cluster_center)

                if distance <= distance_threshold:
                    current_cluster.append(i)
                else:
                    # 檢查叢集是否滿足時間條件
                    if (
                        self._calculate_cluster_duration(points, current_cluster)
                        >= time_threshold
                    ):
                        cluster = self._create_cluster(points, current_cluster)
                        clusters.append(cluster)

                    current_cluster = [i]

        # 處理最後一個叢集
        if (
            current_cluster
            and self._calculate_cluster_duration(points, current_cluster)
            >= time_threshold
        ):
            cluster = self._create_cluster(points, current_cluster)
            clusters.append(cluster)

        self.processing_log.append(f"🎯 檢測到 {len(clusters)} 個靜止點叢集")
        return clusters

    def _calculate_cluster_center(
        self, points: List[GPXPoint], indices: List[int]
    ) -> GPXPoint:
        """計算叢集中心點"""
        if not indices:
            return points[0]

        avg_lat = sum(points[i].latitude for i in indices) / len(indices)
        avg_lon = sum(points[i].longitude for i in indices) / len(indices)
        avg_ele = sum(points[i].elevation or 0 for i in indices) / len(indices)

        return GPXPoint(
            latitude=avg_lat,
            longitude=avg_lon,
            elevation=avg_ele,
            time=points[indices[0]].time,
        )

    def _calculate_cluster_duration(
        self, points: List[GPXPoint], indices: List[int]
    ) -> float:
        """計算叢集持續時間"""
        if len(indices) < 2:
            return 0

        start_time = points[indices[0]].time
        end_time = points[indices[-1]].time
        return (end_time - start_time).total_seconds()

    def _create_cluster(self, points: List[GPXPoint], indices: List[int]) -> Cluster:
        """創建叢集對象"""
        center = self._calculate_cluster_center(points, indices)
        duration = self._calculate_cluster_duration(points, indices)

        cluster = Cluster(indices)
        cluster.center_lat = center.latitude
        cluster.center_lon = center.longitude
        cluster.duration = duration

        return cluster

    def merge_gps_drift(
        self, points: List[GPXPoint], clusters: List[Cluster]
    ) -> List[GPXPoint]:
        """合併GPS漂移點"""
        corrected_points = []
        processed_indices = set()

        # 處理叢集
        for cluster in clusters:
            # 使用叢集中心點代替所有漂移點
            representative_point = GPXPoint(
                latitude=cluster.center_lat,
                longitude=cluster.center_lon,
                elevation=points[cluster.indices[0]].elevation,
                time=points[
                    cluster.indices[len(cluster.indices) // 2]
                ].time,  # 使用中間時間
            )
            corrected_points.append(representative_point)
            processed_indices.update(cluster.indices)

        # 保留非靜止點
        for i, point in enumerate(points):
            if i not in processed_indices:
                corrected_points.append(point)

        # 按時間排序
        corrected_points.sort(key=lambda p: p.time)

        removed_points = len(points) - len(corrected_points)
        self.processing_log.append(f"🔧 GPS漂移修正: 移除了 {removed_points} 個冗餘點")

        return corrected_points

    def savgol_smooth(
        self,
        coordinates: np.array,
        window_length=DenoiseConfig.SG_WINDOW_LEN,
        polyorder=DenoiseConfig.SG_POLY_ORDER,
    ) -> np.array:
        """Savitzky-Golay平滑"""
        if len(coordinates) < window_length:
            window_length = len(coordinates)
            if window_length % 2 == 0:
                window_length -= 1

        if window_length < 3:
            return coordinates

        polyorder = min(polyorder, window_length - 1)

        return savgol_filter(coordinates, window_length, polyorder)

        # ===== 自適應 Savitzky–Golay 平滑 =====

    def adaptive_savgol(
        self,
        pts: List[GPXPoint],
        base_win: int = DenoiseConfig.SG_WINDOW_LEN,
        poly: int = DenoiseConfig.SG_POLY_ORDER,
    ) -> List[GPXPoint]:
        """
        依據局部曲率動態調整視窗大小的 S-G 平滑
        • 拐彎越大 → 視窗越短 → 保留形狀
        """

        # 小工具：估算三點的「餘弦相似度」來近似曲率 (0~1)
        def _curv(p0, p1, p2):
            a = self._haversine_distance(p1, p2)
            b = self._haversine_distance(p0, p2)
            c = self._haversine_distance(p0, p1)
            if a * b * c == 0:
                return 0
            # 餘弦定理：cos∠P1 = (a² + c² − b²) / 2ac
            try:
                cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
                # 限制在 [-1, 1] 範圍內，避免數值誤差
                cos_angle = max(-1, min(1, cos_angle))
                return abs(cos_angle)
            except Exception:
                return 0

        lats = np.array([p.latitude for p in pts])
        lons = np.array([p.longitude for p in pts])
        sm_lats, sm_lons = [], []

        for i in range(len(pts)):
            # 1. 計算局部曲率
            if 1 <= i < len(pts) - 1:
                k = _curv(pts[i - 1], pts[i], pts[i + 1])
            else:
                k = 0

            # 2. 依曲率縮放視窗 (k→1 時代表折返，取最小 3)
            win = max(3, int(base_win * (1 - k)))
            if win % 2 == 0:  # Savitzky 必須奇數
                win += 1

            # 3. 對當前點所在的子序列做一次 S-G
            sl = max(0, i - win // 2)
            sr = min(len(pts), i + win // 2 + 1)

            # 檢查子序列長度
            subseq_len = sr - sl

            # 如果子序列太短，直接使用原始值
            if subseq_len < 3:
                sm_lats.append(lats[i])
                sm_lons.append(lons[i])
                continue

            # 調整視窗大小以適應子序列長度
            actual_win = min(win, subseq_len)
            if actual_win % 2 == 0:
                actual_win -= 1

            # 確保視窗大小至少為 3
            actual_win = max(3, actual_win)

            # 調整多項式階數
            actual_poly = min(poly, actual_win - 1)

            try:
                sm_lat = savgol_filter(lats[sl:sr], actual_win, actual_poly)[i - sl]
                sm_lon = savgol_filter(lons[sl:sr], actual_win, actual_poly)[i - sl]
            except Exception as e:
                # 如果仍然出錯，使用原始值
                print(f"⚠️ Savgol filter error at point {i}: {e}")
                sm_lat = lats[i]
                sm_lon = lons[i]

            sm_lats.append(sm_lat)
            sm_lons.append(sm_lon)

        # 4. 回傳新 GPXPoint 陣列
        return [
            GPXPoint(sm_lats[i], sm_lons[i], pts[i].elevation, pts[i].time)
            for i in range(len(pts))
        ]

    def _haversine_distance(self, point1: GPXPoint, point2: GPXPoint) -> float:
        """計算兩點間的Haversine距離"""
        return self.standardizer._haversine_distance(point1, point2)

    def _avg_offset(self, A, B):
        """
        A, B: 兩列 GPXPoint，長度應相同；多餘長度以 zip() 自動截短
        """
        if not A or not B:
            return 0.0

        distances = []
        for a, b in zip(A, B):
            try:
                distance = self._haversine_distance(a, b)
                distances.append(distance)
            except Exception as e:
                print(f"⚠️ 計算距離時出錯: {e}")
                continue

        if not distances:
            return 0.0

        return float(np.mean(distances))

    def calculate_statistics(
        self, original: List[GPXPoint], processed: List[GPXPoint]
    ) -> Dict:
        """計算處理統計"""

        def calculate_total_distance(points):
            if len(points) < 2:
                return 0
            total = 0
            for i in range(1, len(points)):
                total += self._haversine_distance(points[i - 1], points[i])
            return total

        def calculate_total_time(points):
            if len(points) < 2:
                return 0
            return (points[-1].time - points[0].time).total_seconds()

        original_distance = calculate_total_distance(original)
        processed_distance = calculate_total_distance(processed)
        original_time = calculate_total_time(original)
        processed_time = calculate_total_time(processed)

        return {
            "original_points": len(original),
            "processed_points": len(processed),
            "points_removed": len(original) - len(processed),
            "original_distance_km": original_distance / 1000,
            "processed_distance_km": processed_distance / 1000,
            "distance_reduction_pct": (
                (original_distance - processed_distance) / original_distance * 100
                if original_distance > 0
                else 0
            ),
            "original_duration_hours": original_time / 3600,
            "processed_duration_hours": processed_time / 3600,
            "processing_efficiency": (
                f"{(1 - len(processed)/len(original))*100:.1f}%"
                if len(original) > 0
                else "0%"
            ),
        }

    def save_processed_gpx(
        self, processed_result: ProcessedGPX, output_dir: str = "processed_gpx"
    ) -> str:
        """儲存處理後的GPX檔案"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 建立新的GPX對象
        gpx = gpxpy.gpx.GPX()

        # 建立軌跡
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx_track.name = f"桃山登山軌跡_已處理_{processed_result.filename}"
        gpx.tracks.append(gpx_track)

        # 建立軌跡段落
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        # 添加處理後的GPS點
        for point in processed_result.smoothed:
            gpx_point = gpxpy.gpx.GPXTrackPoint(
                latitude=point.latitude,
                longitude=point.longitude,
                elevation=point.elevation,
                time=point.time,
            )
            gpx_segment.points.append(gpx_point)

        # 建立檔案名稱
        filename_base = processed_result.filename.replace(".gpx", "")
        output_filename = (
            f"{processed_result.folder_name}_{filename_base}_processed.gpx"
        )
        output_path = os.path.join(output_dir, output_filename)

        # 儲存GPX檔案
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(gpx.to_xml())

        print(f"💾 已儲存處理後GPX檔案: {output_path}")
        return output_path

    def create_comparison_map(
        self, processed_result: ProcessedGPX, output_dir: str = "maps"
    ) -> str:
        """生成軌跡比較地圖"""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 轉換為DataFrame方便處理
        original_df = pd.DataFrame(
            [
                {"lat": p.latitude, "lon": p.longitude, "time": p.time}
                for p in processed_result.original
            ]
        )

        standardized_df = pd.DataFrame(
            [
                {"lat": p.latitude, "lon": p.longitude, "time": p.time}
                for p in processed_result.standardized
            ]
        )

        filtered_df = pd.DataFrame(
            [
                {"lat": p.latitude, "lon": p.longitude, "time": p.time}
                for p in processed_result.filtered
            ]
        )

        smoothed_df = pd.DataFrame(
            [
                {"lat": p.latitude, "lon": p.longitude, "time": p.time}
                for p in processed_result.smoothed
            ]
        )

        # 計算地圖中心點
        center_lat = original_df["lat"].mean()
        center_lon = original_df["lon"].mean()

        # 建立地圖
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap"
        )

        # 原始軌跡 (紅色，較粗)
        original_coords = original_df[["lat", "lon"]].values.tolist()
        folium.PolyLine(
            original_coords,
            color="red",
            weight=4,
            opacity=0.6,
            popup=f"原始軌跡 - {processed_result.folder_name}",
        ).add_to(m)

        # 標準化後軌跡 (橙色)
        standardized_coords = standardized_df[["lat", "lon"]].values.tolist()
        folium.PolyLine(
            standardized_coords,
            color="orange",
            weight=3,
            opacity=0.7,
            popup="標準化後軌跡",
        ).add_to(m)

        # Hampel濾波後軌跡 (黃色)
        filtered_coords = filtered_df[["lat", "lon"]].values.tolist()
        folium.PolyLine(
            filtered_coords, color="yellow", weight=3, opacity=0.8, popup="Hampel濾波後"
        ).add_to(m)

        # 最終軌跡 (藍色，較粗)
        final_coords = smoothed_df[["lat", "lon"]].values.tolist()
        folium.PolyLine(
            final_coords, color="blue", weight=4, opacity=0.9, popup="最終處理結果"
        ).add_to(m)

        # 標記起點和終點
        if len(original_coords) > 0:
            # 起點 (綠色)
            folium.Marker(
                original_coords[0],
                popup="起點",
                icon=folium.Icon(color="green", icon="play"),
            ).add_to(m)

            # 終點 (紅色)
            folium.Marker(
                original_coords[-1],
                popup="終點",
                icon=folium.Icon(color="red", icon="stop"),
            ).add_to(m)

        # 統計資訊
        stats = processed_result.statistics

        # 新增圖例和統計資訊
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 300px; height: 220px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; overflow-y: auto;">
        <p><b>🏔️ 桃山軌跡處理結果</b></p>
        <p><b>資料夾:</b> {processed_result.folder_name}</p>
        <p><b>檔案:</b> {processed_result.filename}</p>
        <hr>
        <p><span style="color:red; font-weight:bold;">━━</span> 原始軌跡 ({stats['original_points']}點)</p>
        <p><span style="color:orange; font-weight:bold;">━━</span> 標準化後</p>
        <p><span style="color:yellow; font-weight:bold;">━━</span> Hampel濾波</p>
        <p><span style="color:blue; font-weight:bold;">━━</span> 最終結果 ({stats['processed_points']}點)</p>
        <hr>
        <p><b>📊 處理統計:</b></p>
        <p>• 距離: {stats['processed_distance_km']:.2f} km</p>
        <p>• 效率提升: {stats['processing_efficiency']}</p>
        <p>• 移除點數: {stats['points_removed']}</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # 儲存地圖 (包含資料夾名稱)
        filename_base = processed_result.filename.replace(".gpx", "")
        output_filename = (
            f"{processed_result.folder_name}_{filename_base}_比較地圖.html"
        )
        output_path = os.path.join(output_dir, output_filename)
        m.save(output_path)

        print(f"🗺️ 已生成比較地圖: {output_path}")
        return output_path

    def process_gpx_file(self, filepath: str) -> ProcessedGPX:
        """處理單個GPX檔案的完整流程"""
        filename = os.path.basename(filepath)
        folder_name = (
            os.path.basename(os.path.dirname(filepath))
            if os.path.dirname(filepath)
            else "current"
        )

        self.processing_log = [f"🚀 開始處理: {filename}"]

        # 1. 載入GPX檔案
        original_points = self.load_gpx_file(filepath)
        if not original_points:
            return None

        self.processing_log.append(f"📁 載入完成: {len(original_points)} 個GPS點")

        # 1.5. 混合時空標準化
        self.standardizer.processing_log = []
        standardized_points = self.standardizer.standardize_gps_track(original_points)
        self.processing_log.extend(self.standardizer.processing_log)
        self.processing_log.append(f"📏 標準化完成: {len(standardized_points)} 個GPS點")

        # 2. 速度異常檢測 (使用標準化後的點)
        speed_outliers = self.detect_speed_outliers(standardized_points)

        # 🏔️ 保護最高點不被速度異常修正
        speed_outliers = self.protect_highest_point(standardized_points, speed_outliers)

        # 3. Hampel濾波處理經緯度
        lat_array = np.array([p.latitude for p in standardized_points])
        lon_array = np.array([p.longitude for p in standardized_points])

        lat_mask = self.hampel_mask(
            lat_array,
            DenoiseConfig.HAMPEL_WINDOW,
            DenoiseConfig.HAMPEL_ZTHRESH,
        )
        lon_mask = self.hampel_mask(
            lon_array,
            DenoiseConfig.HAMPEL_WINDOW,
            DenoiseConfig.HAMPEL_ZTHRESH,
        )

        # 🏔️ 保護最高點不被Hampel濾波修正
        lat_mask, lon_mask = self.protect_highest_from_hampel(
            standardized_points, lat_mask, lon_mask
        )

        # 更新座標
        filtered_points = []
        for i, p in enumerate(standardized_points):
            if speed_outliers[i] and lat_mask[i] and lon_mask[i]:
                # 重新抓一個小視窗 (±2) 的中位數當替代值
                s, e = max(0, i - 2), min(len(lat_array), i + 3)
                new_lat = float(np.median(lat_array[s:e]))
                new_lon = float(np.median(lon_array[s:e]))
            else:
                new_lat, new_lon = p.latitude, p.longitude

            filtered_points.append(GPXPoint(new_lat, new_lon, p.elevation, p.time))

        # 4. 靜止點檢測
        stationary_clusters = self.detect_stationary_clusters(filtered_points)

        # 🏔️ 保護最高點不被靜止點叢集處理
        stationary_clusters = self.protect_highest_from_clustering(
            filtered_points, stationary_clusters
        )

        # 5. GPS漂移修正
        drift_corrected_points = self.merge_gps_drift(
            filtered_points, stationary_clusters
        )

        # 6. Savitzky-Golay平滑
        if DenoiseConfig.ENABLE_SG_SMOOTH:
            smoothed_points = self.adaptive_savgol(drift_corrected_points)

            # 視覺差檢測
            offset = self._avg_offset(standardized_points, smoothed_points)
            if offset > 10:
                print(f"⚠️  平均偏移 {offset:.1f} m，啟用保守模式")
                orig_window = DenoiseConfig.SG_WINDOW_LEN
                DenoiseConfig.SG_WINDOW_LEN = max(3, DenoiseConfig.SG_WINDOW_LEN - 2)
                smoothed_points = self.adaptive_savgol(drift_corrected_points)
                DenoiseConfig.SG_WINDOW_LEN = orig_window

        else:
            smoothed_points = drift_corrected_points

        self.processing_log.append(f"🌊 Savitzky-Golay平滑完成")

        # 7. 計算統計
        statistics = self.calculate_statistics(original_points, smoothed_points)

        self.processing_log.append(
            f"📊 處理完成: 原始{statistics['original_points']}點 → 最終{statistics['processed_points']}點"
        )
        self.processing_log.append(
            f"🎯 效率提升: {statistics['processing_efficiency']}"
        )

        return ProcessedGPX(
            original=original_points,
            standardized=standardized_points,
            filtered=drift_corrected_points,
            smoothed=smoothed_points,
            statistics=statistics,
            processing_log=self.processing_log.copy(),
            filename=filename,
            folder_name=folder_name,
        )

    def process_batch(
        self, folder_names: List[str], save_gpx: bool = True, create_maps: bool = True
    ) -> List[ProcessedGPX]:
        """批次處理多個GPX檔案"""
        print("🏔️ 桃山GPX檔案批次處理開始...")

        # 載入所有GPX檔案
        gpx_files = self.load_gpx_files_from_folders(folder_names)

        if not gpx_files:
            print("❌ 沒有找到GPX檔案")
            return []

        print(f"📋 總共找到 {len(gpx_files)} 個GPX檔案")

        # 建立輸出資料夾
        if save_gpx:
            os.makedirs("processed_gpx", exist_ok=True)
        if create_maps:
            os.makedirs("maps", exist_ok=True)

        # 批次處理
        results = []
        for i, filepath in enumerate(gpx_files, 1):
            print(f"\n{'='*60}")
            print(f"處理進度: {i}/{len(gpx_files)}")

            result = self.process_gpx_file(filepath)
            if result:
                results.append(result)

                # 顯示處理日誌
                for log_entry in result.processing_log:
                    print(log_entry)

                # 儲存處理後的GPX檔案
                if save_gpx:
                    try:
                        saved_path = self.save_processed_gpx(result)
                    except Exception as e:
                        print(f"⚠️ 儲存GPX檔案失敗: {e}")

                # 生成比較地圖
                if create_maps:
                    try:
                        map_path = self.create_comparison_map(result)
                    except Exception as e:
                        print(f"⚠️ 生成地圖失敗: {e}")

            else:
                print(f"❌ 處理失敗: {os.path.basename(filepath)}")

        print(f"\n🎉 批次處理完成! 成功處理 {len(results)} 個檔案")

        # 生成處理報告
        if results:
            self._generate_batch_report(results)

        return results

    def _generate_batch_report(self, results: List[ProcessedGPX]):
        """生成批次處理報告"""
        print(f"\n{'='*80}")
        print("📊 桃山GPX處理總結報告")
        print(f"{'='*80}")

        total_original_points = sum(r.statistics["original_points"] for r in results)
        total_processed_points = sum(r.statistics["processed_points"] for r in results)
        avg_efficiency = np.mean(
            [float(r.statistics["processing_efficiency"].rstrip("%")) for r in results]
        )
        total_distance = sum(r.statistics["processed_distance_km"] for r in results)

        print(f"處理檔案數量: {len(results)}")
        print(f"原始GPS點總數: {total_original_points:,}")
        print(f"處理後GPS點總數: {total_processed_points:,}")
        print(f"平均處理效率: {avg_efficiency:.1f}%")
        print(f"總共節省GPS點: {total_original_points - total_processed_points:,}")
        print(f"總登山距離: {total_distance:.2f} km")

        # 顯示每個檔案的統計
        print(f"\n📋 詳細統計:")
        print("=" * 80)
        print(
            f"{'編號':<4} {'資料夾':<12} {'檔案名':<25} {'原始點':<8} {'處理後':<8} {'效率':<8} {'距離(km)':<10}"
        )
        print("-" * 80)

        for i, result in enumerate(results, 1):
            stats = result.statistics
            filename_short = (
                result.filename[:20] + "..."
                if len(result.filename) > 23
                else result.filename
            )
            folder_short = (
                result.folder_name[:10] + "..."
                if len(result.folder_name) > 12
                else result.folder_name
            )

            print(
                f"{i:<4} {folder_short:<12} {filename_short:<25} "
                f"{stats['original_points']:<8} {stats['processed_points']:<8} "
                f"{stats['processing_efficiency']:<8} {stats['processed_distance_km']:<10.2f}"
            )

        # 保存報告到檔案
        self._save_report_to_file(results)

    def _save_report_to_file(self, results: List[ProcessedGPX]):
        """儲存報告到檔案"""
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"桃山GPX處理報告_{timestamp}.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("🏔️ 桃山GPX處理總結報告\n")
            f.write("=" * 80 + "\n")
            f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"處理檔案數量: {len(results)}\n\n")

            total_original_points = sum(
                r.statistics["original_points"] for r in results
            )
            total_processed_points = sum(
                r.statistics["processed_points"] for r in results
            )
            avg_efficiency = np.mean(
                [
                    float(r.statistics["processing_efficiency"].rstrip("%"))
                    for r in results
                ]
            )
            total_distance = sum(r.statistics["processed_distance_km"] for r in results)

            f.write(f"總體統計:\n")
            f.write(f"- 原始GPS點總數: {total_original_points:,}\n")
            f.write(f"- 處理後GPS點總數: {total_processed_points:,}\n")
            f.write(f"- 平均處理效率: {avg_efficiency:.1f}%\n")
            f.write(f"- 總登山距離: {total_distance:.2f} km\n\n")

            f.write("詳細處理結果:\n")
            f.write("-" * 80 + "\n")

            for i, result in enumerate(results, 1):
                stats = result.statistics
                f.write(f"\n{i}. {result.folder_name}/{result.filename}\n")
                f.write(f"   原始點數: {stats['original_points']}\n")
                f.write(f"   處理後點數: {stats['processed_points']}\n")
                f.write(f"   處理效率: {stats['processing_efficiency']}\n")
                f.write(f"   距離: {stats['processed_distance_km']:.2f} km\n")
                f.write(f"   處理時長: {stats['processed_duration_hours']:.2f} 小時\n")

                # 添加處理日誌
                f.write(f"   處理日誌:\n")
                for log in result.processing_log:
                    f.write(f"     {log}\n")

        print(f"📝 處理報告已儲存: {report_path}")


# ===== 使用範例 =====
def main():
    """主函數 - 桃山GPX檔案處理"""

    print("🏔️ 歡迎使用桃山GPX處理系統!")
    print("=" * 50)

    # 初始化處理器
    processor = GPXProcessor()

    # 指定包含GPX檔案的資料夾名稱
    # 你可以修改這裡來指定你的GPX檔案位置
    folder_names = [
        "Clean_MustPass_HBHN_gpx",  # Clean_MustPass_HBHN_gpx目錄
        # 可以添加更多資料夾
    ]

    print("🔍 設定處理選項...")

    # 處理選項
    save_gpx_files = True  # 是否儲存處理後的GPX檔案
    create_comparison_maps = True  # 是否生成比較地圖

    print(f"📁 儲存GPX檔案: {'✅' if save_gpx_files else '❌'}")
    print(f"🗺️ 生成比較地圖: {'✅' if create_comparison_maps else '❌'}")

    # 開始批次處理
    try:
        results = processor.process_batch(
            folder_names=folder_names,
            save_gpx=save_gpx_files,
            create_maps=create_comparison_maps,
        )

        if results:
            print(f"\n✨ 處理完成! 共處理了 {len(results)} 個GPX檔案")

            # 顯示輸出檔案位置
            if save_gpx_files:
                print(f"📁 處理後的GPX檔案儲存在: ./processed_gpx/")

            if create_comparison_maps:
                print(f"🗺️ 比較地圖儲存在: ./maps/")

            print(f"📝 處理報告儲存在: ./reports/")

        else:
            print("❌ 沒有成功處理任何GPX檔案")

    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()


def interactive_mode():
    """互動模式 - 讓用戶選擇處理選項"""

    print("🏔️ 桃山GPX處理系統 - 互動模式")
    print("=" * 50)

    processor = GPXProcessor()

    # 讓用戶輸入資料夾
    print("請輸入包含GPX檔案的資料夾名稱 (用逗號分隔, 按Enter使用預設):")
    print("預設: 當前目錄")

    user_input = input("> ").strip()
    if user_input:
        folder_names = [name.strip() for name in user_input.split(",")]
    else:
        folder_names = ["."]

    print(f"📁 將搜尋資料夾: {folder_names}")

    # 處理選項
    save_gpx = input("是否儲存處理後的GPX檔案? (y/N): ").lower().startswith("y")
    create_maps = input("是否生成軌跡比較地圖? (y/N): ").lower().startswith("y")

    print("\n🚀 開始處理...")

    try:
        results = processor.process_batch(
            folder_names=folder_names, save_gpx=save_gpx, create_maps=create_maps
        )

        if results:
            print(f"\n✨ 處理完成!")
        else:
            print("❌ 沒有找到或成功處理任何GPX檔案")

    except Exception as e:
        print(f"❌ 處理失敗: {e}")


if __name__ == "__main__":
    # 選擇運行模式
    print("選擇運行模式:")
    print("1. 自動模式 (使用預設設定)")
    print("2. 互動模式 (自訂設定)")

    choice = input("請選擇 (1/2): ").strip()

    if choice == "2":
        interactive_mode()
    else:
        main()
