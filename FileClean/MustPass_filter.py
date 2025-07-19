# ==================== 桃山步道路線過濾系統 ====================
# 目標：從多個GPX檔案中過濾出只經過「主要步道（單攻桃山路線）」的軌跡，並進行詳細分析
# 新增功能：
# 1. 從指定的兩個資料夾讀取GPX檔案
# 2. 將符合條件的路線複製到新資料夾

import glob
import math
import os
import shutil
from datetime import timedelta

import gpxpy
import gpxpy.gpx


def haversine(lat1, lon1, lat2, lon2):
    """計算地球上兩點間的大圓距離（Haversine公式）"""
    R = 6371000  # 地球半徑（公尺）
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine公式核心計算
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # 返回距離（公尺）


def check_gpx_duration(file_path, max_hours=24):
    """檢查GPX檔案記錄時間是否超過指定小時數"""
    try:
        with open(file_path, "r", encoding="utf-8") as gpx_file:
            gpx = gpxpy.parse(gpx_file)

            # 收集所有時間戳記
            timestamps = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time:
                            timestamps.append(point.time)

            # 如果沒有時間資訊，視為有效
            if not timestamps:
                return True

            # 計算時間範圍
            start_time = min(timestamps)
            end_time = max(timestamps)
            duration = end_time - start_time

            # 檢查是否超過24小時
            return duration <= timedelta(hours=max_hours)

    except Exception as e:
        print(f"檢查GPX時間時發生錯誤 {file_path}: {e}")
        return True  # 發生錯誤時預設為有效


def check_start_end_points(points, wuling_location, radius=200):
    """檢查軌跡起點和終點是否在武陵山莊附近（單攻路線特徵）"""
    if len(points) < 2:
        return False

    start_point = points[0]
    end_point = points[-1]

    # 檢查起點是否在武陵山莊附近
    start_distance = haversine(
        wuling_location[0], wuling_location[1], start_point[0], start_point[1]
    )

    # 檢查終點是否在武陵山莊附近
    end_distance = haversine(
        wuling_location[0], wuling_location[1], end_point[0], end_point[1]
    )

    return start_distance <= radius and end_distance <= radius


def load_gpx_files_from_folders(folder_names):
    """從指定的資料夾載入所有GPX檔案"""
    all_gpx_files = []
    current_directory = os.getcwd()

    for folder_name in folder_names:
        folder_path = os.path.join(current_directory, folder_name)

        # 檢查資料夾是否存在
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            gpx_files = glob.glob(os.path.join(folder_path, "*.gpx"))
            all_gpx_files.extend(gpx_files)
            print(f"從 {folder_name} 資料夾找到 {len(gpx_files)} 個GPX檔案")
        else:
            print(f"警告：找不到資料夾 {folder_name}")

    return all_gpx_files


def create_output_folder(folder_name):
    """創建輸出資料夾，如果已存在則清空"""
    current_directory = os.getcwd()
    output_folder = os.path.join(current_directory, folder_name)

    # 如果資料夾已存在，先刪除
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"已清空現有的 {folder_name} 資料夾")

    # 創建新資料夾
    os.makedirs(output_folder)
    print(f"已創建 {folder_name} 資料夾")

    return output_folder


def copy_valid_routes_to_folder(valid_routes, output_folder):
    """將符合條件的GPX檔案複製到指定資料夾"""
    copied_count = 0

    for route_path in valid_routes:
        try:
            # 取得檔案名稱
            filename = os.path.basename(route_path)
            # 目標路徑
            destination_path = os.path.join(output_folder, filename)

            # 複製檔案
            shutil.copy2(route_path, destination_path)
            copied_count += 1

        except Exception as e:
            print(f"複製檔案時發生錯誤 {route_path}: {e}")

    print(
        f"成功複製 {copied_count} 個符合條件的GPX檔案到 {os.path.basename(output_folder)} 資料夾"
    )
    return copied_count


def parse_gpx_file(file_path):
    """解析GPX檔案並提取所有軌跡點座標"""
    with open(file_path, "r", encoding="utf-8") as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        points = []
        # 遍歷所有軌跡、片段和點位
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    points.append((point.latitude, point.longitude))
        return points


def get_exclude_radius(excluded_point, default_radius):
    """根據排除點座標返回對應的容錯半徑"""
    # 喀拉業山使用特殊半徑300公尺
    if excluded_point == (24.44000, 121.29800):
        return 300
    else:
        return default_radius


def analyze_routes(
    gpx_files, must_pass_points, excluded_points, pass_radius, exclude_radius
):
    """分析所有路線，統計經過排除點與未經過必經點的情況"""
    # 初始化統計字典
    excluded_point_count = {point: 0 for point in excluded_points}  # 各排除點被經過次數
    must_pass_point_missed = {
        point: 0 for point in must_pass_points
    }  # 各必經點被遺漏次數

    for gpx_file in gpx_files:
        points = parse_gpx_file(gpx_file)

        # 統計經過的排除點（應避免的地點）
        for excluded_point in excluded_points:
            lat, lon = excluded_point
            current_exclude_radius = get_exclude_radius(excluded_point, exclude_radius)

            if any(
                haversine(lat, lon, p_lat, p_lon) <= current_exclude_radius
                for p_lat, p_lon in points
            ):
                excluded_point_count[excluded_point] += 1

        # 統計未經過的必經點（應通過的地點）
        for lat, lon in must_pass_points:
            if not any(
                haversine(lat, lon, p_lat, p_lon) <= pass_radius
                for p_lat, p_lon in points
            ):
                must_pass_point_missed[(lat, lon)] += 1

    return excluded_point_count, must_pass_point_missed


def filter_routes(
    gpx_files,
    must_pass_points,
    excluded_points,
    wuling_location,
    pass_radius,
    exclude_radius,
    pass_ratio,
    start_end_radius=200,
):
    """根據桃山單攻路線條件過濾GPX檔案"""
    valid_routes = []
    filtered_out = {"duration": 0, "points": 0, "start_end": 0, "exclude": 0, "pass": 0}

    for gpx_file in gpx_files:
        # === 條件0：檢查GPX記錄時間是否超過24小時 ===
        if not check_gpx_duration(gpx_file, max_hours=24):
            filtered_out["duration"] += 1
            continue

        points = parse_gpx_file(gpx_file)

        # 檢查軌跡是否有足夠的點位
        if len(points) < 10:
            filtered_out["points"] += 1
            continue

        # === 條件1：檢查起點和終點是否在武陵山莊附近 ===
        if not check_start_end_points(points, wuling_location, start_end_radius):
            filtered_out["start_end"] += 1
            continue

        # === 條件2：計算通過的必經點數量 ===
        pass_count = sum(
            any(
                haversine(lat, lon, p_lat, p_lon) <= pass_radius
                for p_lat, p_lon in points
            )
            for lat, lon in must_pass_points
        )

        # === 條件3：計算經過的排除點數量（使用個別半徑） ===
        exclude_count = 0
        for excluded_point in excluded_points:
            lat, lon = excluded_point
            current_exclude_radius = get_exclude_radius(excluded_point, exclude_radius)

            if any(
                haversine(lat, lon, p_lat, p_lon) <= current_exclude_radius
                for p_lat, p_lon in points
            ):
                exclude_count += 1

        # === 過濾條件檢查 ===
        if exclude_count > 0:
            filtered_out["exclude"] += 1
            continue

        if pass_count < pass_ratio * len(must_pass_points):
            filtered_out["pass"] += 1
            continue

        # 通過所有條件
        valid_routes.append(gpx_file)

    return valid_routes, filtered_out


def main():
    """主函數：執行桃山步道路線過濾邏輯並分析結果"""

    # === 設定讀取的GPX資料夾名稱 ===
    input_folders = ["Clean_HikingBook_gpx", "Clean_HikingNote_gpx"]

    # === 設定輸出資料夾名稱 ===
    output_folder_name = "Clean_MustPass_HBHN_gpx"

    # === 武陵山莊起點座標 ===
    wuling_location = (24.39700, 121.30770)  # 武陵山莊位置

    # === 桃山主要步道必經點座標 ===
    # 從武陵山莊起點到桃山山頂的關鍵點位
    must_pass_points = [
        (24.39700, 121.30770),  # 武陵山莊起點/桃山瀑布步道0k導覽圖
        (24.39890, 121.30830),  # 桃山瀑布步道0.5k
        (24.40520, 121.30750),  # 桃山登山口（0K右轉離開瀑布步道）
        (24.41011, 121.31125),  # 主要步道1K
        (24.41304, 121.30947),  # 主要步道1.5K
        (24.41630, 121.30720),  # 主要步道2K
        (24.42100, 121.30600),  # 主要步道2.5K
        (24.42400, 121.30500),  # 主要步道3K
        (24.42640, 121.30377),  # 主要步道3.5K
        (24.42911, 121.30409),  # 主要步道4K
        (24.43251, 121.30463),  # 主要步道4.5K
        (24.43400, 121.30500),  # 桃山山頂
    ]

    # === 排除點座標 ===
    # 避免選到非主要路線的軌跡（瀑布步道延伸、其他山峰）
    excluded_points = [
        (24.40600, 121.30400),  # 桃山瀑布步道路標指示2.5k
        (24.41402, 121.3027),  # 桃山瀑布步道路標指示4.3k
        (24.44000, 121.29800),  # 喀拉業山 (特殊半徑300m)
        (24.44500, 121.29500),  # 品田山
        (24.44200, 121.31200),  # 池有山
    ]

    # === 過濾參數設定（根據配置檔案調整） ===
    pass_radius = 150  # 必經點容錯半徑（公尺）
    exclude_radius = 50  # 排除點容錯半徑（公尺，喀拉業山除外）
    pass_ratio = 0.7  # 必經點通過比例（70%）- 根據配置檔案修正
    start_end_radius = 200  # 起終點檢查半徑（公尺）

    # === 執行過濾流程 ===
    print("=== 開始執行桃山步道路線過濾系統 ===")

    # 1. 從指定資料夾載入所有GPX檔案
    gpx_files = load_gpx_files_from_folders(input_folders)
    print(f"總共找到 {len(gpx_files)} 個GPX檔案")

    if len(gpx_files) == 0:
        print("錯誤：沒有找到任何GPX檔案，請檢查資料夾是否存在且包含GPX檔案")
        return

    # 2. 過濾符合桃山單攻條件的路線
    print("\n正在過濾路線...")
    valid_routes, filtered_out = filter_routes(
        gpx_files,
        must_pass_points,
        excluded_points,
        wuling_location,
        pass_radius,
        exclude_radius,
        pass_ratio,
        start_end_radius,
    )

    # 3. 創建輸出資料夾並複製符合條件的檔案
    if valid_routes:
        output_folder = create_output_folder(output_folder_name)
        copy_valid_routes_to_folder(valid_routes, output_folder)
    else:
        print("沒有找到符合條件的路線，不創建輸出資料夾")

    # 4. 分析所有路線的詳細統計
    print("\n正在分析路線統計...")
    excluded_point_count, must_pass_point_missed = analyze_routes(
        gpx_files, must_pass_points, excluded_points, pass_radius, exclude_radius
    )

    # === 輸出分析結果到檔案 ===
    with open("route_analysis_results.txt", "w", encoding="utf-8") as result_file:
        # 輸出符合條件的路線清單
        result_file.write("=== 符合條件的桃山單攻路線 ===\n")
        result_file.write(f"共找到 {len(valid_routes)} 條符合條件的路線：\n")
        for route in valid_routes:
            result_file.write(f"✓ {os.path.basename(route)}\n")

        # 輸出過濾統計
        result_file.write(f"\n=== 過濾統計 ===\n")
        result_file.write(f"總檔案數量: {len(gpx_files)}\n")
        result_file.write(f"記錄時間超過24小時: {filtered_out['duration']} 個檔案\n")
        result_file.write(f"軌跡點數不足: {filtered_out['points']} 個檔案\n")
        result_file.write(f"起終點不符合: {filtered_out['start_end']} 個檔案\n")
        result_file.write(f"經過排除點: {filtered_out['exclude']} 個檔案\n")
        result_file.write(f"必經點通過率不足: {filtered_out['pass']} 個檔案\n")

        # 輸出資料夾資訊
        result_file.write(f"\n=== 資料夾資訊 ===\n")
        result_file.write(f"讀取資料夾: {', '.join(input_folders)}\n")
        result_file.write(f"輸出資料夾: {output_folder_name}\n")

        # 輸出最常被經過的排除點（問題點位）
        result_file.write("\n=== 最常被經過的排除點 ===\n")
        if excluded_point_count:
            most_excluded_point = max(
                excluded_point_count, key=excluded_point_count.get
            )
            result_file.write(
                f"座標: {most_excluded_point}, 被經過次數: {excluded_point_count[most_excluded_point]}\n"
            )

        # 輸出經常被遺漏的必經點
        result_file.write("\n=== 經常被遺漏的必經點 ===\n")
        for point, count in must_pass_point_missed.items():
            if count > 0:
                result_file.write(f"座標: {point}, 被遺漏次數: {count}\n")

        # 輸出過濾參數設定
        result_file.write(f"\n=== 過濾參數設定 ===\n")
        result_file.write(f"必經點容錯半徑: {pass_radius}公尺\n")
        result_file.write(f"排除點容錯半徑: {exclude_radius}公尺 (喀拉業山: 300公尺)\n")
        result_file.write(f"必經點通過比例: {pass_ratio*100}%\n")
        result_file.write(f"起終點檢查半徑: {start_end_radius}公尺\n")
        result_file.write(f"最大記錄時間限制: 24小時\n")

    # === 顯示最終結果 ===
    print(f"\n=== 分析完成 ===")
    print(f"分析結果已儲存至 route_analysis_results.txt")
    print(f"符合條件的路線數量: {len(valid_routes)}")
    print(f"因時間超過24小時被過濾: {filtered_out['duration']} 個檔案")
    if valid_routes:
        print(f"符合條件的檔案已複製到 {output_folder_name} 資料夾")


if __name__ == "__main__":
    main()
