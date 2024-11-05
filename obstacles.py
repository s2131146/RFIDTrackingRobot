import time
from typing import ClassVar, Counter, Optional, Tuple
import cv2
import numpy as np
import logging

from constants import Commands

class Obstacles:
    OBS_POS_CENTER: ClassVar[str] = "CENTER"
    OBS_POS_LEFT: ClassVar[str] = "LEFT"
    OBS_POS_RIGHT: ClassVar[str] = "RIGHT"
    OBS_POS_NONE: ClassVar[str] = "NONE"

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        logger: logging.Logger,
        obstacle_distance_threshold: float = 500.0,
        min_obstacle_size: Tuple[int, int] = (50, 30),
        tolerance: float = 100.0,
        min_continuous_increase: int = 20
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = logger
        self.obstacle_distance_threshold = obstacle_distance_threshold
        self.tolerance = tolerance
        self.min_continuous_increase = min_continuous_increase
        self.detection_interval = 0.5  # 0.5秒のインターバル
        self.directions_history = []  # 検出方向の履歴
        self.last_detection_time = time.time()  # 前回の結果集計時刻
        self.last_direction = self.OBS_POS_NONE
        
        # 中央20%の領域設定
        self.central_start = int(self.frame_width * 0.4)
        self.central_end = int(self.frame_width * 0.6)

        self.min_obstacle_size = min_obstacle_size

    def process_obstacles(self, depth_image: np.ndarray, target_x: int) -> Tuple[str, bool, Optional[np.ndarray]]:
        if depth_image is None:
            self.logger.warning("Depth image is None.")
            return self.OBS_POS_NONE, False, None

        target_x = target_x + (self.frame_width // 2)        
        depth_image = cv2.flip(depth_image, 1)

        # 床パターンの検出（中央領域のみ）
        floor_mask = self.detect_floor(depth_image)

        # 中央領域のマスクを作成し、床部分を除外
        central_region = depth_image[:, self.central_start:self.central_end]
        central_mask = np.ones_like(central_region, dtype=bool)
        central_mask[floor_mask[:, self.central_start:self.central_end]] = False

        # メディアンフィルタでノイズ除去
        depth_filtered = cv2.medianBlur(central_region.astype(np.uint16), 5).astype(np.float32)

        # 障害物マスクの作成
        obstacle_mask = (depth_filtered < self.obstacle_distance_threshold) & central_mask

        # 小さな障害物を除去
        obstacle_mask_cleaned = self.remove_small_obstacles(obstacle_mask)

        # 隙間を埋めて一続きの形状にするために膨張処理を行う
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask_dilated = cv2.dilate(obstacle_mask_cleaned.astype(np.uint8), kernel, iterations=1).astype(bool)

        # 元の画像サイズに戻して矩形フィット
        full_mask = np.zeros_like(depth_image, dtype=bool)
        full_mask[:, self.central_start:self.central_end] = obstacle_mask_dilated
        obstacle_mask_rect = self.fit_rectangle(full_mask)

        # 障害物の左右判定
        direction = self.determine_obstacle_position(obstacle_mask_rect)

        # 履歴に追加
        self.directions_history.append(direction)

        # 0.5秒間隔で結果を更新
        current_time = time.time()
        most_common_direction = self.last_direction
        if current_time - self.last_detection_time >= self.detection_interval:
            # 最も頻出する方向を選択
            most_common_direction = Counter(self.directions_history).most_common(1)[0][0]
            self.last_direction = most_common_direction

            # 履歴をリセットして次の集計に備える
            self.directions_history.clear()
            self.last_detection_time = current_time
            
        # 障害物と床を視覚化（障害物は赤色、床は青色で塗りつぶし）
        obstacle_visual = self.visualize_obstacle(depth_image, obstacle_mask_rect, floor_mask, target_x)

        # target_xが中央20%の領域内にある場合、障害物判定をスキップ
        if self.central_start <= target_x < self.central_end:
            return self.OBS_POS_NONE, False, obstacle_visual
        
        return most_common_direction, most_common_direction != self.OBS_POS_NONE, obstacle_visual

    def fit_rectangle(self, obstacle_mask: np.ndarray) -> np.ndarray:
        obstacle_mask_rect = np.zeros_like(obstacle_mask, dtype=bool)
        contours, _ = cv2.findContours(obstacle_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            obstacle_mask_rect[y:y+h, x:x+w] = True  # 矩形の範囲を障害物としてマスク

        return obstacle_mask_rect

    def detect_floor(self, depth_image: np.ndarray) -> np.ndarray:
        floor_mask = np.zeros_like(depth_image, dtype=bool)
        mid_height = self.frame_height // 2

        for col in range(self.central_start, self.central_end):
            depth_values = depth_image[self.frame_height-1:mid_height:-1, col].astype(np.int32)
            depth_values = depth_values[(depth_values > 0) & (depth_values < 2000)]
            if len(depth_values) < self.min_continuous_increase:
                continue

            continuous_increase_count = 0
            for i in range(1, len(depth_values)):
                if 0 < depth_values[i] - depth_values[i - 1] < self.tolerance:
                    continuous_increase_count += 1
                    if continuous_increase_count >= self.min_continuous_increase:
                        floor_mask[mid_height:self.frame_height, col] = True
                        break
                else:
                    continuous_increase_count = 0

        return floor_mask
    
    def remove_small_obstacles(self, obstacle_mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obstacle_mask.astype(np.uint8), connectivity=8)

        cleaned_mask = np.zeros_like(obstacle_mask, dtype=bool)
        for label in range(1, num_labels):
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]

            if width >= self.min_obstacle_size[0] and height >= self.min_obstacle_size[1]:
                cleaned_mask[labels == label] = True

        return cleaned_mask

    def visualize_obstacle(self, depth_image: np.ndarray, obstacle_mask: np.ndarray, floor_mask: np.ndarray, target_x: int) -> np.ndarray:
        depth_normalized = cv2.normalize(
            depth_image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        depth_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)

        blue_overlay = np.zeros_like(depth_rgb, dtype=np.uint8)
        blue_overlay[floor_mask] = [0, 0, 255]
        depth_rgb = cv2.addWeighted(depth_rgb, 0.7, blue_overlay, 0.3, 0)

        red_intensity = 255 - cv2.normalize(depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        depth_rgb[obstacle_mask, 0] = red_intensity[obstacle_mask]
        depth_rgb[obstacle_mask, 1] = 0
        depth_rgb[obstacle_mask, 2] = 0

        cv2.line(depth_rgb, (self.central_start, 0), (self.central_start, self.frame_height), (255, 0, 165), 2)
        cv2.line(depth_rgb, (self.central_end, 0), (self.central_end, self.frame_height), (255, 0, 165), 2)

        cv2.line(
            depth_rgb,
            (target_x, 0),
            (target_x, self.frame_height),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            depth_rgb,
            f"TARGET: {target_x - (self.frame_width // 2)}",
            (target_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        resized_visualization = cv2.resize(
            depth_rgb,
            (int(self.frame_width * 0.35), int(self.frame_height * 0.35)),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_visualization

    def determine_obstacle_position(self, obstacle_mask: np.ndarray) -> str:
        """
        中央20%のエリアをさらに左右で分割し、障害物の位置を判定。
        横方向に80%以上の障害物があればCENTERとし、
        または両端に障害物が触れている場合もCENTERとする。
        """
        # 中央20%のエリア
        central_area = obstacle_mask[:, self.central_start:self.central_end]
        
        # 境界部分を取得
        left_boundary = central_area[:, 0]  # 左端
        right_boundary = central_area[:, -1]  # 右端
        
        # 境界部分に障害物があるか判定
        left_boundary_exists = np.any(left_boundary)
        right_boundary_exists = np.any(right_boundary)
        
        # 横方向に80%以上障害物があるか判定
        horizontal_coverage = np.sum(np.any(central_area, axis=0)) / central_area.shape[1]
        
        # 横方向80%以上が障害物で埋まっている、または両端に障害物がある場合はCENTERと判定
        if horizontal_coverage >= 0.65 or (left_boundary_exists and right_boundary_exists):
            return self.OBS_POS_CENTER
        
        # 左右に分けて存在判定
        left_exists = np.any(central_area[:, :central_area.shape[1] // 2])
        right_exists = np.any(central_area[:, central_area.shape[1] // 2:])

        # 左右の存在状況でLEFTまたはRIGHTを返す
        if right_exists:
            return self.OBS_POS_RIGHT
        elif left_exists:
            return self.OBS_POS_LEFT
        else:
            return self.OBS_POS_NONE
