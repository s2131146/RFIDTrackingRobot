import time
from typing import ClassVar, Counter, Optional, Tuple
import cv2
import numpy as np
import logging


class Obstacles:
    OBS_POS_CENTER: ClassVar[str] = "CENTER"
    OBS_POS_LEFT: ClassVar[str] = "LEFT"
    OBS_POS_RIGHT: ClassVar[str] = "RIGHT"
    OBS_POS_NONE: ClassVar[str] = "NONE"
    OBS_POS_FULL: ClassVar[str] = "FULL"

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        logger: logging.Logger,
        obstacle_distance_threshold: float = 500.0,
        min_obstacle_size: Tuple[int, int] = (50, 30),
        tolerance: float = 25.0,
        min_continuous_increase: int = 20,
        wall_distance_threshold: float = 20000.0,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = logger
        self.obstacle_distance_threshold = obstacle_distance_threshold
        self.tolerance = tolerance
        self.min_continuous_increase = min_continuous_increase
        self.detection_interval = 0.5  # 0.5秒のインターバル
        self.directions_history = []  # 検出方向の履歴
        self.wall_positions_history = []  # 壁位置の履歴
        self.avoid_walls_history = []  # 壁回避の履歴
        self.last_detection_time = time.time()  # 前回の結果集計時刻
        self.last_direction = self.OBS_POS_NONE
        self.wall_distance_threshold = wall_distance_threshold
        self.last_wall_position = self.OBS_POS_NONE
        self.last_wall_avoid = False

        # 中央20%の領域設定
        self.central_start = int(self.frame_width * 0.4)
        self.central_end = int(self.frame_width * 0.6)

        self.min_obstacle_size = min_obstacle_size

    def process_obstacles(
        self,
        depth_image: np.ndarray,
        target_x: int,
        target_bbox=None,
    ) -> Tuple[str, bool, Optional[np.ndarray], str, bool]:
        if depth_image is None:
            self.logger.warning("Depth image is None.")
            return self.OBS_POS_NONE, False, None, (100, 100), False

        # ターゲットのバウンディングボックスからX範囲を取得
        ignore_x_start = ignore_x_end = None
        if target_bbox:
            x, _, w, _ = target_bbox[0].bboxes
            ignore_x_start = x
            ignore_x_end = w

        target_x = target_x + (self.frame_width // 2)
        depth_image = cv2.flip(depth_image, 1)
        depth_image = depth_image[:, :-25]

        # 障害物が目の前か
        obstacle_mask_full = depth_image < self.obstacle_distance_threshold
        overall_direction = self.determine_obstacle_position_full(obstacle_mask_full)

        # 床パターンの検出（中央領域のみ）
        floor_mask = self.detect_floor(depth_image)

        # 壁の検出
        wall_mask = self.detect_walls(depth_image, floor_mask)
        wall_position = self.determine_wall_position(depth_image, wall_mask)

        # 画面の中央領域に壁が存在するかをチェック
        central_area = wall_mask[:, self.central_start : self.central_end]
        wall_in_center = np.any(central_area)

        # 壁の位置履歴に追加
        self.wall_positions_history.append(wall_position)
        self.avoid_walls_history.append(wall_in_center)

        # 中央領域のマスクを作成し、床部分を除外
        central_region = depth_image[:, self.central_start : self.central_end]
        central_mask = np.ones_like(central_region, dtype=bool)
        central_mask[floor_mask[:, self.central_start : self.central_end]] = False

        # メディアンフィルタでノイズ除去
        depth_filtered = cv2.medianBlur(central_region.astype(np.uint16), 5).astype(
            np.float32
        )

        # 障害物マスクの作成
        obstacle_mask = np.zeros_like(depth_filtered, dtype=bool)

        # 小さな障害物を除去
        obstacle_mask_cleaned = self.remove_small_obstacles(obstacle_mask)

        # 隙間を埋めて一続きの形状にするために膨張処理を行う
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask_dilated = cv2.dilate(
            obstacle_mask_cleaned.astype(np.uint8), kernel, iterations=1
        ).astype(bool)

        # 元の画像サイズに戻して矩形フィット
        full_mask = np.zeros_like(depth_image, dtype=bool)
        full_mask[:, self.central_start : self.central_end] = obstacle_mask_dilated
        obstacle_mask_rect = self.fit_rectangle(full_mask)

        # 障害物の左右判定
        direction = self.determine_obstacle_position(obstacle_mask_rect)

        # 履歴に追加
        self.directions_history.append(direction)

        # 0.5秒間隔で結果を更新
        current_time = time.time()
        most_common_direction = self.last_direction
        most_common_wall_position = self.last_wall_position
        most_common_avoid_wall = self.last_wall_avoid
        if current_time - self.last_detection_time >= self.detection_interval:
            # 最も頻出する方向を選択
            most_common_direction = Counter(self.directions_history).most_common(1)[0][
                0
            ]
            self.last_direction = most_common_direction

            # 壁の位置で最も頻出する位置を選択
            most_common_wall_position = Counter(
                self.wall_positions_history
            ).most_common(1)[0][0]
            self.last_wall_position = most_common_wall_position

            most_common_avoid_wall = Counter(self.avoid_walls_history).most_common(1)[
                0
            ][0]
            self.last_wall_avoid = most_common_avoid_wall

            # 履歴をリセットして次の集計に備える
            self.directions_history.clear()
            self.wall_positions_history.clear()
            self.avoid_walls_history.clear()
            self.last_detection_time = current_time

        # 障害物と床を視覚化（障害物は赤色、床は青色で塗りつぶし）
        obstacle_visual = self.visualize_obstacle(
            depth_image,
            obstacle_mask_rect,
            floor_mask,
            wall_mask,
            target_x,
            ignore_x_start,
            ignore_x_end,
        )

        if overall_direction:
            return self.OBS_POS_FULL, obstacle_visual

        # target_xが中央20%の領域内にある場合、障害物判定をスキップ
        if target_bbox:
            if ignore_x_start < self.central_end and ignore_x_end > self.central_start:
                return (
                    self.OBS_POS_NONE,
                    False,
                    obstacle_visual,
                    most_common_wall_position,
                    most_common_avoid_wall,
                )

        return (
            most_common_direction,
            most_common_direction != self.OBS_POS_NONE,
            obstacle_visual,
            most_common_wall_position,
            most_common_avoid_wall,
        )

    def determine_obstacle_position_full(self, obstacle_mask: np.ndarray) -> bool:
        """障害物が目の前にあるか判定

        Args:
            obstacle_mask (np.ndarray): 全体の障害物マスク

        Returns:
            bool: 結果
        """
        total_coverage = np.sum(obstacle_mask) / (
            obstacle_mask.shape[0] * obstacle_mask.shape[1]
        )

        if total_coverage >= 0.6:
            return True

        return False

    def determine_wall_position(
        self, depth_image: np.ndarray, wall_mask: np.ndarray
    ) -> str:
        """
        壁の位置を `wall_mask` と `depth_image` に基づいて判定します。

        Args:
            depth_image (np.ndarray): 深度画像
            wall_mask (np.ndarray): 壁と判断された部分のマスク（壁がある部分がTrue、それ以外はFalse）

        Returns:
            str: 壁の位置（LEFT, RIGHT, CENTER, NONE）
        """
        _, width = wall_mask.shape
        left_area = wall_mask[:, : width // 5]
        right_area = wall_mask[:, -width // 5 :]

        # 深度画像から左エリアと右エリアの距離データを抽出
        left_depth_values = depth_image[:, : width // 5][left_area]
        right_depth_values = depth_image[:, -width // 5 :][right_area]

        # 外れ値除去のために、手前の壁とみなせる深度範囲（例えば50cm以内）にフィルタ
        depth_threshold = 2000  # 例: 2000mm（2m）以内を壁とみなす
        left_depth_values = left_depth_values[
            (left_depth_values > 0) & (left_depth_values < depth_threshold)
        ]
        right_depth_values = right_depth_values[
            (right_depth_values > 0) & (right_depth_values < depth_threshold)
        ]

        # 左右のエリアに有効な距離が存在する場合はその最小距離を、存在しない場合は inf を設定
        left_distance = (
            np.min(left_depth_values) if left_depth_values.size > 0 else float("inf")
        )
        right_distance = (
            np.min(right_depth_values) if right_depth_values.size > 0 else float("inf")
        )

        logging.debug(f"WALL L: {left_distance} R: {right_distance}")

        # 両方が `inf` の場合は壁が検出されていないため `NONE` を返す
        if left_distance == float("inf") and right_distance == float("inf"):
            return self.OBS_POS_NONE

        # 距離がほぼ等しい場合のみ `CENTER` を返す
        if abs(left_distance - right_distance) <= 30:
            return self.OBS_POS_CENTER
        elif left_distance < right_distance:
            return self.OBS_POS_LEFT
        else:
            return self.OBS_POS_RIGHT

    def fit_rectangle(self, obstacle_mask: np.ndarray) -> np.ndarray:
        obstacle_mask_rect = np.zeros_like(obstacle_mask, dtype=bool)
        contours, _ = cv2.findContours(
            obstacle_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            obstacle_mask_rect[y : y + h, x : x + w] = (
                True  # 矩形の範囲を障害物としてマスク
            )

        return obstacle_mask_rect

    def detect_floor(self, depth_image: np.ndarray) -> np.ndarray:
        floor_mask = np.zeros_like(depth_image, dtype=bool)
        mid_height = self.frame_height // 2

        for col in range(self.central_start, self.central_end):
            depth_values = depth_image[
                self.frame_height - 1 : mid_height : -1, col
            ].astype(np.int32)
            depth_values = depth_values[(depth_values > 0) & (depth_values < 2000)]
            if len(depth_values) < self.min_continuous_increase:
                continue

            continuous_increase_count = 0
            for i in range(1, len(depth_values)):
                if 0 < depth_values[i] - depth_values[i - 1] < self.tolerance:
                    continuous_increase_count += 1
                    if continuous_increase_count >= self.min_continuous_increase:
                        floor_mask[mid_height : self.frame_height, col] = True
                        break
                else:
                    continuous_increase_count = 0

        return floor_mask

    def detect_walls(
        self, depth_image: np.ndarray, floor_mask: np.ndarray
    ) -> np.ndarray:
        """
        左右のエリアにおいて、深度が徐々に遠くなる部分を壁として検出します。
        壁の検出は、深度の増加が途切れるまで続けます。
        壁検出の距離を80cm以内に制限します。

        Args:
            depth_image (np.ndarray): 深度画像

        Returns:
            np.ndarray: 壁と判断された部分のマスク（壁がある部分がTrue、それ以外はFalse）
        """
        wall_mask = np.zeros_like(depth_image, dtype=bool)
        _, width = depth_image.shape

        # 左側の壁検出
        self._detect_wall_side(
            depth_image, floor_mask, wall_mask, 0, width // 2, step=1
        )

        # 右側の壁検出
        self._detect_wall_side(
            depth_image, floor_mask, wall_mask, width - 1, width // 2 - 1, step=-1
        )

        # 膨張処理と連結成分を使った矩形フィット処理
        return self._dilate_and_fit_rectangle(wall_mask)

    def _detect_wall_side(
        self, depth_image, floor_mask, wall_mask, start_col, end_col, step
    ):
        """特定の方向（左または右）での壁検出処理"""
        for col in range(start_col, end_col, step):
            depth_values = depth_image[:, col]
            depth_values = depth_values[
                (depth_values <= self.wall_distance_threshold) & (~floor_mask[:, col])
            ]
            if len(depth_values) < self.min_continuous_increase:
                continue

            for i in range(1, len(depth_values)):
                if 0 < depth_values[i] - depth_values[i - 1] < self.tolerance:
                    wall_mask[:, col] = True  # 壁としてマスク
                else:
                    break  # 増加が途切れた場合、検出を終了

    def _dilate_and_fit_rectangle(
        self, wall_mask: np.ndarray, min_wall_width=30
    ) -> np.ndarray:
        """膨張処理と連結成分を使って矩形領域にフィット"""
        kernel = np.ones((5, 5), np.uint8)
        wall_mask_dilated = cv2.dilate(
            wall_mask.astype(np.uint8), kernel, iterations=5  # 膨張処理の回数を増加
        ).astype(bool)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            wall_mask_dilated.astype(np.uint8), connectivity=8
        )
        wall_mask_rect = np.zeros_like(wall_mask_dilated, dtype=bool)

        # 最小横幅のフィルタリングを追加
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area > 100 and w >= min_wall_width:  # 最小横幅の条件を追加
                wall_mask_rect[y : y + h, x : x + w] = True

        return wall_mask_rect

    def remove_small_obstacles(self, obstacle_mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            obstacle_mask.astype(np.uint8), connectivity=8
        )

        cleaned_mask = np.zeros_like(obstacle_mask, dtype=bool)
        for label in range(1, num_labels):
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]

            if (
                width >= self.min_obstacle_size[0]
                and height >= self.min_obstacle_size[1]
            ):
                cleaned_mask[labels == label] = True

        return cleaned_mask

    def apply_colormap_to_depth(self, depth_image: np.ndarray) -> np.ndarray:
        # 深度画像を0-255の範囲に正規化
        depth_normalized = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # ガンマ補正を適用して、手前の変化を強調
        depth_normalized = np.power(depth_normalized / 255.0, 0.3) * 255
        depth_normalized = depth_normalized.astype(np.uint8)

        # カラーマップ「JET」を適用して、青から赤へのグラデーションを再現
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        return depth_colormap

    def visualize_obstacle(
        self,
        depth_image: np.ndarray,
        obstacle_mask: np.ndarray,
        floor_mask: np.ndarray,
        wall_mask: np.ndarray,
        target_x: int,
        ignore_x_start: int,
        ignore_x_end: int,
    ) -> np.ndarray:
        depth_rgb = self.apply_colormap_to_depth(depth_image)
        blue_overlay = np.zeros_like(depth_rgb, dtype=np.uint8)

        blue_overlay[floor_mask] = [0, 0, 255]
        depth_rgb = cv2.addWeighted(depth_rgb, 1, blue_overlay, 1, 0)

        red_overlay = np.zeros_like(depth_rgb, dtype=np.uint8)
        red_overlay[wall_mask] = [255, 0, 0]
        depth_rgb = cv2.addWeighted(depth_rgb, 0.7, red_overlay, 0.3, 0)

        red_intensity = 255 - cv2.normalize(
            depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        depth_rgb[obstacle_mask, 0] = red_intensity[obstacle_mask]
        depth_rgb[obstacle_mask, 1] = 0
        depth_rgb[obstacle_mask, 2] = 0

        cv2.line(
            depth_rgb,
            (self.central_start, 0),
            (self.central_start, self.frame_height),
            (255, 0, 165),
            2,
        )
        cv2.line(
            depth_rgb,
            (self.central_end, 0),
            (self.central_end, self.frame_height),
            (255, 0, 165),
            2,
        )
        cv2.line(
            depth_rgb,
            (ignore_x_start, 0),
            (ignore_x_start, self.frame_height),
            (0, 255, 165),
            2,
        )
        cv2.line(
            depth_rgb,
            (ignore_x_end, 0),
            (ignore_x_end, self.frame_height),
            (0, 255, 165),
            2,
        )
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
        central_area = obstacle_mask[:, self.central_start : self.central_end]

        # 境界部分を取得
        left_boundary = central_area[:, 0]  # 左端
        right_boundary = central_area[:, -1]  # 右端

        # 境界部分に障害物があるか判定
        left_boundary_exists = np.any(left_boundary)
        right_boundary_exists = np.any(right_boundary)

        # 横方向に80%以上障害物があるか判定
        horizontal_coverage = (
            np.sum(np.any(central_area, axis=0)) / central_area.shape[1]
        )

        # 横方向80%以上が障害物で埋まっている、または両端に障害物がある場合はCENTERと判定
        if horizontal_coverage >= 0.65 or (
            left_boundary_exists and right_boundary_exists
        ):
            return self.OBS_POS_CENTER

        # 左右に分けて存在判定
        left_exists = np.any(central_area[:, : central_area.shape[1] // 2])
        right_exists = np.any(central_area[:, central_area.shape[1] // 2 :])

        # 左右の存在状況でLEFTまたはRIGHTを返す
        if right_exists:
            return self.OBS_POS_RIGHT
        elif left_exists:
            return self.OBS_POS_LEFT
        else:
            return self.OBS_POS_NONE
