# obstacles.py

import cv2
import numpy as np
from typing import List, Tuple, Optional


class Obstacles:
    """障害物検出クラス: 深度情報を使用して障害物と安全な通路を検出・可視化する"""

    def __init__(self, frame_width: int, frame_height: int, logger):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = logger

        # 深度閾値設定（ミリメートル）
        self.obstacle_distance_threshold = 900  # 0.9メートル以内を障害物とみなす
        self.safe_floor_max_distance = 5000  # 最大深度（ミリメートル）

        # 壁除外マージン（ピクセル）
        self.wall_exclusion_margin = 50

        # ロボットが通れる最小の幅（ピクセル）
        self.min_safe_width = 50  # 例: 50ピクセル

    def create_obstacle_mask(
        self, depth_clipped: np.ndarray, target_bboxes: List[List[int]]
    ) -> np.ndarray:
        # ノイズ削減のためガウシアンブラーを適用
        depth_blurred = cv2.GaussianBlur(depth_clipped, (5, 5), 0)

        obstacle_mask = cv2.inRange(depth_blurred, 0, self.obstacle_distance_threshold)

        # 壁のマージンを除外
        obstacle_mask[:, : self.wall_exclusion_margin] = 0
        obstacle_mask[:, -self.wall_exclusion_margin :] = 0

        # 追跡対象を除外
        for bbox in target_bboxes:
            x1, y1, x2, y2 = bbox
            obstacle_mask[y1:y2, x1:x2] = 0

        # ノイズ除去（オープニングとクロージング）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

        return obstacle_mask

    def find_safe_x_coordinate(self, obstacle_mask: np.ndarray, target_x: int) -> int:
        """
        障害物マスクから安全な領域を検出し、対象のX座標に最も近い安全なX座標を返す。

        Args:
            obstacle_mask (numpy.ndarray): 障害物マスク
            target_x (int): 追跡対象のX座標（画面中心が0、左が負、右が正）

        Returns:
            int: 安全なX座標（画面中心が0、左が負、右が正）
        """
        # 各X座標ごとの障害物ピクセル数を計算
        obstacle_columns = np.sum(obstacle_mask, axis=0)

        # 障害物がない（ピクセル値が0）のX座標を特定
        safe_columns = obstacle_columns == 0

        # 壁のマージンを除外
        safe_columns[: self.wall_exclusion_margin] = False
        safe_columns[-self.wall_exclusion_margin :] = False

        # 安全な領域を見つける
        safe_regions = []
        start_idx = None
        for idx, is_safe in enumerate(safe_columns):
            if is_safe and start_idx is None:
                start_idx = idx
            elif not is_safe and start_idx is not None:
                end_idx = idx - 1
                width = end_idx - start_idx + 1
                if width >= self.min_safe_width:
                    safe_regions.append((start_idx, end_idx))
                start_idx = None
        # 最後まで安全な領域が続いていた場合
        if start_idx is not None:
            end_idx = len(safe_columns) - 1
            width = end_idx - start_idx + 1
            if width >= self.min_safe_width:
                safe_regions.append((start_idx, end_idx))

        if not safe_regions:
            # 安全な領域がない場合、対象の位置に基づいて左端または右端を選択
            self.logger.debug("No safe regions found.")
            if target_x < 0:
                # 対象が左側にいる場合、左端を選択
                safe_x = -self.frame_width // 2
            else:
                # 対象が右側にいる場合、右端を選択
                safe_x = self.frame_width // 2
            self.logger.debug(
                f"Defaulting Safe X to {safe_x} based on target position."
            )
            return safe_x

        # 各安全な領域の中心X座標を計算
        safe_centers = [(start + end) // 2 for start, end in safe_regions]

        # X座標を画面中心が0、左が負、右が正に変換
        safe_centers = [x - self.frame_width // 2 for x in safe_centers]

        # 追跡対象のX座標に最も近い安全な領域を選択
        closest_safe_x = min(safe_centers, key=lambda x: abs(x - target_x))

        self.logger.debug(f"Safe regions: {safe_regions}")
        self.logger.debug(f"Safe centers: {safe_centers}")
        self.logger.debug(f"Selected Safe X: {closest_safe_x}")

        return closest_safe_x

    def visualize_depth_and_obstacles(
        self,
        depth_image: np.ndarray,
        obstacle_mask: np.ndarray,
        safe_x: int,
    ) -> np.ndarray:
        """深度情報と障害物を視覚化し、結果を0.35倍にリサイズして返す

        Args:
            depth_image (numpy.ndarray): 深度画像
            obstacle_mask (numpy.ndarray): 障害物マスク
            safe_x (int): 安全なX座標（画面中心が0、左が負、右が正）

        Returns:
            numpy.ndarray: リサイズされた視覚化画像
        """
        if depth_image is None:
            self.logger.warning("Depth image is None. Cannot visualize depth.")
            return None

        # 深度画像をグレースケールに正規化（手前が白、奥が黒）
        depth_normalized = cv2.normalize(depth_image, None, 255, 0, cv2.NORM_MINMAX)
        depth_gray = depth_normalized.astype(np.uint8)
        depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

        # 障害物を赤色でオーバーレイ
        if cv2.countNonZero(obstacle_mask) > 0:
            red_overlay = np.zeros_like(depth_bgr, dtype=np.uint8)
            red_overlay[:] = (0, 0, 255)  # BGR形式で赤色
            mask_red = cv2.merge([obstacle_mask, obstacle_mask, obstacle_mask])
            depth_bgr = np.where(mask_red == 255, red_overlay, depth_bgr)

        # 安全なX座標を緑色の線で表示（BGR形式）
        # safe_x を画面座標に変換
        safe_x_screen = safe_x + self.frame_width // 2

        cv2.line(
            depth_bgr,
            (safe_x_screen, 0),
            (safe_x_screen, self.frame_height),
            (0, 255, 0),  # BGR形式で緑色
            2,
        )
        cv2.putText(
            depth_bgr,
            f"Safe X: {safe_x}",
            (safe_x_screen + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # BGR形式で緑色
            2,
        )

        # 画像を0.35倍にリサイズ
        scale_factor = 0.35
        resized_depth_bgr = cv2.resize(
            depth_bgr,
            (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_depth_bgr

    def process_obstacles(
        self,
        frame: np.ndarray,
        depth_image: np.ndarray = None,
        target_bboxes: List[List[int]] = [],
        target_x: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool, int, Optional[np.ndarray]]:
        """障害物を処理し、視覚化画像を返す

        Args:
            frame (numpy.ndarray): フレーム画像
            depth_image (numpy.ndarray, optional): 深度画像. Defaults to None.
            target_bboxes (List[List[int]], optional): 対象のバウンディングボックス. Defaults to [].
            target_x (int, optional): 対象のX座標（画面中心が0、左が負、右が正）. Defaults to None.

        Returns:
            Tuple[numpy.ndarray, bool, int, Optional[numpy.ndarray]]: フレーム画像、障害物の有無、障害物回避の中心X座標（画面中心が0、左が負、右が正）、視覚化画像
        """
        if depth_image is not None:
            depth_clipped = np.clip(depth_image, 0, self.safe_floor_max_distance)

            obstacle_mask = self.create_obstacle_mask(depth_clipped, target_bboxes)

            # target_x が渡されていない場合は中央をデフォルトとする
            if target_x is None:
                target_x = 0  # 画面中心が0

            # Safe Xを計算
            safe_x = self.find_safe_x_coordinate(obstacle_mask, target_x)
            self.logger.debug(f"Calculated Safe X: {safe_x}")

            depth_frame = self.visualize_depth_and_obstacles(
                depth_clipped, obstacle_mask, safe_x
            )

            obstacle_present = np.any(obstacle_mask == 255)

            return frame, not obstacle_present, safe_x, depth_frame
        else:
            return frame, True, 0, None  # 画面中心が0
