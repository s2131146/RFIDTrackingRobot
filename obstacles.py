import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging


class Obstacles:
    """障害物検出クラス: 深度情報を使用して障害物と安全な通路を検出・可視化する"""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        logger: logging.Logger,
        wall_exclusion_margin: int = 50,
        obstacle_distance_threshold: float = 900.0,  # ミリメートル
        min_safe_width: int = 50,
        min_obstacle_size: Tuple[int, int] = (30, 30),  # (縦, 横) ピクセル
        max_obstacle_size: Tuple[int, int] = (300, 300),  # (縦, 横) ピクセル
        consecutive_overlap_pixels: int = 20,
        depth_difference_threshold: float = 200.0,  # 平均深度 ±20cm
        obstacle_person_distance: float = 200.0,  # 人から25cm近いもの
    ):
        """
        初期化メソッド

        Args:
            frame_width (int): フレームの幅（ピクセル）
            frame_height (int): フレームの高さ（ピクセル）
            logger (logging.Logger): ログ出力用のロガー
            wall_exclusion_margin (int, optional): 壁近くを除外するマージン（ピクセル）
            obstacle_distance_threshold (float, optional): 障害物とみなす深度の閾値（ミリメートル）
            min_safe_width (int, optional): ロボットが通過できる最小の幅（ピクセル）
            min_obstacle_size (Tuple[int, int], optional): 障害物と認識する最小の縦横サイズ（ピクセル）
            max_obstacle_size (Tuple[int, int], optional): 障害物と認識する最大の縦横サイズ（ピクセル）
            consecutive_overlap_pixels (int, optional): 重なりとみなす連続ピクセル数
            depth_difference_threshold (float, optional): 深度差の閾値（ミリメートル）
            obstacle_person_distance (float, optional): 人から近いとみなす深度差（ミリメートル）
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = logger

        self.obstacle_distance_threshold = obstacle_distance_threshold
        self.wall_exclusion_margin = wall_exclusion_margin
        self.min_safe_width = min_safe_width
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.consecutive_overlap_pixels = consecutive_overlap_pixels
        self.depth_difference_threshold = depth_difference_threshold
        self.obstacle_person_distance = obstacle_person_distance

        self.last_target_x: Optional[int] = None

    def process_obstacles(
        self,
        depth_image: np.ndarray,
        target_bboxes: List[List[int]] = [],
        target_x: Optional[int] = None,
    ) -> Tuple[int, bool, Optional[np.ndarray]]:
        """
        障害物を処理し、安全なX座標を計算する

        Args:
            depth_image (numpy.ndarray): 深度画像（左右反転済み）
            target_bboxes (List[List[int]], optional): 対象のバウンディングボックスリスト。各バウンディングボックスは [x1, y1, x2, y2]。
            target_x (int, optional): ターゲットのX座標（画面中心が0、左が負、右が正）

        Returns:
            Tuple[int, bool, Optional[np.ndarray]]:
                - 安全なX座標（画面中心基準、左が負、右が正）
                - 障害物の有無（True: 障害物なし, False: 障害物あり）
                - 視覚化画像（RGB形式の画像。障害物と安全なX座標が表示されている）
        """
        if depth_image is None:
            self.logger.warning("Depth image is None. Cannot process obstacles.")
            return 0, True, None

        if depth_image.ndim != 2:
            self.logger.error(
                f"Depth image must be 2D. Received shape: {depth_image.shape}"
            )
            return 0, True, None

        # 深度画像を左右反転
        depth_image_flipped = cv2.flip(depth_image, 1)
        self.logger.debug(f"Depth image flipped. Shape: {depth_image_flipped.shape}")

        # メディアンフィルタでノイズ除去
        depth_filtered = cv2.medianBlur(
            depth_image_flipped.astype(np.uint16), 5
        ).astype(np.float32)
        self.logger.debug("Applied median filter to depth image.")

        # 障害物マスクを生成（深度閾値以下）
        obstacle_mask = depth_filtered < self.obstacle_distance_threshold
        self.logger.debug(
            f"Initial obstacle mask created. Total obstacles (pixels): {np.sum(obstacle_mask)}"
        )

        # 連結成分分析を実施して障害物をラベリング
        obstacle_mask_cleaned, labels, stats = self.filter_obstacles(obstacle_mask)
        self.logger.debug(
            f"Obstacle mask after filtering small obstacles. Total obstacles (pixels): {np.sum(obstacle_mask_cleaned)}"
        )

        # Safe Xの初期設定
        if target_bboxes and target_x is not None:
            self.logger.debug(f"Target detected at X: {target_x}")
            self.last_target_x = target_x
            safe_x = target_x
        else:
            if self.last_target_x is not None:
                self.logger.debug(
                    f"No target detected. Using last_target_x: {self.last_target_x}"
                )
                safe_x = self.last_target_x
            else:
                self.logger.debug(
                    "No target detected and no last_target_x. Defaulting Safe X to center."
                )
                safe_x = 0

        safe_x_screen = safe_x + (self.frame_width // 2)
        self.logger.debug(f"Safe X in screen coordinates: {safe_x_screen}")

        obstacle_present = False

        person_mask = None
        overlapping_labels = []

        if target_bboxes and target_x is not None:
            # 人の検出
            person_mask = self.detect_person_contour(depth_filtered, target_bboxes)
            if person_mask is not None:
                self.logger.debug("Person detected using depth contour.")

                # 人と障害物の平均深度を比較して、重なっている障害物を検出
                overlapping_labels = self.find_overlapping_obstacles(
                    depth_filtered, labels, person_mask
                )
                if overlapping_labels:
                    obstacle_present = True
                    self.logger.debug("Obstacles detected overlapping with person.")
            else:
                self.logger.debug("No person detected.")
                obstacle_present = False
        else:
            self.logger.debug(
                "No target detected. Finding Safe X based on last_target_x or center."
            )
            safe_x = self.find_safe_x(
                depth_filtered, safe_x_screen, obstacle_mask_cleaned
            )
            obstacle_present = safe_x != (self.frame_width // 2)

        safe_x_centered = safe_x - (self.frame_width // 2)
        self.logger.debug(f"Final Safe X (centered): {safe_x_centered}")

        # 視覚化画像の生成
        depth_frame = self.visualize_depth(
            depth_filtered,
            safe_x_screen,
            labels,
            overlapping_labels,
            person_mask,
        )

        return safe_x_centered, not obstacle_present, depth_frame

    def detect_person_contour(
        self, depth_filtered: np.ndarray, target_bboxes: List[List[int]]
    ) -> Optional[np.ndarray]:
        """
        深度情報を用いて、人の輪郭を検出し、マスクを返す

        Args:
            depth_filtered (numpy.ndarray): フィルタリング済みの深度画像
            target_bboxes (List[List[int]]): 対象のバウンディングボックスリスト

        Returns:
            Optional[numpy.ndarray]: 人のマスク
        """
        if not target_bboxes:
            self.logger.debug("No target bounding boxes provided.")
            return None

        x1, y1, x2, y2 = target_bboxes[0]
        self.logger.debug(f"Using bounding box: {(x1, y1, x2, y2)}")

        # バウンディングボックス内の平均深度を計算
        person_roi = depth_filtered[y1:y2, x1:x2]
        if person_roi.size == 0:
            self.logger.debug("Person ROI is empty.")
            return None

        person_depth = np.nanmean(person_roi)
        self.logger.debug(f"Person average depth: {person_depth}")

        # 深度画像全体から、平均深度 ±15cm の範囲のピクセルをマスク
        depth_diff = np.abs(depth_filtered - person_depth)
        person_mask_initial = depth_diff <= self.depth_difference_threshold

        # 連結成分を取得
        num_labels, labels = cv2.connectedComponents(
            person_mask_initial.astype(np.uint8), connectivity=8
        )
        self.logger.debug(f"Connected components in person mask: {num_labels}")

        # ラベルごとの平均深度を計算し、同一の人や障害物の判定基準（±15cm）を適用
        person_labels = []
        for label in range(1, num_labels):
            label_mask = labels == label
            label_depth = np.nanmean(depth_filtered[label_mask])
            if np.abs(label_depth - person_depth) <= self.depth_difference_threshold:
                person_labels.append(label)

        if not person_labels:
            self.logger.debug("No labels within depth threshold for person.")
            return None

        # 人のマスクを作成（複数のラベルを含む場合もある）
        person_mask = np.isin(labels, person_labels)
        self.logger.debug(f"Person labels determined: {person_labels}")

        return person_mask

    def find_overlapping_obstacles(
        self,
        depth_filtered: np.ndarray,
        labels: np.ndarray,
        person_mask: np.ndarray,
    ) -> List[int]:
        """
        人と重なっている障害物のラベルを検出

        Args:
            depth_filtered (np.ndarray): フィルタリング済みの深度画像
            labels (np.ndarray): ラベルマップ
            person_mask (np.ndarray): 人のマスク

        Returns:
            List[int]: 重なっている障害物のラベルリスト
        """
        # 人の平均深度を再計算
        person_depth = np.nanmean(depth_filtered[person_mask])
        self.logger.debug(f"Person depth for overlapping detection: {person_depth}")

        overlapping_labels = []
        # 障害物ラベルのリストを取得
        unique_labels = np.unique(labels[labels > 0])

        for label in unique_labels:
            if label == 0:
                continue
            obstacle_mask = labels == label
            # 障害物と人のマスクの重なりを確認
            overlap = np.logical_and(person_mask, obstacle_mask)
            if np.any(overlap):
                # 障害物の平均深度を計算
                obstacle_depth = np.nanmean(depth_filtered[obstacle_mask])
                depth_diff = person_depth - obstacle_depth
                if depth_diff >= self.obstacle_person_distance:
                    overlapping_labels.append(label)
                    self.logger.debug(
                        f"Obstacle label {label} overlaps with person and is {depth_diff}mm closer."
                    )

        return overlapping_labels

    def filter_obstacles(
        self, obstacle_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        連結成分分析を使用して、小さな障害物や大きすぎる障害物を除外する

        Args:
            obstacle_mask (numpy.ndarray): 初期の障害物マスク

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: フィルタリング後の障害物マスク、ラベル、統計情報
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            obstacle_mask.astype(np.uint8), connectivity=8
        )
        self.logger.debug(f"Connected components found: {num_labels}")

        # 最小および最大サイズを満たすラベルのみを保持
        mask = (
            (stats[:, cv2.CC_STAT_WIDTH] >= self.min_obstacle_size[1])
            & (stats[:, cv2.CC_STAT_HEIGHT] >= self.min_obstacle_size[0])
            & (stats[:, cv2.CC_STAT_WIDTH] <= self.max_obstacle_size[1])
            & (stats[:, cv2.CC_STAT_HEIGHT] <= self.max_obstacle_size[0])
        )
        mask[0] = False  # ラベル0は背景

        filtered_mask = mask[labels]
        kept_labels = np.unique(labels[filtered_mask])
        kept_labels = kept_labels[kept_labels != 0]
        self.logger.debug(
            f"Filtered obstacles based on size. Labels kept: {kept_labels}"
        )

        # フィルタリング後の障害物マスクを作成
        obstacle_mask_filtered = np.isin(labels, kept_labels)

        return obstacle_mask_filtered, labels, stats

    def visualize_depth(
        self,
        depth_filtered: np.ndarray,
        safe_x: int,
        labels: np.ndarray,
        overlapping_labels: List[int],
        person_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        深度画像と安全なX座標を視覚化する

        Args:
            depth_filtered (numpy.ndarray): ノイズ除去された深度画像
            safe_x (int): 安全なX座標（画面座標）
            labels (numpy.ndarray): 連結成分ラベル
            overlapping_labels (List[int]): 人と重なっている障害物のラベル番号リスト
            person_mask (Optional[np.ndarray]): 人のマスク

        Returns:
            numpy.ndarray: 視覚化されたRGB画像
        """
        # 深度画像の正規化
        depth_normalized = cv2.normalize(
            depth_filtered,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        self.logger.debug("Depth image normalized.")

        # グレースケール画像をRGBに変換
        depth_gray_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
        self.logger.debug("Depth image converted to RGB grayscale.")

        # 視覚化画像の初期化
        visualization_image = depth_gray_rgb.copy()

        # その他の障害物を赤色で表示
        obstacle_mask = labels > 0
        for label in overlapping_labels:
            obstacle_mask[labels == label] = False  # 重なっている障害物は後で色付け
        if person_mask is not None:
            obstacle_mask = obstacle_mask & (~person_mask)  # 人のマスクを除外
        visualization_image[obstacle_mask] = [255, 0, 0]  # 赤色 (RGB)
        self.logger.debug("Other obstacles colored red.")

        # 人と重なっている障害物を黄色で表示
        for label in overlapping_labels:
            overlap_mask = labels == label
            visualization_image[overlap_mask] = [255, 255, 0]  # 黄色 (RGB)
            self.logger.debug(f"Overlapping obstacle Label {label} colored yellow.")

        # 人を青色で表示
        if person_mask is not None:
            visualization_image[person_mask] = [0, 0, 255]  # 青色 (RGB)
            self.logger.debug("Person colored blue.")

        # 安全なX座標を緑色の線で表示
        cv2.line(
            visualization_image,
            (safe_x, 0),
            (safe_x, self.frame_height),
            (0, 255, 0),  # 緑色 (RGB)
            2,
        )
        cv2.putText(
            visualization_image,
            f"Safe X: {safe_x - (self.frame_width // 2)}",
            (safe_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # 緑色 (RGB)
            2,
        )
        self.logger.debug(f"Safe X line and text added at X={safe_x}")

        # 画像をリサイズ（オプション）
        resized_visualization_image = cv2.resize(
            visualization_image,
            (int(self.frame_width * 0.35), int(self.frame_height * 0.35)),
            interpolation=cv2.INTER_LINEAR,
        )
        self.logger.debug(
            f"Visualization image resized. New shape: {resized_visualization_image.shape}"
        )

        return resized_visualization_image

    def find_safe_x(
        self,
        depth_filtered: np.ndarray,
        safe_x_screen: int,
        obstacle_mask_cleaned: np.ndarray,
    ) -> int:
        """
        安全なX座標を見つける（詳細な実装は省略）

        Args:
            depth_filtered (numpy.ndarray): ノイズ除去された深度画像
            safe_x_screen (int): 現在のSafe Xの画面座標
            obstacle_mask_cleaned (numpy.ndarray): フィルタリング後の障害物マスク

        Returns:
            int: 新しい安全なX座標（画面座標）
        """
        # ここでは簡略化のため、現在のsafe_x_screenを返す
        return safe_x_screen
