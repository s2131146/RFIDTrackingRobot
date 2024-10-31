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
        depth_difference_threshold: float = 70.0,  # ミリメートル
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
            target_bboxes (List[List[int]], optional): 対象のバウンディングボックス。各バウンディングボックスは [x1, y1, x2, y2]。
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
                self.logger.debug("Person detected using contour.")
                # 人の下部の領域を障害物とする
                obstacle_below_person = self.get_obstacles_below_person(
                    person_mask, depth_filtered.shape
                )
                if obstacle_below_person is not None:
                    overlapping_labels = self.get_labels_from_mask(
                        obstacle_below_person, labels
                    )
                    if overlapping_labels:
                        obstacle_present = True
                        self.logger.debug("Obstacles detected below person.")
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
        人の輪郭を検出し、マスクを返す

        Args:
            depth_filtered (numpy.ndarray): フィルタリング済みの深度画像
            target_bboxes (List[List[int]]): 対象のバウンディングボックスリスト

        Returns:
            Optional[np.ndarray]: 人のマスク
        """
        # ここでは最初のバウンディングボックスを使用
        if not target_bboxes:
            self.logger.debug("No target bounding boxes provided.")
            return None

        x1, y1, x2, y2 = target_bboxes[0]
        self.logger.debug(f"Using bounding box: {(x1, y1, x2, y2)}")

        # 人の領域を抽出
        person_roi = depth_filtered[y1:y2, x1:x2]

        # 人の領域を正規化して8ビットに変換
        person_roi_normalized = cv2.normalize(
            person_roi, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # 二値化
        _, person_mask = cv2.threshold(
            person_roi_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 輪郭検出
        contours, _ = cv2.findContours(
            person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            self.logger.debug("No contours found in person ROI.")
            return None

        # 最大の輪郭を人とみなす
        largest_contour = max(contours, key=cv2.contourArea)

        # 人のマスクを作成
        person_mask_full = np.zeros(depth_filtered.shape, dtype=np.uint8)
        cv2.drawContours(
            person_mask_full[y1:y2, x1:x2],
            [largest_contour],
            -1,
            color=255,
            thickness=cv2.FILLED,
        )

        return person_mask_full.astype(bool)

    def get_obstacles_below_person(
        self, person_mask: np.ndarray, image_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        人の輪郭線より下の領域を障害物とみなす

        Args:
            person_mask (np.ndarray): 人のマスク
            image_shape (Tuple[int, int]): 画像の形状

        Returns:
            Optional[np.ndarray]: 障害物のマスク
        """
        # 人の下端の座標を取得
        ys, xs = np.where(person_mask)
        if ys.size == 0:
            self.logger.debug("Person mask is empty.")
            return None

        bottom_y = ys.max()
        self.logger.debug(f"Bottom Y coordinate of person: {bottom_y}")

        # 人の下の領域を障害物とする
        obstacle_mask = np.zeros(image_shape, dtype=bool)
        obstacle_mask[bottom_y + 1 :, :] = True

        return obstacle_mask

    def get_labels_from_mask(self, mask: np.ndarray, labels: np.ndarray) -> List[int]:
        """
        マスク内のラベルを取得

        Args:
            mask (np.ndarray): マスク
            labels (np.ndarray): ラベルマップ

        Returns:
            List[int]: マスク内のラベルリスト
        """
        label_values = np.unique(labels[mask])
        label_values = label_values[label_values != 0]  # 背景ラベルを除外
        self.logger.debug(f"Labels in mask: {label_values}")
        return label_values.tolist()

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
        mask[0] = 0  # ラベル0は背景

        filtered_mask = mask[labels]
        kept_labels = np.unique(labels[filtered_mask])
        kept_labels = kept_labels[kept_labels != 0]
        self.logger.debug(
            f"Filtered obstacles based on size. Labels kept: {kept_labels}"
        )

        return filtered_mask, labels, stats

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
