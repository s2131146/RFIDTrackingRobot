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
        obstacle_distance_min: float = 1000.0,  # ミリメートル (1メートル)
        obstacle_distance_max: float = 20000.0,  # ミリメートル (20メートル)
        background_depth_tolerance: float = 500.0,  # ミリメートル (±50cm)
        obstacle_person_distance: float = 250.0,  # ミリメートル (25cm)
        min_safe_width: int = 50,
        min_background_area: int = 5000,  # ピクセル
        min_obstacle_size: Tuple[int, int] = (30, 30),  # (縦, 横) ピクセル
        max_obstacle_size: Tuple[int, int] = (300, 300),  # (縦, 横) ピクセル
        consecutive_overlap_pixels: int = 20,
        margin_percentage: float = 0.10,  # マージンを10%に設定
    ):
        """
        初期化メソッド

        Args:
            frame_width (int): フレームの幅（ピクセル）
            frame_height (int): フレームの高さ（ピクセル）
            logger (logging.Logger): ログ出力用のロガー
            obstacle_distance_min (float, optional): 障害物とみなす深度の最小閾値（ミリメートル）
            obstacle_distance_max (float, optional): 障害物とみなす深度の最大閾値（ミリメートル）
            background_depth_tolerance (float, optional): 背景の深度許容範囲（ミリメートル）
            obstacle_person_distance (float, optional): 人と障害物の深度差の閾値（ミリメートル）
            min_safe_width (int, optional): ロボットが通過できる最小の幅（ピクセル）
            min_background_area (int, optional): 背景とみなす最小面積（ピクセル）
            min_obstacle_size (Tuple[int, int], optional): 障害物と認識する最小の縦横サイズ（ピクセル）
            max_obstacle_size (Tuple[int, int], optional): 障害物と認識する最大の縦横サイズ（ピクセル）
            consecutive_overlap_pixels (int, optional): 重なりとみなす連続ピクセル数
            margin_percentage (float, optional): 両端に設けるマージンの割合（0.10は10%）
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = logger

        self.obstacle_distance_min = obstacle_distance_min
        self.obstacle_distance_max = obstacle_distance_max
        self.background_depth_tolerance = background_depth_tolerance
        self.obstacle_person_distance = obstacle_person_distance
        self.min_safe_width = min_safe_width
        self.min_background_area = min_background_area
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.consecutive_overlap_pixels = consecutive_overlap_pixels

        self.margin_percentage = margin_percentage
        self.margin_width = int(self.frame_width * self.margin_percentage)

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

        # 反転前後の深度画像を視覚化（デバッグ用）
        cv2.imshow("Original Depth Image", depth_image)
        cv2.imshow("Flipped Depth Image", depth_image_flipped)
        cv2.waitKey(1)

        # メディアンフィルタでノイズ除去
        depth_filtered = cv2.medianBlur(
            depth_image_flipped.astype(np.uint16), 5
        ).astype(np.float32)
        self.logger.debug("Applied median filter to depth image.")

        # 深度画像の前処理: NaNやInfを処理
        depth_filtered = np.where(np.isfinite(depth_filtered), depth_filtered, 0)
        self.logger.debug("Replaced NaN and Inf in depth image with 0.")

        # 深度画像の統計情報を確認
        min_depth = depth_filtered.min()
        max_depth = depth_filtered.max()
        mean_depth = depth_filtered.mean()
        self.logger.debug(
            f"Depth Image Min: {min_depth} mm, Max: {max_depth} mm, Mean: {mean_depth} mm"
        )

        # 深度画像の欠損データを確認
        nan_count = np.isnan(depth_image).sum()
        inf_count = np.isinf(depth_image).sum()
        self.logger.debug(
            f"NaN count in original depth image: {nan_count}, Inf count: {inf_count}"
        )

        # 背景マスクを作成
        background_mask = self.detect_background(depth_filtered)
        self.logger.debug(
            f"Background mask created. Total background pixels: {np.sum(background_mask)}"
        )

        # 壁検出のための両端の深度情報を検出
        wall_depth_from_edges = self.detect_wall_from_edges(depth_filtered)

        # 壁マスクを作成
        if wall_depth_from_edges is not None:
            # 壁として認識する深度範囲を定義
            wall_mask_edges = depth_filtered >= (
                wall_depth_from_edges - self.background_depth_tolerance
            )
            wall_mask_edges = wall_mask_edges.astype(np.uint8)
            num_labels, labels_walls = cv2.connectedComponents(
                wall_mask_edges, connectivity=8
            )
            self.logger.debug(
                f"ConnectedComponents for wall from edges returned num_labels: {num_labels}"
            )

            wall_labels = []
            for label in range(1, num_labels):
                label_mask = labels_walls == label
                area = np.sum(label_mask)
                if area >= self.min_background_area:
                    wall_labels.append(label)
                    self.logger.debug(
                        f"Label {label} considered as wall from edges with area: {area}"
                    )

            final_wall_mask_edges = np.isin(labels_walls, wall_labels)
            self.logger.debug(
                f"Final wall mask from edges created. Total wall pixels: {np.sum(final_wall_mask_edges)}"
            )

            # 全ての壁を一つのマスクに統合
            wall_mask = background_mask | final_wall_mask_edges

            # 壁より後ろにある物体（深度が大きい）も壁としてマスク
            if wall_depth_from_edges > 0:
                wall_mask |= depth_filtered >= wall_depth_from_edges
                self.logger.debug(
                    f"Wall mask updated to include objects behind walls. Total wall pixels: {np.sum(wall_mask)}"
                )
        else:
            wall_mask = background_mask
            self.logger.debug("Using only background mask as wall mask.")

        self.logger.debug(
            f"Total wall pixels after all wall detection: {np.sum(wall_mask)}"
        )

        # 障害物マスクを作成（背景および壁よりも手前に存在する物体）
        obstacle_mask = (depth_filtered >= self.obstacle_distance_min) & (
            depth_filtered <= self.obstacle_distance_max
        )
        obstacle_mask = obstacle_mask & (~wall_mask)
        self.logger.debug(
            f"Obstacle mask after excluding walls and applying distance thresholds. Total obstacles (pixels): {np.sum(obstacle_mask)}"
        )

        # 連結成分分析を実施して障害物をラベリング
        obstacle_mask_cleaned, labels, stats = self.filter_obstacles(obstacle_mask)
        self.logger.debug(
            f"Obstacle mask after filtering by size. Total obstacles (pixels): {np.sum(obstacle_mask_cleaned)}"
        )

        # 視覚化用のデバッグコードを追加
        self.visualize_debug_masks(
            depth_filtered,
            background_mask,
            final_wall_mask_edges,
            wall_mask,
            obstacle_mask_cleaned,
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
            wall_mask,  # 背景と壁のマスクを渡す
        )

        return safe_x_centered, not obstacle_present, depth_frame

    def detect_background(self, depth_filtered: np.ndarray) -> np.ndarray:
        """
        深度画像から背景を検出する。背景は最も後ろに存在し、一定以上の面積を持つ物体とする。

        Args:
            depth_filtered (numpy.ndarray): フィルタリング済みの深度画像

        Returns:
            numpy.ndarray: 背景のマスク
        """
        # 最大深度の範囲を定義
        max_depth = np.nanmax(depth_filtered)
        self.logger.debug(f"Maximum depth detected: {max_depth} mm")

        # 最大深度に近い領域を背景としてマスク
        background_mask_initial = depth_filtered >= (
            max_depth - self.background_depth_tolerance
        )
        self.logger.debug(
            f"Initial background mask created. Pixels within tolerance: {np.sum(background_mask_initial)}"
        )

        # 連結成分分析を実施
        num_labels, labels = cv2.connectedComponents(
            background_mask_initial.astype(np.uint8), connectivity=8
        )
        self.logger.debug(
            f"ConnectedComponents for background mask returned num_labels: {num_labels}"
        )

        background_labels = []
        for label in range(1, num_labels):
            label_mask = labels == label
            area = np.sum(label_mask)
            if area >= self.min_background_area:
                background_labels.append(label)
                self.logger.debug(
                    f"Label {label} considered as background with area: {area}"
                )

        # 背景マスクの生成
        final_background_mask = np.isin(labels, background_labels)
        self.logger.debug(
            f"Final background mask created. Total background pixels: {np.sum(final_background_mask)}"
        )

        return final_background_mask

    def detect_wall_from_edges(self, depth_filtered: np.ndarray) -> Optional[float]:
        """
        画面の両端（マージン内）の深度情報を比較し、近ければその深度情報を持つ物体を壁としてまとめる。
        また、その壁より後ろにある物体の深度を返す。

        Args:
            depth_filtered (numpy.ndarray): フィルタリング済みの深度画像

        Returns:
            Optional[float]: 壁の深度。検出できなかった場合は None。
        """
        # マージンを計算
        margin = self.margin_width
        self.logger.debug(f"Using margin width: {margin} pixels")

        # 両端からマージンを考慮して深度情報を取得
        left_edge = depth_filtered[:, :margin]
        right_edge = depth_filtered[:, -margin:]

        # 両端の平均深度を計算
        left_depth = np.nanmean(left_edge)
        right_depth = np.nanmean(right_edge)
        self.logger.debug(
            f"Left edge average depth: {left_depth} mm, Right edge average depth: {right_depth} mm"
        )

        # 深度差が閾値以下であれば、壁の深度とみなす
        depth_difference = abs(left_depth - right_depth)
        depth_threshold = 2000.0  # ミリメートル（例: 2m）に設定
        self.logger.debug(
            f"Depth difference between edges: {depth_difference} mm (Threshold: {depth_threshold} mm)"
        )

        if depth_difference <= depth_threshold:
            # 壁の深度を平均値とする
            wall_depth = (left_depth + right_depth) / 2
            self.logger.debug(f"Wall detected from edges with depth: {wall_depth} mm")
            return wall_depth
        else:
            # 深い方の端の深度を壁の深度とする
            if left_depth > right_depth:
                wall_depth = left_depth
                self.logger.debug(f"Wall depth set to left edge depth: {wall_depth} mm")
            else:
                wall_depth = right_depth
                self.logger.debug(
                    f"Wall depth set to right edge depth: {wall_depth} mm"
                )
            return wall_depth

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
        # 形態学的変換でマスクをクリーンアップ
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.morphologyEx(
            obstacle_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
        )
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        obstacle_mask = obstacle_mask.astype(bool)
        self.logger.debug("Applied morphological operations to obstacle mask.")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            obstacle_mask.astype(np.uint8), connectivity=8
        )
        self.logger.debug(
            f"ConnectedComponentsWithStats returned num_labels: {num_labels}"
        )

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

        # フィルタリング後のラベル数をログに出力
        self.logger.debug(
            f"Filtered obstacles based on size. Labels kept: {kept_labels}, Count: {len(kept_labels)}"
        )

        return obstacle_mask_filtered, labels, stats

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

        # 複数のバウンディングボックスを処理
        person_masks = []
        for bbox in target_bboxes:
            x1, y1, x2, y2 = bbox
            self.logger.debug(f"Using bounding box: {(x1, y1, x2, y2)}")

            # バウンディングボックス内の平均深度を計算
            person_roi = depth_filtered[y1:y2, x1:x2]
            if person_roi.size == 0:
                self.logger.debug("Person ROI is empty.")
                continue

            person_depth = np.nanmean(person_roi)
            self.logger.debug(f"Person average depth: {person_depth} mm")

            # 深度画像全体から、平均深度 ±25cm の範囲のピクセルをマスク
            depth_diff = np.abs(depth_filtered - person_depth)
            person_mask_initial = depth_diff <= self.obstacle_person_distance

            # 連結成分を取得
            num_labels, labels = cv2.connectedComponents(
                person_mask_initial.astype(np.uint8), connectivity=8
            )
            self.logger.debug(f"Connected components in person mask: {num_labels}")

            # ラベルごとの平均深度を計算し、同一の人や障害物の判定基準（±25cm）を適用
            person_labels = []
            for label in range(1, num_labels):
                label_mask = labels == label
                label_depth = np.nanmean(depth_filtered[label_mask])
                if np.abs(label_depth - person_depth) <= self.obstacle_person_distance:
                    person_labels.append(label)

            if not person_labels:
                self.logger.debug("No labels within depth threshold for person.")
                continue

            # 人のマスクを作成（複数のラベルを含む場合もある）
            person_mask = np.isin(labels, person_labels)
            self.logger.debug(f"Person labels determined: {person_labels}")

            person_masks.append(person_mask)

        if not person_masks:
            return None

        # 全ての人物マスクを統合
        combined_person_mask = np.logical_or.reduce(person_masks)
        return combined_person_mask

    def find_overlapping_obstacles(
        self,
        depth_filtered: np.ndarray,
        labels: np.ndarray,
        person_mask: np.ndarray,
    ) -> List[int]:
        """
        人と重なっている障害物のラベルを検出

        Args:
            depth_filtered (numpy.ndarray): フィルタリング済みの深度画像
            labels (numpy.ndarray): ラベルマップ
            person_mask (numpy.ndarray): 人のマスク

        Returns:
            List[int]: 重なっている障害物のラベルリスト
        """
        # 人の平均深度を再計算
        person_depth = np.nanmean(depth_filtered[person_mask])
        self.logger.debug(f"Person depth for overlapping detection: {person_depth} mm")

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
                        f"Obstacle label {label} overlaps with person and is {depth_diff} mm closer."
                    )

        return overlapping_labels

    def visualize_depth(
        self,
        depth_filtered: np.ndarray,
        safe_x: int,
        labels: np.ndarray,
        overlapping_labels: List[int],
        person_mask: Optional[np.ndarray],
        wall_mask: np.ndarray,
    ) -> np.ndarray:
        """
        深度画像と安全なX座標を視覚化する

        Args:
            depth_filtered (numpy.ndarray): ノイズ除去された深度画像
            safe_x (int): 安全なX座標（画面座標）
            labels (numpy.ndarray): 連結成分ラベル
            overlapping_labels (List[int]): 人と重なっている障害物のラベル番号リスト
            person_mask (Optional[numpy.ndarray]): 人のマスク
            wall_mask (numpy.ndarray): 背景と壁のマスク

        Returns:
            numpy.ndarray: 視覚化されたRGB画像
        """
        # 深度画像の正規化（白黒のグラデーション）
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

        # 壁をシアン色で表示
        visualization_image[wall_mask] = [0, 255, 255]  # シアン (RGB)
        self.logger.debug("Walls colored cyan.")

        # 障害物を赤色で表示
        obstacle_mask = labels > 0
        for label in overlapping_labels:
            obstacle_mask[labels == label] = False  # 重なっている障害物は後で色付け
        if person_mask is not None:
            obstacle_mask = obstacle_mask & (~person_mask)  # 人のマスクを除外
        visualization_image[obstacle_mask] = [255, 0, 0]  # 赤色 (RGB)
        self.logger.debug("Other obstacles colored red.")

        # 人と重なっている障害物を黄色で表示（オプション）
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

        # 視覚化画像を保存（デバッグ用）
        cv2.imwrite("visualization_image.png", visualization_image)
        self.logger.debug("Saved visualization image as visualization_image.png")

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

    def visualize_debug_masks(
        self,
        depth_filtered: np.ndarray,
        background_mask: np.ndarray,
        final_wall_mask_edges: np.ndarray,
        wall_mask: np.ndarray,
        obstacle_mask_cleaned: np.ndarray,
    ):
        """
        マスクと深度画像を視覚化する。

        Args:
            depth_filtered (numpy.ndarray): フィルタリング済みの深度画像
            background_mask (numpy.ndarray): 背景マスク
            final_wall_mask_edges (numpy.ndarray): 両端から検出された壁マスク
            wall_mask (numpy.ndarray): 背景と壁のマスク
            obstacle_mask_cleaned (numpy.ndarray): フィルタリング後の障害物マスク
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

        # マスクを個別に保存
        cv2.imwrite("depth_normalized.png", depth_normalized)
        cv2.imwrite("background_mask.png", (background_mask * 255).astype(np.uint8))
        cv2.imwrite(
            "final_wall_mask_edges.png", (final_wall_mask_edges * 255).astype(np.uint8)
        )
        cv2.imwrite("wall_mask.png", (wall_mask * 255).astype(np.uint8))
        cv2.imwrite(
            "obstacle_mask_cleaned.png", (obstacle_mask_cleaned * 255).astype(np.uint8)
        )

        self.logger.debug("Saved individual mask images for debugging.")

    def find_safe_x(
        self,
        depth_filtered: np.ndarray,
        safe_x_screen: int,
        obstacle_mask_cleaned: np.ndarray,
    ) -> int:
        """
        安全なX座標を見つける。障害物マスクをスキャンして、安全な通路を探す。
        ここでは、障害物のない領域の中心を安全な経路とする。

        Args:
            depth_filtered (numpy.ndarray): ノイズ除去された深度画像
            safe_x_screen (int): 現在のSafe Xの画面座標
            obstacle_mask_cleaned (numpy.ndarray): フィルタリング後の障害物マスク

        Returns:
            int: 新しい安全なX座標（画面座標）
        """
        # 障害物マスクを縦方向に集約（各列に障害物が存在するか）
        obstacle_columns = np.any(obstacle_mask_cleaned, axis=0)

        # 障害物のない列を探す
        safe_columns = np.where(~obstacle_columns)[0]

        if len(safe_columns) == 0:
            self.logger.debug("No safe columns found. Defaulting Safe X to center.")
            return self.frame_width // 2

        # 中心から最も近い安全な列を選択
        center = self.frame_width // 2
        safe_x = safe_columns[np.argmin(np.abs(safe_columns - center))]

        self.logger.debug(f"Found safe X at column: {safe_x}")
        return safe_x
