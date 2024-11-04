import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging


class Obstacles:
    """
    障害物検出クラス: 深度情報を使用して障害物と安全な通路を検出・可視化する。
    このクラスは、深度画像を処理して障害物を検出し、安全なX座標を計算します。
    また、検出結果を視覚化するためのRGB画像を生成します。
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        logger: logging.Logger,
        wall_exclusion_margin: int = 50,
        obstacle_distance_threshold: float = 500.0,  # ミリメートル（調整済み）
        min_safe_width: int = 50,
        min_obstacle_size: Tuple[int, int] = (70, 40),  # (縦, 横) ピクセル
        max_obstacle_size: Tuple[int, int] = (640, 480),  # (縦, 横) ピクセル
        consecutive_overlap_pixels: int = 20,
        depth_difference_threshold: float = 250.0,  # 平均深度 ±20cm
        obstacle_person_distance: float = 300.0,  # 人から25cm近いもの
    ):
        """
        クラスの初期化メソッド。

        Args:
            frame_width (int): フレームの幅（ピクセル）。
            frame_height (int): フレームの高さ（ピクセル）。
            logger (logging.Logger): ログ出力用のロガーオブジェクト。
            wall_exclusion_margin (int, optional): 壁近くを除外するマージン（ピクセル）。デフォルトは50。
            obstacle_distance_threshold (float, optional): 障害物とみなす深度の閾値（ミリメートル）。デフォルトは500.0。
            min_safe_width (int, optional): ロボットが通過できる最小の幅（ピクセル）。デフォルトは50。
            min_obstacle_size (Tuple[int, int], optional): 障害物と認識する最小の縦横サイズ（ピクセル）。デフォルトは(50, 40)。
            max_obstacle_size (Tuple[int, int], optional): 障害物と認識する最大の縦横サイズ（ピクセル）。デフォルトは(300, 300)。
            consecutive_overlap_pixels (int, optional): 重なりとみなす連続ピクセル数。デフォルトは20。
            depth_difference_threshold (float, optional): 深度差の閾値（ミリメートル）。デフォルトは200.0。
            obstacle_person_distance (float, optional): 人から近いとみなす深度差（ミリメートル）。デフォルトは200.0。
        """
        # フレームの幅と高さを設定
        self.frame_width = frame_width - 25
        self.frame_height = frame_height

        # ロガーを設定
        self.logger = logger

        # 障害物検出に使用する各種パラメータを設定
        self.obstacle_distance_threshold = obstacle_distance_threshold
        self.wall_exclusion_margin = wall_exclusion_margin
        self.min_safe_width = min_safe_width
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.consecutive_overlap_pixels = consecutive_overlap_pixels
        self.depth_difference_threshold = depth_difference_threshold
        self.obstacle_person_distance = obstacle_person_distance

        # 最後に検出されたターゲットのX座標を保持（初期値はNone）
        self.last_target_x: Optional[int] = 0
        self.last_safe_x = self.frame_width // 2

    def process_obstacles(
        self,
        depth_image: np.ndarray,
        target_bboxes: List[List[int]] = [],
        target_x: Optional[int] = -1,
    ) -> Tuple[int, bool, Optional[np.ndarray]]:
        """
        障害物を処理し、安全なX座標を計算する。

        Args:
            depth_image (numpy.ndarray): 深度画像（左右反転済み）。
            target_bboxes (List[List[int]], optional): 対象のバウンディングボックスリスト。各バウンディングボックスは [x1, y1, x2, y2]。
            target_x (int, optional): ターゲットのX座標（画面中心が0、左が負、右が正）。

        Returns:
            Tuple[int, bool, Optional[numpy.ndarray]]:
                - 安全なX座標（画面中心基準、左が負、右が正）。
                - 障害物の有無（True: 障害物なし, False: 障害物あり）。
                - 視覚化画像（RGB形式の画像。障害物と安全なX座標が表示されている）。
        """
        if depth_image is None:
            self.logger.warning("Depth image is None. Cannot process obstacles.")
            return 0, True, None

        if depth_image.ndim != 2:
            self.logger.error(f"Depth image must be 2D. Received shape: {depth_image.shape}")
            return 0, True, None

        # 深度画像を左右反転
        depth_image_flipped = cv2.flip(depth_image, 1)
        depth_image_flipped = depth_image_flipped[:, :-25]
        self.logger.debug(f"Depth image flipped. Shape: {depth_image_flipped.shape}")

        # メディアンフィルタを適用してノイズを除去
        depth_filtered = cv2.medianBlur(depth_image_flipped.astype(np.uint16), 5).astype(np.float32)
        self.logger.debug("Applied median filter to depth image.")

        # 障害物マスクを生成
        mask_height = int(self.frame_height * 0.1)
        obstacle_mask = depth_filtered < self.obstacle_distance_threshold
        obstacle_mask[-mask_height:, :] = False 
        self.logger.debug(f"Initial obstacle mask created. Total obstacles (pixels): {np.sum(obstacle_mask)}")

        # 連結成分分析を実施して障害物をラベリング
        obstacle_mask_cleaned, labels, stats = self.filter_obstacles(obstacle_mask)
        self.logger.debug(f"Obstacle mask after filtering small obstacles. Total obstacles (pixels): {np.sum(obstacle_mask_cleaned)}")

        # 画面中央15%範囲の障害物確認
        central_region_start = int(self.frame_width * 0.425)  # 画面の中央から7.5%左
        central_region_end = int(self.frame_width * 0.575)    # 画面の中央から7.5%右
        central_obstacle_present = np.any(obstacle_mask_cleaned[:, central_region_start:central_region_end])

        # 人のマスクと重なっている障害物のラベルを保持するリスト
        person_mask = None
        overlapping_labels = []

        if target_bboxes and target_x != -1:
            # ターゲット（人）の検出
            person_mask = self.detect_person_contour(depth_filtered, target_bboxes)
            if person_mask is not None:
                self.logger.debug("Person detected using depth contour.")
                overlapping_labels = self.find_overlapping_obstacles(depth_filtered, labels, person_mask)
                if overlapping_labels:
                    for label in overlapping_labels:
                        obstacle_mask_cleaned[labels == label] = True
                    self.logger.debug("Obstacles detected overlapping with person.")

        # 安全なX座標の初期設定
        if target_bboxes and target_x != -1:
            self.last_target_x = target_x
            safe_x = target_x
        else:
            # ターゲットが検出されていない場合、最後に検出されたターゲットのX座標を使用
            if self.last_target_x != -1:
                safe_x = self.last_target_x
            else:
                safe_x = 0  # デフォルトで画面中央

        # 安全なX座標を画面座標に変換
        safe_x_screen = safe_x + (self.frame_width // 2)
        safe_x_centered = safe_x
        self.logger.debug(f"Safe X in screen coordinates: {safe_x_screen}")

        if not (target_bboxes and target_x != -1 and not overlapping_labels):
            # 安全なX座標を再計算
            safe_x = self.find_safe_x(
                central_obstacle_present, depth_filtered, safe_x_screen, obstacle_mask_cleaned, target_x
            )
            self.logger.debug(f"Safe X recalculated: {safe_x}")

            # 安全なX座標を画面中心基準に調整
            safe_x_centered = safe_x - (self.frame_width // 2)
            self.logger.debug(f"Final Safe X (centered): {safe_x_centered}")
        else:
            safe_x = safe_x_screen

        # 深度画像と検出結果を視覚化したRGB画像を生成
        depth_frame = self.visualize_depth(
            depth_filtered,
            safe_x,
            labels,
            overlapping_labels if target_bboxes and target_x != -1 else [],
            person_mask if target_bboxes and target_x != -1 else None,
        )

        # obstacle_flagの設定：進行方向（画面中央15%）に障害物があるか、対象に重なっている障害物がある場合のみTrue
        obstacle_flag = central_obstacle_present or bool(overlapping_labels)
        self.logger.debug(f"Obstacle flag set to: {obstacle_flag}")

        # 安全なX座標、障害物の有無、視覚化画像を返す
        return safe_x_centered, not obstacle_flag, depth_frame


    def detect_person_contour(
        self, depth_filtered: np.ndarray, target_bboxes: List[List[int]]
    ) -> Optional[np.ndarray]:
        """
        深度情報を用いて、人の輪郭を検出し、マスクを返す。

        Args:
            depth_filtered (numpy.ndarray): フィルタリング済みの深度画像。
            target_bboxes (List[List[int]]): 対象のバウンディングボックスリスト。

        Returns:
            Optional[np.ndarray]: 人のマスク。人が検出されなかった場合はNone。
        """
        if not target_bboxes:
            # バウンディングボックスが提供されていない場合
            self.logger.debug("No target bounding boxes provided.")
            return None

        # 最初のバウンディングボックスを使用
        x1, y1, x2, y2 = target_bboxes[0]
        self.logger.debug(f"Using bounding box: {(x1, y1, x2, y2)}")

        # バウンディングボックス内の領域（ROI）を抽出
        person_roi = depth_filtered[y1:y2, x1:x2]
        if person_roi.size == 0:
            # ROIが空の場合
            self.logger.debug("Person ROI is empty.")
            return None

        # ROI内の平均深度を計算
        person_depth = np.nanmean(person_roi)
        self.logger.debug(f"Person average depth: {person_depth}")

        # 深度画像全体から、平均深度 ± depth_difference_threshold の範囲のピクセルをマスク
        depth_diff = np.abs(depth_filtered - person_depth)
        person_mask_initial = depth_diff <= self.depth_difference_threshold
        self.logger.debug(
            f"Initial person mask created with threshold {self.depth_difference_threshold}mm."
        )

        # 連結成分分析を実施してマスクをラベリング
        num_labels, labels = cv2.connectedComponents(
            person_mask_initial.astype(np.uint8), connectivity=8
        )
        self.logger.debug(f"Connected components in person mask: {num_labels}")

        # ラベルごとの平均深度を計算し、深度差が閾値内のラベルを人として認識
        person_labels = []
        for label in range(1, num_labels):
            # ラベルごとのマスクを作成
            label_mask = labels == label
            # ラベル内の平均深度を計算
            label_depth = np.nanmean(depth_filtered[label_mask])
            # ラベルの深度が人の深度と閾値内かを確認
            if np.abs(label_depth - person_depth) <= self.depth_difference_threshold:
                person_labels.append(label)
                self.logger.debug(
                    f"Label {label} with depth {label_depth}mm is within threshold."
                )

        if not person_labels:
            # 閾値内のラベルが存在しない場合
            self.logger.debug("No labels within depth threshold for person.")
            return None

        # ラベリングされた領域がつながっているもののみを人のマスクとして認識
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
        人と重なっている障害物のラベルを検出する。

        Args:
            depth_filtered (np.ndarray): フィルタリング済みの深度画像。
            labels (np.ndarray): ラベルマップ。
            person_mask (np.ndarray): 人のマスク。

        Returns:
            List[int]: 重なっている障害物のラベルリスト。
        """
        # 人の平均深度を再計算
        person_depth = np.nanmean(depth_filtered[person_mask])
        self.logger.debug(f"Person depth for overlapping detection: {person_depth}mm")

        overlapping_labels = []
        # 障害物ラベルのユニークなリストを取得（背景ラベル0は除外）
        unique_labels = np.unique(labels[labels > 0])

        for label in unique_labels:
            if label == 0:
                # ラベル0は背景なのでスキップ
                continue
            # 現在のラベルのマスクを作成
            obstacle_mask = labels == label
            # 障害物マスクと人マスクの重なりを確認
            overlap = np.logical_and(person_mask, obstacle_mask)
            if np.any(overlap):
                # 障害物の平均深度を計算
                obstacle_depth = np.nanmean(depth_filtered[obstacle_mask])
                # 人の深度と障害物の深度の差を計算
                depth_diff = person_depth - obstacle_depth
                if depth_diff >= self.obstacle_person_distance:
                    # 深度差が閾値以上の場合、重なっている障害物と判断
                    overlapping_labels.append(label)
                    self.logger.debug(
                        f"Obstacle label {label} overlaps with person and is {depth_diff}mm closer."
                    )

        return overlapping_labels

    def filter_obstacles(
        self, obstacle_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        連結成分分析を使用して、障害物マスクを生成する。

        Args:
            obstacle_mask (numpy.ndarray): 初期の障害物マスク。

        Returns:
            Tuple[numpy.ndarray, np.ndarray, np.ndarray]:
                - 障害物マスク（フィルタリングなし）。
                - ラベルマップ。
                - 各ラベルの統計情報（面積、バウンディングボックスなど）。
        """
        # 連結成分分析を実施してラベルを取得
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            obstacle_mask.astype(np.uint8), connectivity=8
        )
        self.logger.debug(f"Connected components found: {num_labels}")

        # すべての障害物をそのままマスクとして返す
        obstacle_mask_filtered = labels > 0  # ラベル0（背景）を除外

        return obstacle_mask_filtered, labels, stats


    def find_safe_x(
        self,
        obstacle_present,
        depth_filtered: np.ndarray,
        safe_x_screen: int,
        obstacle_mask_cleaned: np.ndarray,
        target_x: Optional[int],
        min_safe_width: int = 50
    ) -> int:
        self.logger.debug("Starting find_safe_x process.")

        # 障害物のマスクを逆転し、安全な列を示すマスクを作成
        safe_columns = ~np.any(obstacle_mask_cleaned, axis=0)
        self.logger.debug(f"Total safe columns identified: {np.sum(safe_columns)}")

        # 安全な列の範囲を検出
        safe_starts = []
        safe_ends = []
        in_safe_region = False

        for x in range(self.frame_width):
            if safe_columns[x] and not in_safe_region:
                safe_starts.append(x)
                in_safe_region = True
            elif not safe_columns[x] and in_safe_region:
                safe_ends.append(x - 1)
                in_safe_region = False

        if in_safe_region:
            safe_ends.append(self.frame_width - 1)

        # 20px未満の安全領域の連結
        merged_safe_starts = []
        merged_safe_ends = []
        i = 0
        while i < len(safe_starts):
            start = safe_starts[i]
            end = safe_ends[i]

            while i + 1 < len(safe_starts) and safe_starts[i + 1] - end <= 20:
                end = safe_ends[i + 1]
                i += 1

            merged_safe_starts.append(start)
            merged_safe_ends.append(end)
            i += 1

        # 最も広い安全領域を見つける
        max_width = 0
        best_safe_x = self.frame_width // 2  # デフォルトは画面中央
        for start, end in zip(merged_safe_starts, merged_safe_ends):
            width = end - start
            if width > max_width:
                max_width = width
                best_safe_x = (start + end) // 2  # 最も広い安全領域の中心

        # 前回の `safe_x` から30ピクセル以上の変動がある場合
        if abs(best_safe_x - self.last_safe_x) > 30:
            # 新しい領域が前回の安全領域の2倍未満である場合
            if max_width < 2 * self.min_safe_width:
                # 前回の `safe_x` の位置に存在する安全領域の中央を計算
                previous_region_start, previous_region_end = None, None
                for start, end in zip(merged_safe_starts, merged_safe_ends):
                    if start <= self.last_safe_x <= end:
                        previous_region_start = start
                        previous_region_end = end
                        break

                if previous_region_start is not None and previous_region_end is not None:
                    # 前回の `safe_x` の位置にある安全領域の中央を新しい `safe_x` に設定
                    self.last_safe_x = (previous_region_start + previous_region_end) // 2
                    self.logger.debug(f"Keeping safe X within previous safe region at X={self.last_safe_x}")
                else:
                    # 安全領域が見つからない場合はデフォルトとして最も広い領域を設定
                    self.last_safe_x = best_safe_x
                    self.logger.debug(f"No previous safe region found, setting safe X to widest region at X={best_safe_x}")
            else:
                # 幅が十分大きい場合、新しい領域の中心を `safe_x` として設定
                self.last_safe_x = best_safe_x
                self.logger.debug(f"Switching to a wider safe region at X={best_safe_x} (width: {max_width})")
        else:
            # 30px以内の変動ならそのまま設定
            self.logger.debug(f"Safe X within 30px change, setting to X={best_safe_x}")
            self.last_safe_x = best_safe_x

        return self.last_safe_x
    
    def visualize_depth(
        self,
        depth_filtered: np.ndarray,
        safe_x: int,
        labels: np.ndarray,
        overlapping_labels: List[int],
        person_mask: Optional[np.ndarray],
        central_safe_mode: bool = False  # 中央に向かうモードかどうかのフラグ
    ) -> np.ndarray:
        depth_normalized = cv2.normalize(
            depth_filtered,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        self.logger.debug("Depth image normalized.")

        depth_gray_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
        self.logger.debug("Depth image converted to RGB grayscale.")

        visualization_image = depth_gray_rgb.copy()

        # 障害物の描画
        obstacle_mask = labels > 0
        for label in overlapping_labels:
            obstacle_mask[labels == label] = False  # 重なっている障害物を除外
        if person_mask is not None:
            obstacle_mask = obstacle_mask & (~person_mask)  # 人のマスクを除外
        visualization_image[obstacle_mask] = [255, 0, 0]
        self.logger.debug("Other obstacles colored red.")

        # 重なっている障害物を黄色で表示
        for label in overlapping_labels:
            overlap_mask = labels == label
            visualization_image[overlap_mask] = [255, 255, 0]  # 黄色
            self.logger.debug(f"Overlapping obstacle Label {label} colored yellow.")

        # 人を青色で表示
        if person_mask is not None:
            visualization_image[person_mask] = [0, 0, 255]
            self.logger.debug("Person colored blue.")

        # 安全な列の判定と20px未満の領域の連結処理
        safe_columns = ~np.any(obstacle_mask | person_mask if person_mask is not None else obstacle_mask, axis=0)
        self.logger.debug(f"Safe columns identified: {np.sum(safe_columns)}")

        safe_starts = []
        safe_ends = []
        in_safe_region = False

        for x in range(self.frame_width):
            if safe_columns[x] and not in_safe_region:
                safe_starts.append(x)
                in_safe_region = True
            elif not safe_columns[x] and in_safe_region:
                safe_ends.append(x - 1)
                in_safe_region = False

        if in_safe_region:
            safe_ends.append(self.frame_width - 1)

        # 20px未満の安全領域の連結
        merged_safe_starts = []
        merged_safe_ends = []
        i = 0
        while i < len(safe_starts):
            start = safe_starts[i]
            end = safe_ends[i]

            while i + 1 < len(safe_starts) and safe_starts[i + 1] - end <= 20:
                end = safe_ends[i + 1]
                i += 1

            merged_safe_starts.append(start)
            merged_safe_ends.append(end)
            i += 1

        # 紫色（半透明）の安全領域を設定
        overlay = visualization_image.copy()
        for start, end in zip(merged_safe_starts, merged_safe_ends):
            overlay[:, start:end] = [128, 0, 128]  # 紫色
        alpha = 0.5  # 半透明のための透明度
        cv2.addWeighted(overlay, alpha, visualization_image, 1 - alpha, 0, visualization_image)
        self.logger.debug("Safe vertical regions colored purple with transparency.")

        # `safe_x` のラインとテキストの色を中央向きかどうかで切り替え
        line_color = (0, 165, 255) if central_safe_mode else (0, 255, 0)  # オレンジまたは緑
        text_color = line_color

        # 安全なX座標をオレンジ色または緑色の線で表示
        cv2.line(
            visualization_image,
            (safe_x, 0),
            (safe_x, self.frame_height),
            line_color,
            2,
        )
        # 安全なX座標の値をテキストで表示
        cv2.putText(
            visualization_image,
            f"Safe X: {safe_x - (self.frame_width // 2)}",
            (safe_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )
        self.logger.debug(f"Safe X line and text added at X={safe_x} with color {line_color}")

        resized_visualization_image = cv2.resize(
            visualization_image,
            (int(self.frame_width * 0.35), int(self.frame_height * 0.35)),
            interpolation=cv2.INTER_LINEAR,
        )
        self.logger.debug(
            f"Visualization image resized. New shape: {resized_visualization_image.shape}"
        )

        return resized_visualization_image
