import time
from typing import ClassVar, Counter, Optional, Tuple
import concurrent.futures
import cv2
import numpy as np
import logging


class Obstacles:
    """
    深度画像を処理し、フレーム内の障害物、床、および壁を検出するクラス。
    ロボットの視界に対する障害物と壁の位置を判断するメソッドを提供します。

    クラス変数:
        OBS_POS_CENTER (ClassVar[str]): 中央に障害物があることを示す定数。
        OBS_POS_LEFT (ClassVar[str]): 左側に障害物があることを示す定数。
        OBS_POS_RIGHT (ClassVar[str]): 右側に障害物があることを示す定数。
        OBS_POS_NONE (ClassVar[str]): 障害物がないことを示す定数。
        OBS_POS_FULL (ClassVar[str]): 目の前に全面的な障害物があることを示す定数。
    """

    # 障害物の位置を示す定数を定義
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
        tolerance: float = 100.0,
        min_continuous_increase: int = 25,
        wall_distance_threshold: float = 5000.0,
    ):
        """
        Obstaclesクラスを初期化します。

        引数:
            frame_width (int): フレームの幅（ピクセル単位）。
            frame_height (int): フレームの高さ（ピクセル単位）。
            logger (logging.Logger): ログを記録するためのロガーオブジェクト。
            obstacle_distance_threshold (float): 障害物とみなす深度の閾値（ミリメートル単位）。デフォルトは500.0。
            min_obstacle_size (Tuple[int, int]): 障害物とみなす最小サイズ（幅、高さ）。デフォルトは(50, 30)。
            tolerance (float): 深度の増加を判断する際の許容誤差。デフォルトは30.0。
            min_continuous_increase (int): 床や壁を検出する際の連続した深度増加の最小数。デフォルトは25。
            wall_distance_threshold (float): 壁とみなす深度の閾値（ミリメートル単位）。デフォルトは1000.0。
        """
        # フレームの幅と高さを設定
        self.frame_width = frame_width
        self.frame_height = frame_height
        # ロガーを設定
        self.logger = logger
        # 障害物とみなす深度の閾値を設定
        self.obstacle_distance_threshold = obstacle_distance_threshold
        # 深度の許容誤差を設定
        self.tolerance = tolerance
        # 連続した深度増加の最小数を設定
        self.min_continuous_increase = min_continuous_increase
        # 障害物検出のインターバルを設定（秒単位）
        self.detection_interval = 0.5  # 0.5秒のインターバル
        # 障害物の検出方向の履歴を保存するリスト
        self.directions_history = []
        # 壁の位置の履歴を保存するリスト
        self.wall_positions_history = []
        # 壁回避の履歴を保存するリスト
        self.avoid_walls_history = []
        # 前回の結果集計時刻を記録
        self.last_detection_time = time.time()
        # 最後に検出された障害物の方向
        self.last_direction = self.OBS_POS_NONE
        # 壁とみなす深度の閾値を設定
        self.wall_distance_threshold = wall_distance_threshold
        # 最後に検出された壁の位置
        self.last_wall_position = self.OBS_POS_NONE
        # 最後に壁を回避したかどうか
        self.last_wall_avoid = False

        # フレームの中央20%の領域を計算
        self.central_start = int(self.frame_width * 0.4)
        self.central_end = int(self.frame_width * 0.6)

        # 障害物とみなす最小サイズを設定
        self.min_obstacle_size = min_obstacle_size

        np.set_printoptions(threshold=np.inf)

    def process_obstacles(
        self,
        depth_image: np.ndarray,
        target_x: int,
        target_bbox=None,
    ) -> Tuple[str, bool, Optional[np.ndarray], str, bool]:
        """
        深度画像を処理して、障害物の位置、壁の位置、回避すべきかを判断します。

        引数:
            depth_image (np.ndarray): 深度画像（2次元配列）。
            target_x (int): ターゲットのx座標（ピクセル単位）。
            target_bbox (Optional): ターゲットのバウンディングボックス情報。

        戻り値:
            Tuple[str, bool, Optional[np.ndarray], str, bool]:
                - 障害物の位置（LEFT, RIGHT, CENTER, NONE, FULL）。
                - 障害物が存在するかのフラグ（True/False）。
                - 障害物の可視化画像（np.ndarray）。
                - 壁の位置（LEFT, RIGHT, CENTER, NONE）。
                - 壁を回避すべきかのフラグ（True/False）。
        """
        # 深度画像が存在しない場合、警告を出してデフォルト値を返す
        if depth_image is None:
            self.logger.warning("Depth image is None.")
            return self.OBS_POS_NONE, False, None, (100, 100), False

        # ターゲットのバウンディングボックスから無視するx範囲を取得
        ignore_x_start = ignore_x_end = None
        if target_bbox:
            x, _, w, _ = target_bbox[0].bboxes
            ignore_x_start = x
            ignore_x_end = w

        # ターゲットのx座標をフレームの中央に合わせる
        target_x = target_x + (self.frame_width // 2)
        # 深度画像の左右の余白を除去
        depth_image = depth_image[:, 25:]

        # 障害物が目の前にあるかを判断
        obstacle_mask_full = depth_image < self.obstacle_distance_threshold
        overall_direction = self.determine_obstacle_position_full(obstacle_mask_full)

        # 床パターンを検出（中央領域のみ）
        floor_mask = self.detect_floor(depth_image)

        # 壁を検出
        wall_mask = self.detect_walls(depth_image, floor_mask)
        wall_position = self.determine_wall_position(depth_image, wall_mask)

        # 中央領域に壁が存在するかをチェック
        central_area = wall_mask[:, self.central_start : self.central_end]
        wall_in_center = np.any(central_area)

        # 壁の位置と回避情報を履歴に追加
        self.wall_positions_history.append(wall_position)
        self.avoid_walls_history.append(wall_in_center)

        # 中央領域のマスクを作成し、床部分を除外
        central_region = depth_image[:, self.central_start : self.central_end]
        central_mask = np.ones_like(central_region, dtype=bool)
        central_mask[floor_mask[:, self.central_start : self.central_end]] = False

        # メディアンフィルタでノイズを除去
        depth_filtered = cv2.medianBlur(central_region.astype(np.uint16), 5).astype(
            np.float32
        )

        # 障害物マスクを初期化
        obstacle_mask = np.zeros_like(depth_filtered, dtype=bool)

        # 小さな障害物を除去
        obstacle_mask_cleaned = self.remove_small_obstacles(obstacle_mask)

        # 膨張処理で隙間を埋める
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

        # 検出された方向を履歴に追加
        self.directions_history.append(direction)

        # 0.5秒間隔で結果を更新
        current_time = time.time()
        most_common_direction = self.last_direction
        most_common_wall_position = self.last_wall_position
        most_common_avoid_wall = self.last_wall_avoid
        if current_time - self.last_detection_time >= self.detection_interval:
            # 最も頻出する障害物の方向を取得
            most_common_direction = Counter(self.directions_history).most_common(1)[0][
                0
            ]
            self.last_direction = most_common_direction

            # 最も頻出する壁の位置を取得
            most_common_wall_position = Counter(
                self.wall_positions_history
            ).most_common(1)[0][0]
            self.last_wall_position = most_common_wall_position

            # 壁を回避すべきかどうかを判断
            most_common_avoid_wall = Counter(self.avoid_walls_history).most_common(1)[
                0
            ][0]
            self.last_wall_avoid = most_common_avoid_wall

            # 履歴をリセットして次の集計に備える
            self.directions_history.clear()
            self.wall_positions_history.clear()
            self.avoid_walls_history.clear()
            self.last_detection_time = current_time

        # 障害物と床を可視化（障害物は赤色、床は青色で塗りつぶし）
        obstacle_visual = self.visualize_obstacle(
            depth_image,
            obstacle_mask_rect,
            floor_mask,
            wall_mask,
            target_x,
            ignore_x_start,
            ignore_x_end,
        )

        # 目の前に全面的な障害物がある場合、FULLを返す
        if overall_direction:
            return self.OBS_POS_FULL, obstacle_visual

        # ターゲットが中央20%の領域内にある場合、障害物判定をスキップ
        if target_bbox:
            if ignore_x_start < self.central_end and ignore_x_end > self.central_start:
                return (
                    self.OBS_POS_NONE,
                    False,
                    obstacle_visual,
                    most_common_wall_position,
                    most_common_avoid_wall,
                )

        # 最終的な結果を返す
        return (
            most_common_direction,
            most_common_direction != self.OBS_POS_NONE,
            obstacle_visual,
            most_common_wall_position,
            most_common_avoid_wall,
        )

    def determine_obstacle_position_full(self, obstacle_mask: np.ndarray) -> bool:
        """
        障害物が目の前に全面的に存在するかを判定します。

        引数:
            obstacle_mask (np.ndarray): 全体の障害物マスク。

        戻り値:
            bool: 障害物が全面に存在する場合はTrue、そうでない場合はFalse。
        """
        # 障害物マスクのカバレッジ率を計算
        total_coverage = np.sum(obstacle_mask) / (
            obstacle_mask.shape[0] * obstacle_mask.shape[1]
        )

        # 50%以上を占めていればTrueを返す
        if total_coverage >= 0.5:
            return True

        return False

    def determine_wall_position(
        self, depth_image: np.ndarray, wall_mask: np.ndarray
    ) -> str:
        """
        壁の位置を `wall_mask` と `depth_image` に基づいて判定します。
        中央領域に壁がある場合、左右の位置に基づいて左右の壁位置を決定します。

        引数:
            depth_image (np.ndarray): 深度画像。
            wall_mask (np.ndarray): 壁と判断された部分のマスク。

        戻り値:
            str: 壁の位置（LEFT, RIGHT, CENTER, NONE）。
        """
        _, width = wall_mask.shape
        # 左右と中央の領域を分割
        left_area = wall_mask[:, : width // 2]  # 左半分
        right_area = wall_mask[:, width // 2 :]  # 右半分
        central_area = wall_mask[:, self.central_start : self.central_end]  # 中央領域

        # 中央領域に壁が存在する場合、左右の壁の分布に基づいて判定
        left_count = np.sum(central_area[:, : central_area.shape[1] // 2])
        right_count = np.sum(central_area[:, central_area.shape[1] // 2 :])

        if np.any(central_area):
            if left_count > right_count:
                return self.OBS_POS_LEFT
            elif right_count > left_count:
                return self.OBS_POS_RIGHT
            else:
                return self.OBS_POS_CENTER

        # 左右の壁のカバレッジ率を計算
        left_coverage = np.sum(left_area) / left_area.size
        right_coverage = np.sum(right_area) / right_area.size

        # 左右どちらかが50%以上を占めていれば、その方向を返す
        if left_coverage > 0.5:
            return self.OBS_POS_LEFT
        if right_coverage > 0.5:
            return self.OBS_POS_RIGHT

        # 中央に壁がない場合、深度に基づいて壁の位置を判定
        left_depth_values = depth_image[:, : width // 5][wall_mask[:, : width // 5]]
        right_depth_values = depth_image[:, -width // 5 :][wall_mask[:, -width // 5 :]]

        depth_threshold = 2000  # 2000mm（2m）以内を壁とみなす
        left_depth_values = left_depth_values[
            (left_depth_values > 0) & (left_depth_values < depth_threshold)
        ]
        right_depth_values = right_depth_values[
            (right_depth_values > 0) & (right_depth_values < depth_threshold)
        ]

        # 左右の最小深度を取得
        left_distance = (
            np.min(left_depth_values) if left_depth_values.size > 0 else float("inf")
        )
        right_distance = (
            np.min(right_depth_values) if right_depth_values.size > 0 else float("inf")
        )

        # 両方が無限大の場合、壁が検出されていないためNONEを返す
        if left_distance == float("inf") and right_distance == float("inf"):
            return self.OBS_POS_NONE

        # 一方が無限大の場合、もう一方の方向を返す
        if left_distance == float("inf"):
            return self.OBS_POS_RIGHT
        elif right_distance == float("inf"):
            return self.OBS_POS_LEFT

        # 距離がほぼ等しい場合はCENTERを返す
        if abs(left_distance - right_distance) <= 30:
            return self.OBS_POS_CENTER
        elif left_distance < right_distance:
            return self.OBS_POS_LEFT
        else:
            return self.OBS_POS_RIGHT

    def fit_rectangle(self, obstacle_mask: np.ndarray) -> np.ndarray:
        """
        障害物マスクに矩形をフィットさせ、矩形領域をマスクとして返します。

        引数:
            obstacle_mask (np.ndarray): 障害物マスク。

        戻り値:
            np.ndarray: 矩形にフィットした障害物マスク。
        """
        obstacle_mask_rect = np.zeros_like(obstacle_mask, dtype=bool)
        contours, _ = cv2.findContours(
            obstacle_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 各輪郭に対して矩形をフィット
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            obstacle_mask_rect[y : y + h, x : x + w] = True  # 矩形領域をマスク

        return obstacle_mask_rect

    def detect_floor(self, depth_image: np.ndarray) -> np.ndarray:
        """
        深度画像から床を検出し、床のマスクを生成します。

        引数:
            depth_image (np.ndarray): 深度画像。

        戻り値:
            np.ndarray: 床のマスク。
        """
        floor_mask = np.zeros_like(depth_image, dtype=bool)
        mid_height = self.frame_height // 2  # フレームの中央高さ

        # 中央領域の各列に対して床を検出
        for col in range(self.central_start, self.central_end):
            # 下から上に向かって深度値を取得
            depth_values = depth_image[
                self.frame_height - 1 : mid_height : -1, col
            ].astype(np.int32)
            depth_values = depth_values[(depth_values > 0) & (depth_values < 2000)]
            if len(depth_values) < self.min_continuous_increase:
                continue

            continuous_increase_count = 0
            # 深度の連続した増加をチェック
            for i in range(1, len(depth_values)):
                if 0 < depth_values[i] - depth_values[i - 1] < self.tolerance:
                    continuous_increase_count += 1
                    if continuous_increase_count >= self.min_continuous_increase:
                        # 床と判断し、マスクを更新
                        floor_mask[mid_height : self.frame_height, col] = True
                        break
                else:
                    continuous_increase_count = 0

        return floor_mask

    def detect_walls(
        self, depth_image: np.ndarray, floor_mask: np.ndarray
    ) -> np.ndarray:
        """
        深度画像と床マスクから壁を検出し、壁のマスクを生成します。

        引数:
            depth_image (np.ndarray): 深度画像。
            floor_mask (np.ndarray): 床のマスク。

        戻り値:
            np.ndarray: 壁のマスク。
        """
        wall_mask = np.zeros_like(depth_image, dtype=bool)
        _, width = depth_image.shape

        # 左側の壁検出を実行
        self._detect_wall_side(
            depth_image=depth_image,
            floor_mask=floor_mask,
            wall_mask=wall_mask,
            start_col=0,
            end_col=width // 2,
            step=1,
            side="left",
        )

        # 右側の壁検出を実行
        self._detect_wall_side(
            depth_image=depth_image,
            floor_mask=floor_mask,
            wall_mask=wall_mask,
            start_col=width - 1,
            end_col=width // 2 - 1,
            step=-1,
            side="right",
        )

        # 膨張処理と連結成分を使って矩形にフィット
        wall_mask_rect = self._dilate_and_fit_rectangle(wall_mask)

        return wall_mask_rect

    def _detect_wall_side(
        self, depth_image, floor_mask, wall_mask, start_col, end_col, step, side
    ):
        """
        特定の方向（左または右）での壁検出処理を行います。

        引数:
            depth_image (np.ndarray): 深度画像。
            floor_mask (np.ndarray): 床のマスク。
            wall_mask (np.ndarray): 壁のマスク（更新対象）。
            start_col (int): 開始列。
            end_col (int): 終了列。
            step (int): 列の増分（1または-1）。
            side (str): 壁検出の方向（"left" または "right"）。
        """
        # 壁として認識する深度の最大値と勾配閾値
        depth_threshold = 1000  # 壁を検出する最大距離（必要に応じて調整）
        gradient_threshold = 30  # 勾配閾値（小さな変化を無視）

        for col in range(start_col, end_col, step):
            # 各列の深度値を取得し、0や65535、床部分を除外
            depth_values = depth_image[:, col]
            depth_values = depth_values[
                (depth_values > 0)
                & (depth_values < 65535)
                & (depth_values <= depth_threshold)
                & (~floor_mask[:, col])
            ]

            # 有効な深度値が十分に取れているか確認
            if len(depth_values) < self.min_continuous_increase:
                continue

            avg_depth = np.mean(depth_values)
            # 深度の勾配を計算して急激な変化があれば壁と判断
            gradient = np.abs(np.diff(depth_values))
            valid_gradients = gradient[gradient < 0xEA60]
            if avg_depth < depth_threshold and np.any(
                valid_gradients > gradient_threshold
            ):
                wall_mask[:, col] = True

    def _dilate_and_fit_rectangle(
        self, wall_mask: np.ndarray, min_wall_width=30
    ) -> np.ndarray:
        """
        膨張処理と連結成分解析を用いて、壁の矩形領域をマスクとして返します。

        引数:
            wall_mask (np.ndarray): 壁のマスク。
            min_wall_width (int): 壁とみなす最小の幅。デフォルトは30。

        戻り値:
            np.ndarray: 矩形にフィットした壁のマスク。
        """
        # 膨張処理を行う
        kernel = np.ones((5, 5), np.uint8)
        wall_mask_dilated = cv2.dilate(
            wall_mask.astype(np.uint8), kernel, iterations=5
        ).astype(bool)

        # 連結成分解析でラベル付け
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            wall_mask_dilated.astype(np.uint8), connectivity=8
        )
        wall_mask_rect = np.zeros_like(wall_mask_dilated, dtype=bool)

        # 最小の壁幅を満たす領域のみをマスク
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area > 100 and w >= min_wall_width:
                wall_mask_rect[y : y + h, x : x + w] = True

        return wall_mask_rect

    def remove_small_obstacles(self, obstacle_mask: np.ndarray) -> np.ndarray:
        """
        小さな障害物を除去し、クリーンな障害物マスクを返します。

        引数:
            obstacle_mask (np.ndarray): 障害物マスク。

        戻り値:
            np.ndarray: 小さな障害物を除去した障害物マスク。
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            obstacle_mask.astype(np.uint8), connectivity=8
        )

        cleaned_mask = np.zeros_like(obstacle_mask, dtype=bool)
        # 各ラベルに対してサイズをチェック
        for label in range(1, num_labels):
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]

            # 最小サイズを満たす場合のみマスクを更新
            if (
                width >= self.min_obstacle_size[0]
                and height >= self.min_obstacle_size[1]
            ):
                cleaned_mask[labels == label] = True

        return cleaned_mask

    def apply_colormap_to_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """
        深度画像にカラーマップを適用して可視化用のRGB画像を生成します。

        引数:
            depth_image (np.ndarray): 深度画像。

        戻り値:
            np.ndarray: カラーマップが適用されたRGB画像。
        """
        # 深度画像を0-255の範囲に正規化
        depth_normalized = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # ガンマ補正を適用して手前の変化を強調
        depth_normalized = np.power(depth_normalized / 255.0, 0.3) * 255
        depth_normalized = depth_normalized.astype(np.uint8)

        # カラーマップ「JET」を適用
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
        """
        障害物、床、壁、およびターゲット位置を可視化した画像を生成します。

        引数:
            depth_image (np.ndarray): 深度画像。
            obstacle_mask (np.ndarray): 障害物マスク。
            floor_mask (np.ndarray): 床のマスク。
            wall_mask (np.ndarray): 壁のマスク。
            target_x (int): ターゲットのx座標。
            ignore_x_start (int): 無視する領域の開始x座標。
            ignore_x_end (int): 無視する領域の終了x座標。

        戻り値:
            np.ndarray: 可視化画像。
        """
        # 背景を削除して真っ黒にする
        depth_rgb = np.zeros(
            (depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8
        )

        # 床を青色で塗りつぶし (RGB形式で青)
        blue_overlay = np.zeros_like(depth_rgb, dtype=np.uint8)
        blue_overlay[floor_mask] = [0, 0, 255]  # RGB形式で青

        # 壁を赤色で塗りつぶし（手前が濃く、奥が薄く）
        red_overlay = np.zeros_like(depth_rgb, dtype=np.uint8)
        red_overlay[wall_mask] = [255, 0, 0]

        # 各オーバーレイをdepth_rgbに適用
        depth_rgb = cv2.addWeighted(depth_rgb, 1, blue_overlay, 0.5, 0)
        depth_rgb = cv2.addWeighted(depth_rgb, 1, red_overlay, 0.5, 0)

        # 中央領域と無視する領域をラインで表示 (RGB形式の色指定)
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
            depth_rgb, (target_x, 0), (target_x, self.frame_height), (0, 255, 0), 2
        )

        # ターゲット位置をテキストで表示
        cv2.putText(
            depth_rgb,
            f"TARGET: {target_x - (self.frame_width // 2)}",
            (target_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 可視化画像を縮小
        resized_visualization = cv2.resize(
            depth_rgb,
            (int(self.frame_width * 0.35), int(self.frame_height * 0.35)),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_visualization

    def determine_obstacle_position(self, obstacle_mask: np.ndarray) -> str:
        """
        中央20%のエリアをさらに左右に分割し、障害物の位置を判定します。
        横方向に65%以上の障害物があればCENTERとし、
        または両端に障害物が触れている場合もCENTERとします。

        引数:
            obstacle_mask (np.ndarray): 障害物マスク。

        戻り値:
            str: 障害物の位置（LEFT, RIGHT, CENTER, NONE）。
        """
        # 中央20%のエリアを取得
        central_area = obstacle_mask[:, self.central_start : self.central_end]

        # 境界部分を取得
        left_boundary = central_area[:, 0]  # 左端
        right_boundary = central_area[:, -1]  # 右端

        # 境界部分に障害物があるか判定
        left_boundary_exists = np.any(left_boundary)
        right_boundary_exists = np.any(right_boundary)

        # 横方向に障害物が存在する割合を計算
        horizontal_coverage = (
            np.sum(np.any(central_area, axis=0)) / central_area.shape[1]
        )

        # 横方向65%以上が障害物で埋まっている、または両端に障害物がある場合はCENTERと判定
        if horizontal_coverage >= 0.65 or (
            left_boundary_exists and right_boundary_exists
        ):
            return self.OBS_POS_CENTER

        # 左右に分割して障害物の存在を判定
        left_exists = np.any(central_area[:, : central_area.shape[1] // 2])
        right_exists = np.any(central_area[:, central_area.shape[1] // 2 :])

        # 左右の存在状況に基づいて結果を返す
        if right_exists:
            return self.OBS_POS_RIGHT
        elif left_exists:
            return self.OBS_POS_LEFT
        else:
            return self.OBS_POS_NONE
