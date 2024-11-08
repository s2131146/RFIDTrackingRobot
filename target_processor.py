import math
import cv2
from ultralytics import YOLO
from constants import Commands
import time
from typing import List, Tuple, Optional


class Target:
    """ターゲットオブジェクトのクラス"""

    def __init__(self, x1, y1, x2, y2, confidence, clothing_color_rgb):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.clothing_color_rgb = clothing_color_rgb  # 服の色をRGBで保持
        self.distance = 0  # 距離の初期値
        self.bboxes = [x1, y1, x2, y2]

    @property
    def center_x(self):
        return self.x1 + (self.x2 - self.x1) // 2

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class TargetProcessor:
    """対象検出クラス: 人物の検出と距離に基づく速度調整を行う"""

    model: YOLO

    from tracker import Tracker

    def __init__(
        self, frame_width, frame_height, model: YOLO, logger, tracker: Tracker
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = logger
        self.tracker = tracker
        self.model = model.to("cuda")

        self.prev_command = Commands.STOP_TEMP
        self.lost_target_command = Commands.STOP_TEMP

        self.last_target_features = None  # 直前のターゲットの特徴を保存
        self.current_target = None  # 現在追跡中のターゲット
        self.color_tolerance = 50  # 色の一致許容範囲（ユークリッド距離）

        # 前回のターゲットの中心座標を保持
        self.last_target_center_x = None
        self.last_target_center_y = None

    def update_speed_based_on_distance(self):
        """対象の占有率に基づいてself.default_speedを滑らかに更新"""

        # 定義された占有率と速度のポイント
        # 占有率: 0% -> 200, 40% -> 200, 60% -> 100
        ratios = [
            0.0,
            self.tracker.CLOSE_OCCUPANCY_RATIO,  # 例: 0.4
            self.tracker.AUTO_STOP_OCCUPANCY_RATIO,  # 例: 0.6
        ]
        speeds = [350, 250, 100]

        occupancy = self.tracker.occupancy_ratio

        if occupancy <= ratios[0]:
            # 占有率が0%以下の場合（理論的には0%）
            default_speed = speeds[0]
        elif ratios[0] < occupancy < ratios[1]:
            # 0% < 占有率 < 40% の範囲で線形補間
            slope = (speeds[1] - speeds[0]) / (ratios[1] - ratios[0])
            default_speed = speeds[0] + slope * (occupancy - ratios[0])
        elif ratios[1] <= occupancy < ratios[2]:
            # 40% <= 占有率 < 60% の範囲で線形補間
            slope = (speeds[2] - speeds[1]) / (ratios[2] - ratios[1])
            default_speed = speeds[1] + slope * (occupancy - ratios[1])
        else:
            # 占有率が60%以上の場合
            default_speed = speeds[2]

        # スピードを整数値に丸める
        default_speed = int(default_speed)

        return default_speed

    def get_target_pos_str(self, target_center_x):
        # X座標の中心を0に調整
        x_centered = target_center_x - (self.frame_width // 2)

        # 中央の判定幅を画面幅の6分の1に設定
        central_threshold = self.frame_width // 6

        if x_centered < -central_threshold:
            target_position = Commands.GO_LEFT
            self.tracker.lost_target_command = Commands.ROTATE_LEFT
        elif x_centered > central_threshold:
            target_position = Commands.GO_RIGHT
            self.tracker.lost_target_command = Commands.ROTATE_RIGHT
        else:
            target_position = Commands.GO_CENTER
            self.tracker.lost_target_command = Commands.STOP_TEMP

        return target_position

    def select_target(self, detected_targets: List[Target]):
        """検出されたターゲットから追跡すべきターゲットを選択します。

        優先度:
            1. 前フレームの中心座標Xとバウンディングボックスのサイズが近いもの
            2. 中心座標Xが近いもの（バウンディングボックスのサイズが大きく異なる場合）

        Args:
            detected_targets (List[Target]): 検出されたターゲットのリスト

        Returns:
            Target: 選択されたターゲット
        """
        if not detected_targets:
            self.logger.info("No targets detected.")
            self.current_target = None
            return None

        if self.last_target_center_x is None or self.last_target_center_y is None:
            # 追跡中のターゲットがない場合、最も大きな面積のターゲットを選択
            selected_target = self.select_closest_target(detected_targets)
        else:
            # 前回のターゲットのサイズと位置に基づきターゲットを選択
            targets_with_distances = []
            last_target_area = self.current_target.area if self.current_target else None

            for target in detected_targets:
                # 中心座標の距離とバウンディングボックスの面積の差を計算
                center_distance = abs(target.center_x - self.last_target_center_x)
                area_difference = (
                    abs(target.area - last_target_area) if last_target_area else 0
                )

                # 中心座標とサイズが近いターゲットが優先されるようにタプルで保持
                targets_with_distances.append(
                    (target, center_distance, area_difference)
                )

            # 中心座標とバウンディングボックスの大きさが近い順にソート
            # まず面積差が少ないもの、その後中心座標が近いものを優先
            selected_target = sorted(
                targets_with_distances, key=lambda x: (x[2], x[1])
            )[0][0]

        # 選択されたターゲットの情報を更新
        self.current_target = selected_target
        self.last_target_center_x = selected_target.center_x
        self.last_target_center_y = (selected_target.y1 + selected_target.y2) // 2

        return selected_target

    def select_closest_target(self, targets: List[Target]) -> Target:
        """最も近いターゲットを選択します。"""
        return max(targets, key=lambda t: t.area)  # 面積が大きいほど近いと仮定

    def color_distance(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]
    ) -> float:
        """2つのRGB色の間のユークリッド距離を計算します。"""
        return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

    def features_match(
        self, target: Target, features: dict, tolerance: float = 30.0
    ) -> bool:
        """
        ターゲットの特徴が指定された特徴と一致するかを判定します。
        RGB色の距離が許容範囲内かをチェックします。

        Args:
            target (Target): 比較対象のターゲット
            features (dict): 比較元の特徴
            tolerance (float): 色の距離の許容範囲

        Returns:
            bool: 一致する場合はTrue、そうでない場合はFalse
        """
        target_color = target.clothing_color_rgb
        reference_color = features.get("clothing_color_rgb")
        if reference_color is None:
            return False

        distance = self.color_distance(target_color, reference_color)
        return distance <= tolerance

    def select_target_by_features(
        self, targets: List[Target], features: dict, tolerance: float = 30.0
    ) -> Optional[Target]:
        """指定された特徴に最も一致するターゲットを選択します。"""
        matching_targets = [
            t for t in targets if self.features_match(t, features, tolerance)
        ]
        if matching_targets:
            return self.select_closest_target(matching_targets)
        return None

    def handle_single_target(self, target: Target, frame):
        """単一ターゲットを処理します。"""
        current_time = time.time()
        if self.tracker.target_detection_start_time is None:
            self.tracker.target_detection_start_time = current_time
        else:
            self.tracker.time_detected = (
                current_time - self.tracker.target_detection_start_time
            )

            if self.tracker.time_detected >= 0.3:
                self.tracker.target_last_seen_time = current_time
                self.tracker.stop_temp = False
                result = self.process_target([target], frame)
                if result is not None:
                    (target_center_x, target_x, target_bboxes) = result
                    self.tracker.target_position_str = self.get_target_pos_str(
                        target_center_x
                    )
                    # 選択されたターゲットの特徴を保存
                    self.last_target_features = {
                        "clothing_color_rgb": target.clothing_color_rgb
                        # 他の必要な特徴を追加
                    }

    def process_target(
        self, targets: List[Target], frame
    ) -> Optional[Tuple[int, int, List[Tuple[int, int, int, int]]]]:
        """
        画像中の対象を囲み、中心座標と占有率を取得

        Args:
            targets (List[Target]): 検出された対象のリスト
            frame: MatLike

        Returns:
            Tuple[Optional[int], Optional[int], List[Tuple[int, int, int, int]]]: (target_center_x, target_x, target_bboxes)
        """
        target_bboxes = []
        selected_target = None

        if len(targets) == 0:
            return None

        # 特徴に基づくターゲット選択
        if self.last_target_features:
            selected_target = self.select_target_by_features(
                targets, self.last_target_features
            )

        # 特徴に一致するターゲットが見つからなかった場合、最も近いターゲットを選択
        if not selected_target:
            selected_target = self.select_closest_target(targets)

        if selected_target:
            # 選択されたターゲットの処理
            target_center_x = selected_target.center_x
            target_x = math.floor(target_center_x - self.frame_width / 2)

            # バウンディングボックスの面積を計算
            bbox_area = selected_target.area
            frame_area = self.frame_width * self.frame_height
            self.tracker.occupancy_ratio = bbox_area / frame_area  # 占有率

            self.tracker.default_speed = self.update_speed_based_on_distance()

            if self.tracker.occupancy_ratio >= self.tracker.AUTO_STOP_OCCUPANCY_RATIO:
                if not self.tracker.auto_stop:
                    self.tracker.auto_stop = True
                    self.tracker.stop_motor()
            else:
                if self.tracker.auto_stop:
                    self.tracker.auto_stop = False
                    self.tracker.start_motor()

            # 対象を矩形で囲む
            cv2.rectangle(
                frame,
                (selected_target.x1, selected_target.y1),
                (selected_target.x2, selected_target.y2),
                (0, 255, 0),
                2,
            )

            text_org = (selected_target.x1, selected_target.y1 - 10)
            distance_text = f"{self.tracker.occupancy_ratio:.2%}"
            cv2.putText(
                frame,
                f"TARGET {distance_text}",
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            target_bboxes.append(
                (
                    selected_target.x1,
                    selected_target.y1,
                    selected_target.x2,
                    selected_target.y2,
                )
            )

            return target_center_x, target_x, target_bboxes

        return None

    def draw_all_targets(self, detected_targets, selected_target, frame):
        """
        対象を全て描画します。選択された対象は緑色、それ以外は青色で囲み、'HUMAN'と表示します。

        Args:
            detected_targets (List[Target]): 検出された対象のリスト
            selected_target (Target): 選択された対象
            frame: MatLike
        """
        for target in detected_targets:
            if target == selected_target:
                color = (0, 255, 0)  # 緑色 (BGR形式)
            else:
                color = (255, 0, 0)  # 青色 (BGR形式)

            # 対象を矩形で囲む
            cv2.rectangle(
                frame, (target.x1, target.y1), (target.x2, target.y2), color, 2
            )

            # テキストを描画
            text_org = (target.x1, target.y1 - 10)
            cv2.putText(
                frame,
                "PERSON",
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    def detect_targets(
        self, frame, rfid_only_mode=False, rfid_enabled=False
    ) -> List[Target]:
        """対象を検出するメソッド

        Args:
            frame (numpy.ndarray): フレーム
            rfid_only_mode (bool): RFIDのみのモードか
            rfid_enabled (bool): RFIDが有効か

        Returns:
            List[Target]: 検出された対象のリスト
        """
        if rfid_only_mode and rfid_enabled:
            return []

        detected_targets = []
        results = self.model.predict(
            source=frame, conf=0.5, verbose=False, device="cuda"
        )
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                confidence = box.conf
                if cls == 0 and confidence > 0.5:  # クラス0が人物の場合
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)

                    # 服の色などの特徴を抽出するロジックをここに追加
                    # 例として、バウンディングボックス内の平均色を服の色とする
                    bbox = frame[y1:y2, x1:x2]
                    if bbox.size == 0:
                        clothing_color_rgb = (0, 0, 0)  # 黒色
                    else:
                        average_color = cv2.mean(bbox)[:3]  # BGR
                        clothing_color_rgb = self.extract_color_features(average_color)

                    target = Target(x1, y1, x2, y2, confidence, clothing_color_rgb)
                    detected_targets.append(target)
        return detected_targets

    def extract_color_features(
        self, average_color_bgr: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        """平均色から服の色の特徴を抽出します。

        Args:
            average_color_bgr (Tuple[float, float, float]): 平均BGR色

        Returns:
            Tuple[int, int, int]: RGB色
        """
        # BGRからRGBに変換
        b, g, r = average_color_bgr
        return (int(r), int(g), int(b))
