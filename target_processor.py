import math
import cv2
from ultralytics import YOLO
from constants import Commands, Position
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
    detected_targets: List[Target] = []

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        model: YOLO,
        logger,
        tracker,
        min_target_size: Tuple[int, int] = (30, 60),
    ):
        """
        対象検出クラスの初期化

        Args:
            frame_width (int): フレームの幅
            frame_height (int): フレームの高さ
            model (YOLO): YOLOモデル
            logger: ロガー
            tracker: Trackerオブジェクト
            min_target_size (Tuple[int, int]): 対象とみなす最小の幅と高さ (デフォルト: (30, 60))
        """
        from tracker import Tracker

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = logger
        self.tracker: Tracker = tracker
        self.model = model.to("cuda")
        self.min_target_width, self.min_target_height = (
            min_target_size  # 最小サイズを設定
        )
        self.prev_command = Commands.STOP_TEMP
        self.last_target_features = None
        self.current_target = None
        self.color_tolerance = 30
        self.last_target_center_x = None
        self.last_target_center_y = None
        self.target_clothing_color = None
        self.last_bbox = None
        self.detected_target_list = []  # 映っている全てのターゲットのリスト
        self.target_index = -1  # 現在選択中のターゲットのインデックス

    def select_target(self, detected_targets: List[Target]):
        """検出されたターゲットから選択"""
        self.detected_targets = detected_targets
        # self.update_detected_target_list()

        # 現在のターゲットがリストに含まれていなければリセット
        if self.current_target not in self.detected_target_list:
            self.reset_target()

        return self.current_target

    def update_detected_target_list(self):
        """現在のフレームのターゲットをリストに反映"""
        # 現在のフレームで検出されたターゲットをキーにした辞書
        current_detected_colors = {
            tuple(target.clothing_color_rgb): target for target in self.detected_targets
        }

        # リストから現在映っていないターゲットを削除
        self.detected_target_list: List[Target] = [
            target
            for target in self.detected_target_list
            if tuple(target.clothing_color_rgb) in current_detected_colors
        ]

        # 新しく検出されたターゲットをリストに追加
        for color, target in current_detected_colors.items():
            if all(
                self.color_distance(color, t.clothing_color_rgb) > self.color_tolerance
                for t in self.detected_target_list
            ):
                self.detected_target_list.append(target)

        # 現在のターゲットがリストにいない場合、リストの最初を選択
        if (
            self.current_target not in self.detected_target_list
            and self.detected_target_list
        ):
            self.target_index = 0
            self.current_target = self.detected_target_list[self.target_index]
            self.target_clothing_color = self.current_target.clothing_color_rgb

    def reset_target(self):
        """保持している色と異なる、最もカメラに近いターゲットを新たな対象に選び、保持色をリセットします。"""
        self.update_detected_target_list()  # リストを最新の状態に更新

        if not self.detected_target_list:
            self.current_target = None
            self.target_clothing_color = None
            self.tracker.gui.update_status("No targets to reset to.")
            return

        # インデックスを更新して次のターゲットを選択
        self.target_index = (self.target_index + 1) % len(self.detected_target_list)
        self.current_target = self.detected_target_list[self.target_index]
        self.target_clothing_color = self.current_target.clothing_color_rgb

        self.tracker.gui.update_status(
            f"Target reset to {self.target_index + 1} / {len(self.detected_target_list)}."
        )

    def update_speed_based_on_distance(self):
        """対象の占有率に基づいてself.default_speedを滑らかに更新"""
        ratios = [
            0.0,
            self.tracker.CLOSE_OCCUPANCY_RATIO,
            self.tracker.AUTO_STOP_OCCUPANCY_RATIO,
        ]
        speeds = [350, 250, 150]

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

    def get_target_pos_str(self, target_center_x, current_bbox):
        import tracker

        # 前回のバウンディングボックスを取得
        last_bbox = self.last_bbox

        # バウンディングボックスが存在する場合、位置を計算
        x_centered = target_center_x - (self.frame_width // 2)

        # 中央の判定幅を画面幅の6分の1に設定
        central_threshold = self.frame_width // 10
        target_position = Position.CENTER

        if x_centered < -central_threshold:
            target_position = Position.LEFT
            self.tracker.lost_target_command = (
                Commands.ROTATE_LEFT
                if not tracker.DEBUG_INVERT_MOTOR
                else Commands.ROTATE_RIGHT
            )
        elif x_centered > central_threshold:
            target_position = Position.RIGHT
            self.tracker.lost_target_command = (
                Commands.ROTATE_RIGHT
                if not tracker.DEBUG_INVERT_MOTOR
                else Commands.ROTATE_LEFT
            )
        else:
            # 画面中央のバウンディングボックスがどちらに消えたかで判断
            if last_bbox and current_bbox:
                last_x_min, _, last_x_max, _ = last_bbox
                crr_x_min, y_min, crr_x_max, y_max = current_bbox

                if y_max - y_min > self.frame_height * 0.8:
                    target_position = Position.CENTER
                    self.tracker.lost_target_command = Commands.STOP_TEMP
                else:
                    if last_x_min < crr_x_min:  # 右方向に消えた
                        target_position = Position.RIGHT
                        self.tracker.lost_target_command = (
                            Commands.ROTATE_RIGHT
                            if not tracker.DEBUG_INVERT_MOTOR
                            else Commands.ROTATE_LEFT
                        )
                    elif last_x_max > crr_x_max:  # 左方向に消えた
                        target_position = Position.LEFT
                        self.tracker.lost_target_command = (
                            Commands.ROTATE_LEFT
                            if not tracker.DEBUG_INVERT_MOTOR
                            else Commands.ROTATE_RIGHT
                        )
                    else:  # 中央で消えた場合
                        target_position = Position.CENTER

        self.last_bbox = current_bbox

        return target_position

    def select_target(self, detected_targets: List[Target], no_target: bool = False):
        """検出されたターゲットから追跡すべきターゲットを選択します。

        優先度:
            最初は最も近いターゲット（バウンディングボックスのサイズが大きい）を選択
            以降はターゲットの色に基づいて最も一致するターゲットを選択

        Args:
            detected_targets (List[Target]): 検出されたターゲットのリスト

        Returns:
            Optional[Target]: 選択されたターゲット
        """
        if not detected_targets:
            self.logger.info("No targets detected.")
            self.current_target = None
            return None

        self.detected_targets = detected_targets

        # 最初の対象がまだ選ばれていない場合、最も近いターゲットを選択し、その色を記録
        if not self.target_clothing_color:
            selected_target = self.select_closest_target(detected_targets)
        else:
            # 既存の色に最も近いターゲットを選択 (対象が見つからない場合は条件を緩める)
            selected_target = self.select_target_by_color(
                detected_targets,
                self.target_clothing_color,
                self.color_tolerance,
            )

        # 光や映っている範囲で色が変わるため、逐一更新
        if selected_target:
            self.target_clothing_color = selected_target.clothing_color_rgb

        # 色に基づいて選択されたターゲットがいない場合、対象をなしに設定
        if selected_target:
            self.current_target = selected_target
        else:
            self.current_target = None
            if not no_target:
                return self.select_target(detected_targets, no_target=True)

        return self.current_target

    def select_target_by_color(
        self, targets: List[Target], target_color: Tuple[int, int, int], threshold: int
    ) -> Optional[Target]:
        """
        指定された色に最も近いターゲットを選択します。
        一致する色の中で、前回検出したターゲットの座標に最も近いターゲットを選択します。
        """
        # 色が一致するターゲットをフィルタリング
        matching_targets = [
            t
            for t in targets
            if self.color_distance(t.clothing_color_rgb, target_color) <= threshold
        ]

        # 一致するターゲットが存在しない場合
        if not matching_targets:
            return None

        # 前回のターゲットの座標が利用可能な場合、最も近いターゲットを選択
        if (
            self.last_target_center_x is not None
            and self.last_target_center_y is not None
        ):
            selected_target = min(
                matching_targets,
                key=lambda t: self.calculate_distance(
                    (t.center_x, t.y1 + (t.y2 - t.y1) // 2),
                    (self.last_target_center_x, self.last_target_center_y),
                ),
            )
        else:
            # 前回のターゲット情報がない場合は最も近いターゲットを選択
            selected_target = self.select_closest_target(matching_targets)

        # 選択されたターゲットの色情報を更新
        self.target_clothing_color = selected_target.clothing_color_rgb

        # 前回のターゲットの座標を更新
        self.last_target_center_x = selected_target.center_x
        self.last_target_center_y = (
            selected_target.y1 + (selected_target.y2 - selected_target.y1) // 2
        )

        return selected_target

    def calculate_distance(
        self, point1: Tuple[int, int], point2: Tuple[int, int]
    ) -> float:
        """
        2点間のユークリッド距離を計算します。

        Args:
            point1 (Tuple[int, int]): 座標1 (x, y)
            point2 (Tuple[int, int]): 座標2 (x, y)

        Returns:
            float: 2点間の距離
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

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

    def process_target(
        self, targets: List[Target], frame
    ) -> Optional[Tuple[int, int, List[Tuple[int, int, int, int]]]]:
        selected_target = None

        if len(targets) == 0:
            return None

        if self.last_target_features:
            selected_target = self.select_target_by_features(
                targets, self.last_target_features
            )

        if not selected_target:
            selected_target = self.select_closest_target(targets)

        if selected_target:
            target_center_x = selected_target.center_x
            target_x = math.floor(target_center_x - self.frame_width / 2)
            bbox_area = selected_target.area
            frame_area = self.frame_width * self.frame_height
            self.tracker.occupancy_ratio = bbox_area / frame_area
            self.tracker.default_speed = self.update_speed_based_on_distance()

            if self.tracker.occupancy_ratio >= self.tracker.AUTO_STOP_OCCUPANCY_RATIO:
                if not self.tracker.auto_stop and not self.tracker.stop:
                    self.tracker.auto_stop = True
                    self.tracker.stop_motor()
            else:
                if self.tracker.auto_stop and not self.tracker.stop_exec_cmd_gui:
                    self.tracker.auto_stop = False
                    self.tracker.start_motor()

            cv2.rectangle(
                frame,
                (selected_target.x1, selected_target.y1),
                (selected_target.x2, selected_target.y2),
                (0, 255, 0),
                2,
            )

            bbox_center = (
                (selected_target.x1 + selected_target.x2) // 2,
                (selected_target.y1 + selected_target.y2) // 2,
            )
            color_distance = self.color_distance(
                selected_target.clothing_color_rgb,
                self.target_clothing_color or (0, 0, 0),
            )
            text = f"TARGET {color_distance:.2f}"

            cv2.putText(
                frame,
                text,
                bbox_center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            target_bbox = (
                selected_target.x1,
                selected_target.y1,
                selected_target.x2,
                selected_target.y2,
            )

            return target_center_x, target_x, target_bbox

        return None

    def draw_all_targets(self, detected_targets: List[Target], selected_target, frame):
        detected_targets = [
            target for target in detected_targets if target != selected_target
        ]
        for target in detected_targets:
            if target == selected_target:
                return
            else:
                color = (255, 0, 0)  # 青色 (BGR形式)

            cv2.rectangle(
                frame, (target.x1, target.y1), (target.x2, target.y2), color, 2
            )

            bbox_center = (
                (target.x1 + target.x2) // 2,
                (target.y1 + target.y2) // 2,
            )
            color_distance = self.color_distance(
                target.clothing_color_rgb, self.target_clothing_color or (0, 0, 0)
            )
            text = f"PERSON {color_distance:.2f}"

            cv2.putText(
                frame,
                text,
                bbox_center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
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
            source=frame, conf=0.6, verbose=False, device="cuda"
        )
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                confidence = box.conf
                if cls == 0 and confidence > 0.5:  # クラス0が人物の場合
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)

                    # バウンディングボックスの幅と高さを計算
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # 最小サイズを満たさない場合はスキップ
                    if (
                        bbox_width < self.min_target_width
                        or bbox_height < self.min_target_height
                    ):
                        continue

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
