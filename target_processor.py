import math

import cv2
from constants import Commands


class TargetProcessor:
    """対象検出クラス: 人物の検出と距離に基づく速度調整を行う"""

    def __init__(self, frame_width, frame_height, model, logger, tracker):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model = model
        self.logger = logger
        self.tracker = tracker

        self.prev_command = Commands.STOP_TEMP
        self.lost_target_command = Commands.STOP_TEMP

    def update_speed_based_on_distance(self):
        """対象の占有率に基づいてself.default_speedを滑らかに更新"""

        # 定義された占有率と速度のポイント
        # 占有率: 0% -> 400, 40% -> 350, 60% -> 150
        ratios = [
            0.0,
            self.tracker.CLOSE_OCCUPANCY_RATIO,
            self.tracker.AUTO_STOP_OCCUPANCY_RATIO,
        ]
        speeds = [400, 350, 150]

        occupancy = self.tracker.occupancy_ratio

        if occupancy <= ratios[0]:
            # 占有率が0%以下の場合（理論的には0%）
            default_speed = speeds[0]
        elif ratios[0] < occupancy < ratios[1]:
            # 0% < 占有率 < 40% の範囲で線形補間
            # 比例定数を計算
            slope = (speeds[1] - speeds[0]) / (ratios[1] - ratios[0])
            default_speed = speeds[0] + slope * (occupancy - ratios[0])
        elif ratios[1] <= occupancy < ratios[2]:
            # 40% <= 占有率 < 60% の範囲で線形補間
            slope = (speeds[2] - speeds[1]) / (ratios[2] - ratios[1])
            default_speed = speeds[1] + slope * (occupancy - ratios[1])
        else:
            # 占有率が60%以上の場合
            default_speed = speeds[2]

        # スピードを整数値に丸める（必要に応じて）
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

    def process_target(self, targets, frame):
        """
        画像中の対象を囲み、中心座標と占有率を取得

        Args:
            targets (List[Tuple[int, int, int, int, float]]): 検出された対象のリスト (x1, y1, x2, y2, confidence)
            frame: MatLike

        Returns:
            Tuple[Optional[int], Optional[int], frame]: (target_center_x, target_x, frame)
        """
        for x1, y1, x2, y2, confidence in targets:
            target_center_x = x1 + (x2 - x1) // 2
            target_x = math.floor(target_center_x - self.frame_width / 2)

            # バウンディングボックスの面積を計算
            bbox_area = (x2 - x1) * (y2 - y1)
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text_org = (x1, y1 - 10)
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

            return target_center_x, target_x

        return None

    def detect_targets(self, frame, rfid_only_mode=False, rfid_enabled=False):
        """対象を検出するメソッド

        Args:
            frame (numpy.ndarray): フレーム
            rfid_only_mode (bool): RFIDのみのモードか
            rfid_enabled (bool): RFIDが有効か

        Returns:
            List[Tuple[int, int, int, int, float]]: 検出された対象のリスト
        """
        if rfid_only_mode and rfid_enabled:
            return []

        detected_targets = []
        results = self.model.predict(source=frame, conf=0.5, verbose=False)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                confidence = box.conf
                if cls == 0 and confidence > 0.5:  # クラス0が人物の場合
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    detected_targets.append((x1, y1, x2, y2, confidence))
        return detected_targets
