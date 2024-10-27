import cv2
import numpy as np
import math


class Obstacles:
    """障害物検出クラス: 障害物の検出と描画を行う"""

    def __init__(self, frame_width, frame_height, logger):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = logger

        # 深度範囲を調整（1.0mから2.5mに設定。必要に応じて調整）
        self.min_distance = 1000  # 1000mm = 1.0m
        self.max_distance = 2500  # 2500mm = 2.5m

        # バウンディングボックスフィルタリングの閾値
        self.max_box_height_ratio = 0.6  # フレーム高さの60%を超えるボックスは除外
        self.max_aspect_ratio = 3.0  # アスペクト比が3.0を超えるボックスは除外

    def preprocess_depth(self, roi_depth):
        """深度ROIを前処理する関数

        Args:
            roi_depth (numpy.ndarray): 深度フレームのROI

        Returns:
            numpy.ndarray: 前処理済みの深度フレーム
        """
        # ガウシアンブラーとメディアンブラーを適用してノイズを低減
        roi_depth_blurred = cv2.GaussianBlur(roi_depth, (5, 5), 0)
        roi_depth_filtered = cv2.medianBlur(roi_depth_blurred, 5)
        return roi_depth_filtered

    def create_obstacle_mask(self, roi_depth):
        """障害物マスクを作成する関数

        Args:
            roi_depth (numpy.ndarray): 前処理済みの深度フレームのROI

        Returns:
            numpy.ndarray: 障害物マスク
        """
        # 深度範囲内のピクセルをマスク
        mask = cv2.inRange(roi_depth, self.min_distance, self.max_distance)
        return mask

    def apply_morphology(self, obstacle_mask):
        """モルフォロジー処理を適用する関数

        Args:
            obstacle_mask (numpy.ndarray): 障害物マスク

        Returns:
            numpy.ndarray: 処理後の障害物マスク
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # クロージングとオープニングを適用してノイズを除去
        obstacle_image_processed = cv2.morphologyEx(
            obstacle_mask, cv2.MORPH_CLOSE, kernel
        )
        obstacle_image_processed = cv2.morphologyEx(
            obstacle_image_processed, cv2.MORPH_OPEN, kernel
        )
        return obstacle_image_processed

    def find_obstacle_contours(self, obstacle_image):
        """障害物の輪郭を検出する関数

        Args:
            obstacle_image (numpy.ndarray): 処理後の障害物マスク

        Returns:
            list: 検出された輪郭のリスト
        """
        contours, _ = cv2.findContours(
            obstacle_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def merge_bounding_boxes(self, boxes, distance_threshold=50):
        """近接するバウンディングボックスを統合する簡単な手法

        Args:
            boxes (List[List[int]]): バウンディングボックスのリスト
            distance_threshold (int): ボックス間の最大距離

        Returns:
            List[List[int]]: 統合後のバウンディングボックスのリスト
        """
        merged_boxes = []
        for box in boxes:
            if not merged_boxes:
                merged_boxes.append(box)
                continue

            merged = False
            for m_box in merged_boxes:
                # 中心点を計算
                m_center = ((m_box[0] + m_box[2]) / 2, (m_box[1] + m_box[3]) / 2)
                box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                distance = math.hypot(
                    m_center[0] - box_center[0], m_center[1] - box_center[1]
                )

                if distance < distance_threshold:
                    # ボックスを統合
                    m_box[0] = min(m_box[0], box[0])
                    m_box[1] = min(m_box[1], box[1])
                    m_box[2] = max(m_box[2], box[2])
                    m_box[3] = max(m_box[3], box[3])
                    merged = True
                    break

            if not merged:
                merged_boxes.append(box)

        return merged_boxes

    def process_contours(self, contours, roi_color, center_x):
        """輪郭を処理して情報を取得する関数

        Args:
            contours (list): 検出された輪郭のリスト
            roi_color (numpy.ndarray): カラーフレームのROI
            center_x (int): ROIの中心X座標

        Returns:
            tuple: (障害物がないか, 安全なX座標)
        """
        no_obstacle_detected = True
        detected_obstacles_info = []

        # 各輪郭のバウンディングボックスを取得
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:
                x, y, w, h = cv2.boundingRect(contour)

                # バウンディングボックスの高さ比とアスペクト比を計算
                height_ratio = h / self.frame_height
                aspect_ratio = w / h if h > 0 else 0

                # 高さ比とアスペクト比に基づいてフィルタリング
                if (
                    height_ratio < self.max_box_height_ratio
                    and aspect_ratio < self.max_aspect_ratio
                ):
                    bounding_boxes.append([x, y, x + w, y + h])
                else:
                    self.logger.debug(
                        f"Excluded bounding box due to size or aspect ratio: x={x}, y={y}, w={w}, h={h}, "
                        f"height_ratio={height_ratio:.2f}, aspect_ratio={aspect_ratio:.2f}"
                    )

        # バウンディングボックスを近いもの同士で統合
        bounding_boxes = self.merge_bounding_boxes(bounding_boxes)

        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            obstacle_center_x = x1 + w // 2
            distance_to_center = obstacle_center_x - center_x

            detected_obstacles_info.append((x1, y1, w, h, distance_to_center))
            no_obstacle_detected = False

            # 障害物を描画
            cv2.rectangle(roi_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text_org = (x1, y1 - 10)
            cv2.putText(
                roi_color,
                f"OBSTACLE X:{distance_to_center}",
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # 障害物のX座標を計算
        if detected_obstacles_info:
            all_obstacle_x_positions = [
                info[0] + info[2] // 2 for info in detected_obstacles_info
            ]
            min_obstacle_x = min(all_obstacle_x_positions)
            max_obstacle_x = max(all_obstacle_x_positions)
            obstacle_free_center_x = (min_obstacle_x + max_obstacle_x) // 2
        else:
            obstacle_free_center_x = center_x

        # フレームの中心を基準にX座標を調整
        obstacle_free_center_x = math.floor(
            obstacle_free_center_x - self.frame_width / 2
        )

        return no_obstacle_detected, obstacle_free_center_x

    def process_obstacles(self, frame, depth_image=None):
        """障害物を検知し、描画

        Args:
            frame (numpy.ndarray): フレーム
            depth_image (numpy.ndarray, optional): 深度フレーム

        Returns:
            tuple: (描画後フレーム, 障害物がないか, 安全なX座標)
        """
        if depth_image is not None:
            roi_depth_filtered = self.preprocess_depth(depth_image)
            obstacle_mask = self.create_obstacle_mask(roi_depth_filtered)
            obstacle_mask_processed = self.apply_morphology(obstacle_mask)
            contours = self.find_obstacle_contours(obstacle_mask_processed)
        else:
            # 深度画像がない場合はカラー画像のみで処理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, obstacle_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            obstacle_mask_processed = self.apply_morphology(obstacle_mask)
            contours = self.find_obstacle_contours(obstacle_mask_processed)

        center_x = frame.shape[1] // 2

        no_obstacle_detected, obstacle_free_center_x = self.process_contours(
            contours, frame, center_x
        )
        return frame, no_obstacle_detected, obstacle_free_center_x
