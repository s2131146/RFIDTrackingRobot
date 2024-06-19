import pyrealsense2 as rs
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["LIBREALSENSE_API_VERSION"] = "2"

import cv2

# RealSenseカメラの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# カメラを開始
pipeline.start(config)


# 障害物の検出と避ける方向の決定の例（簡略化したもの）
def detect_and_avoid_obstacle(depth_image):
    # 深度情報から障害物を検出する処理（例: 距離が近いピクセルの座標を抽出）

    # ここではダミーの処理として、最も近い障害物のピクセル位置を模擬
    nearest_obstacle_position = np.unravel_index(
        np.argmin(depth_image), depth_image.shape
    )

    # ロボットの中心から最も近い障害物の位置に基づいて避ける方向を決定する（例: 簡単な方向決定）
    robot_center_x = depth_image.shape[1] // 2
    obstacle_x = nearest_obstacle_position[1]
    if obstacle_x < robot_center_x:
        print("Turn right to avoid obstacle")
    else:
        print("Turn left to avoid obstacle")


def draw_bounding_boxes(color_image, depth_image):
    min_distance_index = np.unravel_index(np.argmin(depth_image), depth_image.shape)

    x, y = min_distance_index[::-1]  # RealSenseの座標系での位置

    # 四角形の左上と右下の座標を計算する
    rect_x1 = int(x - 20)
    rect_y1 = int(y - 20)
    rect_x2 = int(x + 20)
    rect_y2 = int(y + 20)

    # 画像の端を超えないようにクリッピング
    rect_x1 = max(rect_x1, 0)
    rect_y1 = max(rect_y1, 0)
    rect_x2 = min(rect_x2, color_image.shape[1])
    rect_y2 = min(rect_y2, color_image.shape[0])

    # 四角を描画
    cv2.rectangle(color_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)

    return color_image


def visualize_depth(depth_image):
    # 深度データを0-255の範囲に正規化してカラーマップに変換
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    return depth_colormap


try:
    while True:
        try:
            # フレームを待機
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if depth_frame is None or color_frame is None:
                print("Failed to retrieve depth frame or color frame")
                continue

            # 深度データをnumpy配列に変換
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 障害物を四角で囲んで可視化
            annotated_image = draw_bounding_boxes(color_image, depth_image)

            detect_and_avoid_obstacle(depth_image)

            # カラー画像をウィンドウに表示
            cv2.imshow('Annotated Image', annotated_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

finally:
    pipeline.stop()
