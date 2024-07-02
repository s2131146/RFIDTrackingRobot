import pyrealsense2 as rs
import numpy as np
import cv2

# RealSenseのパイプラインを作成
pipeline = rs.pipeline()
config = rs.config()

# カラーストリームと深度ストリームを有効にする
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline.start(config)

try:
    while True:
        # フレームを取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        # カラー画像と深度画像をnumpy配列に変換
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 関心領域（ROI）の設定：画像の上部3/4のみを使用
        roi_depth = depth_image[:int(depth_image.shape[0]*3/4), :]
        roi_color = color_image[:int(color_image.shape[0]*3/4), :]

        # 深度画像の平滑化
        roi_depth = cv2.GaussianBlur(roi_depth, (5, 5), 0)

        # 障害物検知のための閾値設定
        max_distance = 300  # 300mm (0.3m)
        min_distance = 100  # 100mm (0.1m)

        # 深度データに基づいて閾値処理
        mask = np.logical_and(roi_depth > min_distance, roi_depth < max_distance)
        obstacle_image = np.zeros_like(roi_depth)
        obstacle_image[mask] = 255

        # ノイズ除去のためのモルフォロジー処理
        kernel = np.ones((5, 5), np.uint8)
        obstacle_image = cv2.morphologyEx(obstacle_image, cv2.MORPH_CLOSE, kernel)
        obstacle_image = cv2.morphologyEx(obstacle_image, cv2.MORPH_OPEN, kernel)

        # 障害物の輪郭を検出
        contours, _ = cv2.findContours(obstacle_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭に沿って矩形を描画
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # 小さいノイズを無視するための閾値設定
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ウィンドウに表示
        cv2.imshow('Obstacle Detection', color_image)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # クリーンアップ
    pipeline.stop()
    cv2.destroyAllWindows()
