import os 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO 
import cv2 
import numpy as np
import os 
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline 
import ssl 
import requests
import torch 

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['HF_HUB_DISABLE_XET'] = "1"

class DepthAnythingNode(Node):
    
    def __init__(self):
        super().__init__("depth_anything_node")
        model_path = "/home/uik03945/ros2_wsl_ws/models/Depth-Anything-V2-Small-hf"

        print(f"Loading model from local path: {model_path}")

        self.image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True)


        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.model.to(self.device)
        self.model.eval()

        print("Model Loaded Successfully from Disk!")

        self.model_name = "Depth Anything V2 on carla image"
        self.bridge = CvBridge()
        self.subscription= self.create_subscription(Image, '/output/image_FSC330_1', self.listener_callback, 10)
        self.output_dir = f"saved_images/{self.model_name}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.get_logger().info("Depth Anything v2 is started")

    def listener_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        inputs = self.image_processor(images=frame, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode = "bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_min, depth_max = depth_map.min(), depth_map.max()
        d_max = depth_max
        d_min = depth_min
        if d_max > d_min:
            depth_norm = (depth_map - d_min) / (d_max - d_min) * 255.0
        else:
            depth_norm = depth_map * 0.0

        depth_norm = depth_norm.astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


        # results = self.model(frame, conf=0.1)

        # annotated_frame = results[0].plot(boxes=False,labels=False)
        # annotated_frame_with_results=results[0].plot()
        # black_bg = np.zeros_like(frame)
        # mask_only = results[0].plot(boxes = False, labels = False, img = black_bg)


        path1 = os.path.join(self.output_dir, f"cleaned_result_{self.model_name}.jpg")
        # path2 = os.path.join(self.output_dir, f"labelled_result_{self.model_name}.jpg")
        # path3 = os.path.join(self.output_dir, f"only_result_{self.model_name}.jpg")

        cv2.imwrite(path1, depth_color)

        

        self.get_logger().info(f"Image saved succesfully to {self.output_dir}")

        raise SystemExit

def main():
    rclpy.init()
    rclpy.spin(DepthAnythingNode())
    rclpy.shutdown()

if __name__ == "__main__":
    main()