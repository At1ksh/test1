import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import onnxruntime as ort
import cv2
import numpy as np

class PureOnnxNode(Node):
    def __init__(self):
        super().__init__('pure_onnx_node')
        
        # 1. Initialize ONNX Runtime with TensorRT (for Orin speed)
        # It will fall back to CPU if TensorRT isn't found
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession("fcn-resnet18-mhp-512x320.onnx", providers=providers)
        
        self.subscription = self.create_subscription(Image, '/output/image_FSC330_1', self.callback, 10)
        self.bridge = CvBridge()
        self.get_logger().info("Pure ONNX Node Started")

    def callback(self, msg):
        # 2. Pre-processing (Manually resizing/normalizing)
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        input_img = cv2.resize(cv_img, (512, 320)) # Match your model's expected size
        input_img = input_img.transpose(2, 0, 1)   # HWC to CHW
        input_img = input_img.astype(np.float32)
        input_img = np.expand_dims(input_img, axis=0)

        # 3. Run Inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_img})

        # 4. Post-processing (The 'Mask')
        # This part is manual now since jetson-inference isn't doing it for you
        mask = np.argmax(outputs[0][0], axis=0)
        
        # Show result
        mask_viz = (mask * (255 // mask.max())).astype(np.uint8)
        cv2.imshow("Segmentation Mask", mask_viz)
        cv2.waitKey(1)

def main():
    rclpy.init()
    rclpy.spin(PureOnnxNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
