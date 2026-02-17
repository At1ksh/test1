 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO 
import cv2 

class YOLOTestNode(Node):
    def __init__(self):
        super().__init__("yolo_test_node")
        self.model = YOLO("yolo26s-seg.onnx")
        self.bridge = CvBridge()
        self.subscription= self.create_subscription(Image, '/output/image_FSC330_1', self.listener_callback, 10)

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame, conf=0.1,save=True)

        annotated_frame = results[0].plot(boxes=False,labels=False)
        cv2.imshow("Laptop YOLO Test", annotated_frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    rclpy.spin(YOLOTestNode())
    rclpy.shutdown()

if __name__ == "__main__":
    main()