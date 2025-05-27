import cv2
import numpy as np
import requests
import time

# This script is used to capture images from the left and right cameras
# and compute the depth map. It is used to get the depth of the object
# in front of the robot.

# This script is currently not used in the control loop, but it is a good
# starting point for a depth perception system.

# --- Configuration ---
LEFT_CAM_URL = 'http://192.168.1.207/cam.jpg'
RIGHT_CAM_URL = 'http://192.168.1.206/cam.jpg'
CAMERA_BASELINE = 4.5 # cm
FOCAL_LENGTH_PIXELS = 225  # to be calibrated properly
FRAME_DELAY = 0.01  # seconds between frames

# --- Helper Functions ---
def get_camera_image(url):
    resp = requests.get(url, stream=True)
    img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img = cv2.flip(img, 1)  # horizontal flip (mirror)
    img = cv2.rotate(img, cv2.ROTATE_180)  # vertical flip (rotate 180)
    img = cv2.bilateralFilter(img, 7, 50, 50)  # smooth image to reduce noise
    return img

def compute_depth_map(left_img, right_img):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,  # must be divisible by 16
        blockSize=7,
        P1=8 * 1 * 7**2,
        P2=32 * 1 * 7**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    print(len(disparity[0]))
    return disparity

def disparity_to_depth(disparity):
    with np.errstate(divide='ignore'):
        depth_map = (FOCAL_LENGTH_PIXELS * CAMERA_BASELINE) / (disparity + 1e-6)
    return depth_map

def get_object_depth(depth_map, bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    depth = depth_map[center_y, center_x]
    return center_x, center_y, depth

# --- Main Loop ---
if __name__ == '__main__':
    while True:
        try:
            left = get_camera_image(LEFT_CAM_URL)
            right = get_camera_image(RIGHT_CAM_URL)

            disparity = compute_depth_map(left, right)
            depth_map = disparity_to_depth(disparity)

            dummy_bbox = (100, 100, 140, 140)
            u, v, depth = get_object_depth(depth_map, dummy_bbox)

            cx = left.shape[1] / 2
            cy = left.shape[0] / 2

            x = (u - cx) * depth / FOCAL_LENGTH_PIXELS
            y = (v - cy) * depth / FOCAL_LENGTH_PIXELS
            z = depth

            print(f"3D Position: x={x:.2f} cm, y={y:.2f} cm, z={z:.2f} cm")

            disp_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(disparity, alpha=2),
                cv2.COLORMAP_JET
            )

            cv2.imshow("Left", left)
            cv2.imshow("Right", right)
            cv2.imshow("Disparity", disp_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(FRAME_DELAY)

        except Exception as e:
            print("Error:", e)
            time.sleep(1)

    cv2.destroyAllWindows()
