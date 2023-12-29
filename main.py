import logging
import cv2
import pyk4a
from pyk4a import PyK4APlayback
import configargparse
import os
import json

from utils import setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Output File Structure:
- output
    - recording_holisitcortake1
        - colorimage
            - camera{cam_id:02}_colorimage-{frame_id:06}.jpg
            - ...
        - depthimage_highres
            - camera{cam_id:02}_depthimage_highres-{frame_id:06}.tiff
            - ...
        - camera{cam_id:02}_eventlog.json
    - recording_holisitcortake2
        - ...
"""


def parse_args():
    parser = configargparse.ArgParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--root_dir', type=str, default='data', help='root directory for data')
    parser.add_argument('--take', type=int, default=1, help='take number')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--undistort', action='store_true', default=True, help='undistort images')
    parser.add_argument('--cam_ids', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6], help='list of camera ids')
    parser.add_argument('--skip_frames', type=int, default=0, help='number of frames to skip')

    logging.info("Parsing arguments...")
    return parser.parse_args()

class Exporter:
    def __init__(self, cfg):
        setup()
        self.cfg = cfg
        self.playbacks = []
        for cam_id in cfg.cam_ids:
            playback = PyK4APlayback(f'{cfg.root_dir}/capture_cn{cam_id:02}-20210906-holisticortake{cfg.take}.mkv')
            playback.open()
            self.playbacks.append(playback)

        self.event_logs = {cam_id: [] for cam_id in cfg.cam_ids}

        self.W = 2048
        self.H = 1536
        self._set_camera_params()
        self._create_output_dir()

        logging.info(f"Initializing exporter for take {cfg.take} with camera IDs {cfg.cam_ids}")

    def _create_output_dir(self):
        os.makedirs(f'{self.cfg.output_dir}/recording_holisticortake{self.cfg.take}/colorimage', exist_ok=True)
        os.makedirs(f'{self.cfg.output_dir}/recording_holisticortake{self.cfg.take}/depthimage_highres', exist_ok=True)
        logging.info(f"Creating output directories for take {self.cfg.take}")

    def synchronize_and_export(self):
        global_counter = 0
        file_name_counter = 0
        finished_playbacks = [False] * len(self.cfg.cam_ids)

        logging.info("Starting synchronization and export process...")

        while not all(finished_playbacks):
            logging.info(f'Processing frame {global_counter}...')
            captures = [playback.get_next_capture() for playback in self.playbacks]
            
            # Check if all captures have color and depth
            for i, capture in enumerate(captures):
                while capture.color is None or capture.depth is None:
                    try:
                        capture = self.playbacks[i].get_next_capture()
                    except EOFError:
                        finished_playbacks[i] = True
                        break
                captures[i] = capture

            if all(finished_playbacks):
                break

            timestamps = [capture.color_timestamp_usec if capture.color is not None and capture.depth is not None else 0 for capture in captures]
            max_timestamp = max(timestamps)

            # Synchronization
            for i in range(len(self.cfg.cam_ids)):
                while True:
                    try:
                        next_capture = self.playbacks[i].get_next_capture()
                        if next_capture.color is None or next_capture.depth is None:
                            break
                        
                        next_timestamp = next_capture.color_timestamp_usec
                        if abs(next_timestamp - max_timestamp) < abs(timestamps[i] - max_timestamp):
                            captures[i] = next_capture
                            timestamps[i] = next_timestamp
                        else:
                            break
                    except EOFError:
                        finished_playbacks[i] = True
                        break

            if global_counter % (self.cfg.skip_frames + 1) == 0:
                logging.info(f"Exporting frame {file_name_counter} for cameras {self.cfg.cam_ids}")
                for i, (cam_id, capture) in enumerate(zip(self.cfg.cam_ids, captures)):
                    color_image = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                    depth_image_highres = pyk4a.depth_image_to_color_camera(capture.depth, self.playbacks[i].calibration, thread_safe=False)

                    # Undistort images if needed
                    if self.cfg.undistort:
                        color_image, depth_image_highres = self.undistort(cam_id, color_image, depth_image_highres)

                    # Save color image
                    color_image_path = f'{self.cfg.output_dir}/recording_holisticortake{self.cfg.take}/colorimage/camera{cam_id:02}_colorimage-{file_name_counter:06}.jpg'
                    cv2.imwrite(color_image_path, color_image)

                    # Save depth image
                    depth_image_path = f'{self.cfg.output_dir}/recording_holisticortake{self.cfg.take}/depthimage_highres/camera{cam_id:02}_depthimage_highres-{file_name_counter:06}.tiff'
                    cv2.imwrite(depth_image_path, depth_image_highres)

                    # Update event logs
                    self.event_logs[cam_id].append({
                        "value0": {
                            "ts": capture.depth_timestamp_usec,
                            "ch": 0,
                            "fr": file_name_counter
                        }
                    })
                    self.event_logs[cam_id].append({
                        "value0": {
                            "ts": capture.color_timestamp_usec,
                            "ch": 1,
                            "fr": file_name_counter
                        }
                    })

                file_name_counter += 1
            global_counter += 1
        logging.info("Synchronization and export process completed.")
                       

        # Save event logs
        for cam_id, logs in self.event_logs.items():
            with open(f'{self.cfg.output_dir}/recording_holisticortake{self.cfg.take}/camera{cam_id:02}_eventlog.json', 'w') as f:
                json.dump(logs, f, indent=4)

    def _set_camera_params(self):
        self.camera_params = {cam_id: {"color": {}, "depth": {}} for cam_id in self.cfg.cam_ids}
        for cam_id, playback in zip(self.cfg.cam_ids, self.playbacks):
            calibration = playback.calibration
            for cam_type in ["color", "depth"]:
                intrinsic = calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR if cam_type == "color" else pyk4a.calibration.CalibrationType.DEPTH)
                distortion = calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR if cam_type == "color" else pyk4a.calibration.CalibrationType.DEPTH)
                newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (self.W, self.H), 0)
                map1, map2 = cv2.initUndistortRectifyMap(intrinsic, distortion, None, newCameraMatrix, (self.W, self.H), cv2.CV_32FC1)
                self.camera_params[cam_id][cam_type]["intrinsic"] = intrinsic
                self.camera_params[cam_id][cam_type]["distortion"] = distortion
                self.camera_params[cam_id][cam_type]["map1"] = map1
                self.camera_params[cam_id][cam_type]["map2"] = map2

    def undistort(self, cam_id, color_img, depth_img):
        color_cam_params = self.camera_params[cam_id]["color"]
        depth_cam_params = self.camera_params[cam_id]["depth"]
        undistorted_color_img = cv2.remap(color_img, color_cam_params["map1"], color_cam_params["map2"], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        undistorted_depth_img = cv2.remap(depth_img, depth_cam_params["map1"], depth_cam_params["map2"], cv2.INTER_NEAREST, cv2.BORDER_TRANSPARENT)
        return undistorted_color_img, undistorted_depth_img



if __name__ == '__main__':
    cfg = parse_args()
    exporter = Exporter(cfg)
    exporter.synchronize_and_export()