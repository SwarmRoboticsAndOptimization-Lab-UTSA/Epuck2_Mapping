import cv2
import threading
import numpy as np
from pupil_apriltags import Detector
from obj_det_utils.utils import *
from obj_det_utils.roboDict import *

class Camera:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Camera, cls).__new__(cls)
                cls._instance.cap = cv2.VideoCapture(0)
                cls._instance.frame = None
                cls._instance.lock = threading.Lock()
                cls._instance.processed_frame = None
                cls._instance.detected_tags = {}
                cls._instance.at_detector = Detector(
                    families="tagCustom48h12",
                    nthreads=1,
                    quad_decimate=2.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0,
                )
        return cls._instance

    def get_frame(self):
        with self.lock:
            return self.frame

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_processed_frame(self):
        with self.lock:
            return self.processed_frame

    def set_processed_frame(self, frame):
        with self.lock:
            self.processed_frame = frame

    def release(self):
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()

    def capture_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            self.set_frame(frame)
            self.process_frame(frame)

    def process_frame(self, frame):
        # Image processing goes here
        # For example, converting to grayscale
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.set_processed_frame(processed_frame)

    def display_frame(self):
        while True:
            frame = self.get_processed_frame()
            if frame is not None:
                cv2.imshow('Processed Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

    def get_detected_tags(self):
        with self.lock:
            return self.detected_tags

    def set_detected_tags(self, tags):
        with self.lock:
            self.detected_tags = tags

    def process_frame(self, frame):
        # Image processing goes here
        # For example, converting to grayscale
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.set_processed_frame(processed_frame)

        # AprilTag detection
        tags = self.at_detector.detect(processed_frame, estimate_tag_pose=False, camera_params=None, tag_size=None)
        for tag in tags:
            tag_id = tag.tag_id
            center = tag.center
            corners = tag.corners
            center = (int(center[0]), int(center[1]))
            corner_01 = (int(corners[0][0]), int(corners[0][1]))
            corner_02 = (int(corners[1][0]), int(corners[1][1]))
            mid = midpoint(corner_01,corner_02)
            

            cv2.line(processed_frame, (center[0], center[1]),(mid[0], mid[1]), (255, 255, 0), 2)
            display_locations = [[160,66],[253,66],[345,66],[438,66],[515,66],[345,148],[345,243],[345,327],[345,412]]

            for i in display_locations:
                    cv2.circle(processed_frame, i, 10, (0,255,255), -1) #Draw circle goal location

            # Draw detection in the image
            cv2.circle(processed_frame, center, 5, (0, 255, 0), -1)

        self.set_detected_tags(tags)

    def process_tags(self):
        desired_location = [[160,66],[253,66],[345,66],[438,66],[515,66],[345,148],[345,243],[345,327],[345,412]]
        taken_locations = {}
        shared_robot_dict = RobotDict()

        while True:
            detected_tags = Camera().get_detected_tags()
            for tag in detected_tags:
                tag_id = tag.tag_id
                center = tag.center
                corners = tag.corners
                center = (int(center[0]), int(center[1]))
                corner_01 = (int(corners[0][0]), int(corners[0][1]))
                corner_02 = (int(corners[1][0]), int(corners[1][1]))

                mid = midpoint(corner_01,corner_02)
                
                # Calculate heading and other necessary information here
                # This is an example and may need to be adjusted according to your specific calculations
                heading = calculate_heading(center,mid)

                distances = [calculate_distance(mid[0],mid[1],des_loc[0],des_loc[1]) for des_loc in desired_location]

                if distances and (tag_id not in taken_locations):
                        min_distance = distances.index(min(distances)) #Get the index of the smallest distance.
                        taken_locations[str(tag_id)] = desired_location[min_distance] #Use index of the smallest distance to update robot desired location
                        desired_location.pop(min_distance)

                # if str(tag_id) in taken_locations:
                dist = calculate_distance(mid[0],mid[1],taken_locations[str(tag_id)][0],taken_locations[str(tag_id)][1])
                desired_heading = calculate_heading(center,taken_locations[str(tag_id)])
                # Update robot_dic using the thread-safe method
                shared_robot_dict.set(str(tag_id), [heading, desired_heading, dist])
                # Optional: Add visualization code here if needed
