import tensorflow as tf
import numpy as np
import platform
from typing import List, NamedTuple

import json

import cv2
import math

Interpreter = tf.lite.Interpreter
load_delegate = tf.lite.experimental.load_delegate

# pylint: enable=g-import-not-at-top


class ObjectDetectorOptions(NamedTuple):
  """A config to initialize an object detector."""

  enable_edgetpu: bool = False
  """Enable the model to run on EdgeTPU."""

  label_allow_list: List[str] = None
  """The optional allow list of labels."""

  label_deny_list: List[str] = None
  """The optional deny list of labels."""

  max_results: int = -1
  """The maximum number of top-scored detection results to return."""

  num_threads: int = 1
  """The number of CPU threads to be used."""

  score_threshold: float = 0.0
  """The score threshold of detection results to return."""


class Rect(NamedTuple):
  """A rectangle in 2D space."""
  left: float
  top: float
  right: float
  bottom: float


class Category(NamedTuple):
  """A result of a classification task."""
  label: str
  score: float
  index: int


class Detection(NamedTuple):
  """A detected object as the result of an ObjectDetector."""
  bounding_box: Rect
  categories: List[Category]


def edgetpu_lib_name():
  """Returns the library name of EdgeTPU in the current platform."""
  return {
      'Darwin': 'libedgetpu.1.dylib',
      'Linux': 'libedgetpu.so.1',
      'Windows': 'edgetpu.dll',
  }.get(platform.system(), None)



class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output=None, min_output=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.min_output = min_output
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative

        if self.max_output is not None:
            output = min(output, self.max_output)
        if self.min_output is not None:
            output = max(output, self.min_output)

        self.prev_error = error

        return output

_MARGIN = 5  # pixels
_ROW_SIZE = 5  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

def visualize(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:                #Function to visualize objects detected might be removed
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detections: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detections:
    # Draw bounding_box
    left, top = int(detection.bounding_box.left), int(detection.bounding_box.top)
    right, bottom = int(detection.bounding_box.right), int(detection.bounding_box.bottom)

    point1 = (left, bottom)
    point2 = (right, bottom)
    point3 = ((left + right) // 2, top)
    
    pts = np.array([point1, point2, point3, point1], np.int32)
    pts = pts.reshape((-1, 1, 2))

    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    # start_point = detection.bounding_box.left, detection.bounding_box.top
    # end_point = detection.bounding_box.right, detection.bounding_box.bottom
    # cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    class_name = category.label
    probability = round(category.score, 2)
    result_text = class_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + detection.bounding_box.left,
                     _MARGIN + _ROW_SIZE + detection.bounding_box.top)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image

def midpoint(point1, point2): #Function to calculate the mid point between two points
    x1, y1 = point1
    x2, y2 = point2

    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2

    return (int(xm), int(ym))

# # Example usage:
# point1 = (1, 2)
# point2 = (3, 4)
# print(midpoint(point1, point2))  # Output: (2.0, 3.0)

def calculate_heading(point1, point2): #Function to calculate the heading based on two points
    x1, y1 = point1
    x2, y2 = point2

    theta = math.atan2(y2 - y1, x2 - x1)
    bearing = math.degrees(theta)
    
    # Convert from [-180, 180] to [0, 360]
    if bearing < 0:
        bearing += 360

    return bearing

# Example usage:
# point1 = (0, 0)
# point2 = (1, 1)
# print(calculate_heading(point1, point2))  # Should be roughly 45.0

def calculate_rotation_direction(current_heading, desired_heading, tolerance=20):
    # Calculate the angle difference
    angle_difference = desired_heading - current_heading

    # Normalize the angle difference to -180 to 180 degrees
    angle_difference = (angle_difference + 180) % 360 - 180

    # Determine the direction to rotate
    if abs(angle_difference) > tolerance:
        if angle_difference > 0:
          rotation_direction = "right"
        else:
          rotation_direction = "left"
    else:
        rotation_direction = "no rotation"  # Already at the desired angle

    return rotation_direction

#Example usage:
# current_heading = 30  # Current heading in degrees
# desired_heading = 150  # Desired heading in degrees
# rotation_direction = calculate_rotation_direction(current_heading, desired_heading)
# print(f"Rotate to the {rotation_direction} to reach the desired angle.")


def calculate_distance(x1, y1, x2, y2):
    # Calculate the squared differences in x and y coordinates
    x_diff_squared = (x2 - x1) ** 2
    y_diff_squared = (y2 - y1) ** 2
    
    # Calculate the sum of squared differences and then take the square root
    distance = math.sqrt(x_diff_squared + y_diff_squared)
    
    return distance

# Example usage:
# x1, y1 = 0, 0  # Coordinates of the first point
# x2, y2 = 3, 4  # Coordinates of the second point
# distance = calculate_distance(x1, y1, x2, y2)
# print(f"The distance between the two points is {distance:.2f} units.")



def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x}, {y})')  # prints the (x,y) coordinates on left button down