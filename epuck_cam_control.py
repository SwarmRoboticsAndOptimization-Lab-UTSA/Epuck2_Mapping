import cv2
import numpy as np
import tensorflow as tf
from obj_det_utils.utils import *

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera

# # Load the TFLite model
options = ObjectDetectorOptions(
    num_threads=4,
    score_threshold=0.5,
)
detector = ObjectDetector(model_path='epuck_detector_model/model.tflite', options=options)


_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

def visualize(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
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

frame_count = 0  # To generate unique file names for saved frames

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # # Run object detection estimation using the model.
    detections = detector.detect(frame)
    # # Draw keypoints and edges on input image
    image_np = visualize(frame, detections)
    # #Display frame
    cv2.imshow('Object Detection', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p'):  # Save frame when 'p' is pressed
        file_name = f'saved_frame_{frame_count}.jpg'
        cv2.imwrite(file_name, frame)
        print(f"Saved {file_name}")
        frame_count += 1

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
