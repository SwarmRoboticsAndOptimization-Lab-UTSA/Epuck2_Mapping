import cv2
import numpy as np
import tensorflow as tf

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="epuck_detector_model\model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame (resize and reshape)
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)

    #Change data type to UINT8
    input_data = input_data.astype(np.uint8)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Output shape:", output_data.shape)

    # Post-process the output (you'll need to adjust this based on your model and what you want to detect)
    # detection_boxes = output_data[0, :, :4]
    # detection_scores = output_data[0, :, 4]
    
    # # Draw bounding boxes (again, this may need to be adjusted based on your model)
    # for i, score in enumerate(detection_scores):
    #     if score > 0.5:  # Adjust confidence threshold
    #         box = detection_boxes[i]
    #         y1, x1, y2, x2 = (box * [frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]]).astype(int)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # #Display frame
    # cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
