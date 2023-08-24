import cv2
import os

# Initialize the camera
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

save_path = "saved_images"
os.makedirs(save_path, exist_ok=True)
image_count = 1


# Read a frame from the camera
while True:
    ret, frame = camera.read()


    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        exit()

    # Calculate cropping dimensions
    height, width, _ = frame.shape
    crop_width = int(width * 0.01)

    # Crop the frame
    cropped_frame = frame[:, crop_width:-crop_width]

    # Convert the cropped frame to grayscale
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)


    cv2.imshow("Img",gray_frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to exit the loop
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the grayscale image
        while os.path.exists(os.path.join(save_path, f"epuck2_image{image_count}.jpg")):
            image_count += 1
        image_filename = os.path.join(save_path, f"black_and_white_image{image_count}.jpg")

        cv2.imwrite(image_filename, gray_frame)
        print(f"Image saved as '{image_filename}'")
        image_count += 1


# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()