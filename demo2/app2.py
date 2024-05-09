import os

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, session, Response

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Initializing mediapipe segmentation class.
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Setting up Segmentation function.
segment = mp_selfie_segmentation.SelfieSegmentation()


# Lưu đường dẫn tới thư mục chứa file cascade
# cascade_dir = os.path.join(os.path.dirname(cv2.__file__), 'data')
# background_images = 'D:\\for_test\\bg\\h1.jpg'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['background_image']
        if uploaded_file.filename != '':
            background_image_path = 'D:\\demo2\\uploads'
            uploaded_file.save(background_image_path)
            return render_template('index.html', background_image_path=background_image_path)
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    while True:
        # Read a frame.
        ok, frame = camera.read()

        # Check if frame is not read properly.
        if not ok:
            # Continue to the next iteration to read the next frame.
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the currently selected background image path from the session
        background_image_path = session.get('background_image_path', None)

        # Change the background of the frame.
        if background_image_path:
            output_frame, _ = modifyBackground(frame, background_image=background_image_path,
                                               threshold=0.4, display=False, method='changeBackground')
        else:
            output_frame = frame

        # Encode the output frame as JPEG
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()

        # Yield the frame to the video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def modifyBackground(image, background_image=255, blur=95, threshold=0.3, display=True, method='changeBackground'):
    """
    This function will replace, blur, desature or make the background transparent depending upon the passed arguments.
    Args:
        image: The input image with an object whose background is required to modify.
        background_image: The new background image for the object in the input image.
        threshold: A threshold value between 0 and 1 which will be used in creating a binary mask of the input image.
        display: A boolean value that is if true the function displays the original input image and the resultant image
                 and returns nothing.
        method: The method name which is required to modify the background of the input image.
    Returns:
        output_image: The image of the object from the input image with a modified background.
        binary_mask_3: A binary mask of the input image.
    """

    # Convert the input image from BGR to RGB format.
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the segmentation.
    result = segment.process(RGB_img)

    # Get a binary mask having pixel value 1 for the object and 0 for the background.
    # Pixel values greater than the threshold value will become 1 and the remainings will become 0.
    binary_mask = result.segmentation_mask > threshold

    # Stack the same mask three times to make it a three channel image.
    binary_mask_3 = np.dstack((binary_mask, binary_mask, binary_mask))

    if method == 'changeBackground':
        # Resize the background image to become equal to the size of the input image.
        background_image = cv2.imread(background_image)
        background_image = cv2.resize(background_image, (image.shape[1], image.shape[0]))

        # Create an output image with the pixel values from the original sample image at the indexes where the mask have
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, background_image)

    elif method == 'blurBackground':
        # Create a blurred copy of the input image.
        blurred_image = cv2.GaussianBlur(image, (blur, blur), 0)

        # Create an output image with the pixel values from the original sample image at the indexes where the mask have
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, blurred_image)

    elif method == 'desatureBackground':
        # Create a gray-scale copy of the input image.
        grayscale = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

        # Stack the same grayscale image three times to make it a three channel image.
        grayscale_3 = np.dstack((grayscale, grayscale, grayscale))

        # Create an output image with the pixel values from the original sample image at the indexes where the mask have
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, grayscale_3)

    elif method == 'transparentBackground':
        # Stack the input image and the mask image to get a four channel image.
        # Here the mask image will act as an alpha channel.
        # Also multiply the mask with 255 to convert all the 1s into 255.
        output_image = np.dstack((image, binary_mask * 255))

    else:
        # Display the error message.
        print('Invalid Method')

        # Return
        return

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:
        # Return the output image and the binary mask.
        # Also convert all the 1s in the mask into 255 and the 0s will remain the same.
        # The mask is returned in case you want to troubleshoot.
        return output_image, (binary_mask_3 * 255).astype('uint8')


if __name__ == '__main__':
    app.run(debug=True)