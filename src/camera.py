import cv2
import numpy as np
import matplotlib.pyplot as plt

DEVICE_ID = 3
BLUR_SIZE = 7
SCALE = 1


def camera() -> cv2.VideoCapture:
    return cv2.VideoCapture(DEVICE_ID)


def get_image(cap: cv2.VideoCapture) -> np.ndarray:
    ret, image = cap.read()

    if not ret:
        return None

    # reduce highlights

    image = cv2.convertScaleAbs(image, alpha=1, beta=0)

    return image


def get_blur_image(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (BLUR_SIZE, BLUR_SIZE), 0)


def get_gray_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_canny_image(image: np.ndarray) -> np.ndarray:
    return cv2.Canny(image, 100, 200)


def get_resize_image(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, None, fx=SCALE, fy=SCALE)


def get_video_stream(cap: cv2.VideoCapture):
    while True:
        image = get_image(cap)

        if image is None:
            continue

        blurred_image = get_blur_image(image)
        gray_image = get_gray_image(blurred_image)
        canny_image = get_canny_image(gray_image)

        # Convert the grayscale images back to RGB
        gray_image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        canny_image_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)

        resized_images = [
            get_resize_image(img)
            for img in [image, blurred_image, gray_image_rgb, canny_image_rgb]
        ]
        combined_image = np.hstack(resized_images)
        combined_image = np.vstack(
            [np.hstack(resized_images[:2]), np.hstack(resized_images[2:])]
        )
        cv2.imshow("Image Stream", image)

        cv2.imshow("Video Stream", combined_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    cap = camera()
    get_video_stream(cap)


if __name__ == "__main__":
    main()
