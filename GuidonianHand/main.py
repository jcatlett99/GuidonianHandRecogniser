import copy

import mediapipe as mp
import cv2
import csv
import math
import argparse
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pythonosc.udp_client import SimpleUDPClient

from model import KeyPointClassifier


# from model import PointHistoryClassifier

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=177)
    parser.add_argument("--height", help='cap height', type=int, default=100)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    print("running...")

    args = get_args()

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode


    # Create a gesture recognizer instance with the live stream mode:
    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        if len(result.gestures) == 0:
            print("gesture recognition result: None")
        else:
            print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))


    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='guidonian_gesture_recognizer.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result
    )
    with GestureRecognizer.create_from_options(options) as recognizer:

        print("HERE...")

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.height)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.width)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence, )

        keypoint_classifier = KeyPointClassifier()
        # point_history_classifier = PointHistoryClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        # with open(
        #         'model/point_history_classifier/point_history_classifier_label.csv',
        #         encoding='utf-8-sig') as f:
        #     point_history_classifier_labels = csv.reader(f)
        #     point_history_classifier_labels = [
        #         row[0] for row in point_history_classifier_labels
        #     ]

        history_length = 16
        point_history = deque(maxlen=history_length)

        # Finger gesture history ################################################
        finger_gesture_history = deque(maxlen=history_length)

        i = 0
        while True:
            i += 1
            ret, original_frame = cap.read()
            if not ret:
                break

            original_frame = cv2.flip(original_frame, 1)
            frame = copy.deepcopy(original_frame)

            frame.flags.writeable = False
            hand_results = hands.process(frame)
            frame.flags.writeable = True

            if hand_results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                                      hand_results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(frame, hand_landmarks)
                    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, handedness.classification[0].label[0:], (brect[0] + 5, brect[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    # Landmark calculation
                    # landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    images = []
                    results = []
                    # STEP 3: Load the input image.
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    # STEP 4: Recognize gestures in the input image.
                    recognizer.recognize_async(image, i)
                    options.result_callback

                    client = SimpleUDPClient("127.0.0.1", 8000)  # Create client
                    client.send_message("/1/fader1", ((brect[0] + brect[2]) / 2))  # Send float message
                    client.send_message("/1/fader2", ((brect[1] + brect[3]) / 2))  # Send float message

                    images.append(image)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # images = []
    # results = []
    # # STEP 3: Load the input image.
    # image = mp.Image.create_from_file(image_file_path)
    #
    # # STEP 4: Recognize gestures in the input image.
    # recognition_result = recognizer.recognize(image)
    #
    # # STEP 5: Process the result. In this case, visualize it.
    # images.append(image)
    # top_gesture = recognition_result.gestures[0][0]
    # hand_landmarks = recognition_result.hand_landmarks
    # results.append((top_gesture, hand_landmarks))

    print("results: ", results[0][0].category_name)

    exit()

    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'xtick.labelbottom': False,
        'xtick.bottom': False,
        'ytick.labelleft': False,
        'ytick.left': False,
        'xtick.labeltop': False,
        'xtick.top': False,
        'ytick.labelright': False,
        'ytick.right': False
    })

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles


    def display_one_image(image, title, subplot, titlesize=16):
        """Displays one image along with the predicted category name and score."""
        plt.subplot(*subplot)
        plt.imshow(image)
        if len(title) > 0:
            plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment': 'center'},
                      pad=int(titlesize / 1.5))
        return (subplot[0], subplot[1], subplot[2] + 1)


    def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
        """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
        # Images and labels.
        images = [image.numpy_view() for image in images]
        gestures = [top_gesture for (top_gesture, _) in results]
        multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

        # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
        rows = int(math.sqrt(len(images)))
        cols = len(images) // rows

        # Size and spacing.
        FIGSIZE = 13.0
        SPACING = 0.1
        subplot = (rows, cols, 1)
        if rows < cols:
            plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
        else:
            plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

        # Display gestures and hand landmarks.
        for i, (image, gestures) in enumerate(zip(images[:rows * cols], gestures[:rows * cols])):
            title = f"{gestures.category_name} ({gestures.score:.2f})"
            dynamic_titlesize = FIGSIZE * SPACING / max(rows, cols) * 40 + 3
            annotated_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGBA2BGR)

            for hand_landmarks in multi_hand_landmarks_list[i]:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                    hand_landmarks
                ])

                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

        # Layout.
        plt.tight_layout()
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
        plt.show()


    display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
