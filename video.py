import cv2.cv2 as cv2
import numpy as np
from mrcnn import visualize
from random import randint

CLASS_NAMES = ["background", "kayaker"]


def mark_kayaker(frame, boxes, masks, class_ids, scores):
    for box in boxes:
        cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), [255, 0, 0], 4)
        cv2.putText(frame, f"{CLASS_NAMES[class_ids[0]]} {scores[0]}", (int(box[1]), int(box[0])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12))
        contours, _ = cv2.findContours(np.array(masks, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255.0, 255.0, 0.0), lineType=8, thickness=3)
    return frame


def display_frame_cv2(img):
    cv2.imshow('Image', img)
    cv2.waitKey(200)


def mrcnn_visualize(frame, result):
    visualize.display_instances(image=frame,
                                boxes=result['rois'],
                                masks=result['masks'],
                                class_ids=result['class_ids'],
                                class_names=["bg", "kayaker"],
                                scores=result['scores'],
                                show_mask=True)


def predict_kayaker_on_video(model, video_path):
    video = cv2.VideoCapture(video_path)
    codec = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(f'files/output{randint(0, 100000)}.mp4', codec, 30, (1920, 1080))
    while True:
        _, frame = video.read()
        if frame is not None:
            results = model.detect([frame], verbose=1)
            result = results[0]  # rois, class_ids, scores, masks
            marked_frame = mark_kayaker(frame, result['rois'], result['masks'], result['class_ids'], result['scores'])
            out.write(marked_frame)
            display_frame_cv2(marked_frame)
        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()
