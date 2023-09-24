import numpy as np
import cv2
import sys
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from facenet_pytorch import MTCNN
from typing import Any


def predict(predictions: np.ndarray, labels: list) -> list:
    if predictions.shape[0] == 1:
        return labels[np.argmax(predictions)]
    else:
        return labels[np.argmax(predictions.reshape(1, len(predictions)))]


def predict_mtcnn(frame: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mtcnn = MTCNN(
        select_largest=True,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        image_size=160,
        device=device)
    return mtcnn.detect(frame, landmarks=True)


def face_mark(landmark: np.ndarray):
    right_eye = (int(landmark[0, 0]), int(landmark[0, 1]))  # tuple
    left_eye = (int(landmark[1, 0]), int(landmark[1, 1]))
    nose = (int(landmark[2, 0]), int(landmark[2, 1]))
    lip_cright = (int(landmark[3, 0]), int(landmark[3, 1]))
    lip_cleft = (int(landmark[4, 0]), int(landmark[4, 1]))

    return left_eye, right_eye, nose, lip_cright, lip_cleft


def draw_box(frame: np.ndarray, box: np.ndarray,
             left_eye: tuple[int, int], right_eye: tuple[int, int],
             nose: tuple[int, int], lip_cleft: tuple[int, int], lip_cright: tuple[int, int]) -> None:
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
    cv2.circle(frame, left_eye, radius=5, color=(255, 0, 255), thickness=-1)
    cv2.circle(frame, right_eye, radius=5, color=(255, 0, 255), thickness=-1)
    cv2.circle(frame, nose, radius=5, color=(255, 0, 255), thickness=-1)
    cv2.circle(frame, lip_cleft, radius=5, color=(255, 0, 255), thickness=-1)
    cv2.circle(frame, lip_cright, radius=5, color=(255, 0, 255), thickness=-1)


def tensorflow_sess() -> None:
    cap = cv2.VideoCapture(0)
    # Check device for MTCNN
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    print(type(device))

    graph_path = 'tf_files/retrained_graph.pb'
    labels_path = 'tf_files/retrained_labels.txt'

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.io.gfile.GFile(labels_path)]

    # Unpersists graph from file
    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    font = cv2.FONT_HERSHEY_SIMPLEX

    actividad = ""

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (200, 250)
    fontScale = 8
    fontColor = (0, 255, 0)
    lineType = 4
    i = 1
    persona = ""
    box = None
    with tf.compat.v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        actividad = ""
        while cap.isOpened():
            ret, frame = cap.read()
            print(type(ret), type(frame))
            if ret:
                # adhere to TS graph input structure
                frame2 = frame.copy()
                boxes, probs, landmarks = predict_mtcnn(frame=frame2,
                                                        device=device)
                if probs is not None:
                    try:
                        if probs[0] > 0.80:
                            if i > 3:
                                frame = cv2.resize(frame2, (299, 299), interpolation=cv2.INTER_CUBIC)
                                numpy_frame = np.asarray(frame)
                                numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5,
                                                            cv2.NORM_MINMAX)
                                numpy_frame = np.expand_dims(numpy_frame, axis=0)
                                predictions = sess.run(softmax_tensor, {'Mul:0': numpy_frame})
                                persona = predict(predictions, label_lines)
                                i = 0
                                box = boxes[0]
                                landmark = landmarks[0]

                                # Defining points of interest in face
                                right_eye, left_eye, nose, lip_cright, lip_cleft = face_mark(landmark=landmark)
                    except TypeError:
                        print('No face detected in frame')
                        persona = 'None'
                # Define face box = cv2.rectangle(img,(top-left xy: (box[0],box[1])),(bottom-right xy: (box[2],box[3])))
                try:
                    draw_box(frame2, box, right_eye, left_eye, nose, lip_cright, lip_cleft)
                except Exception as e:
                    print(e)
                frame2 = cv2.flip(frame2, 1)  # size = (x=640,y=470,3)
                cv2.putText(frame2, persona, (20, 40), font, 2, fontColor, lineType)

                try:
                    cv2.putText(frame2, f"predictions: {predictions[0]}", (int(20), int(470)), font, 0.5, (255, 255, 0),
                                2)
                    cv2.putText(frame2, f"Face prediction probability: {probs[0]}", (int(20), int(450)), font, 0.5,
                                (255, 0, 0), 2)
                except:
                    cv2.putText(frame2, 'predictions: [0,0]', (int(20), 470), font, 0.5, (255, 255, 0), 2)
                cv2.imshow('frame', frame2)
                i = i + 1
                if cv2.waitKey(32) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()


# Release everything if job is finished
if __name__ == '__main__':
    tensorflow_sess()
