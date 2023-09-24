import numpy as np
import cv2
import sys
import time
from pathlib import Path


def make_image_dir(path: Path = Path('tf_files/images'),
                   person: str = None) -> None:
    if not Path(path / person).is_dir():
        Path(path / person).mkdir()
        print(f'Created file {person}!')
    else:
        print(f"File {person} already exists")


def run():
    # define the name of the directory to be created
    person = sys.argv[1]

    make_image_dir(person=person)

    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (200, 250)
    fontScale = 8
    fontColor = (0, 255, 0)
    lineType = 4

    # Preparar
    salir = False
    j = 3
    while not salir:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.putText(frame, str(j),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if j == 0:
            salir = True
        time.sleep(1)
        j = j - 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Grabo las imagenes
    salir = False
    j = 1000
    count = 1
    while not salir:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # cv2.imwrite(person + "/frame%d.jpg" % count, frame)  # save frame as JPEG file
        cv2.imwrite(f'{person}/frame{count}.jpg', frame)
        count += 1
        cv2.putText(frame, str(j),
                    (20, 40),
                    font,
                    2,
                    fontColor,
                    lineType)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if j == 1:
            break
        j = j - 1

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
