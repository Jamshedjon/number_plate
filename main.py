# pip install -r requirements.txt


import cv2 as cv
import argparse, pytesseract
import re,time
# import numpy as np
# import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

dor = 0
count = 0

licenced_numbers = [
    1233,
    1234,
    3553,
    4646,
    6757,
    5867,
    3453,
    3453,
    5675,
    8698,
    2190,
    803,
    718,
]


def find_number(text):
    global dor
    for number in licenced_numbers:
        number = str(number)
        if number in text:
            print(number, "35")
            dor = 1


min_area = 500


################
def recognize_plate2(box):
    gray = cv.cvtColor(box, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=4, fy=4, interpolation=cv.INTER_CUBIC)
    ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilation = cv.dilate(thresh, rect_kern, iterations=1)
    try:
        contours, hierarchy = cv.findContours(
            dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
    except:
        ret_img, contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
    sorted_contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])
    im2 = gray.copy()
    plate_num = ""
    count = 0
    for cnt in sorted_contours:
        x, y, w, h = cv.boundingRect(cnt)
        height, width = im2.shape
        if height / float(h) > 6:
            continue
        ratio = h / float(w)
        if ratio < 1:
            continue
        if width / float(w) > 20:
            continue
        area = h * w
        if area < 100:
            continue
        rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = thresh[y - 5 : y + h + 5, x - 5 : x + w + 5]
        roi = cv.bitwise_not(roi)
        count += 1
        if count == 3 or count > 6:
            try:
                text = pytesseract.image_to_string(
                    roi,
                    config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3",
                )
                clean_text = re.sub(r"[\W_]+", "", text)
                plate_num += clean_text
            except:
                text = None
        else:
            try:
                text = pytesseract.image_to_string(
                    roi,
                    config="-c tessedit_char_whitelist=0123456789 --psm 8 --oem 3",
                )
                clean_text = re.sub(r"[\W_]+", "", text)
                plate_num += clean_text
            except:
                text = None
    if plate_num != None:
        # cv.imshow("Character's Segmented", im2)
        # cv.waitKey(500)
        return plate_num


################


def detextCarNumber(frame):
    global count
    frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    cascade = plate_cascade.detectMultiScale(frame_gray)
    if type(cascade) != "tuple" and dor == 0:
        count += 1
        for x, y, w, h in cascade:
            area = w * h
            if area > min_area:
                frame_gray = frame[y : y + h, x : x + w]
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                result = recognize_plate2(frame_gray)
                if result:
                    find_number(result)

    cv.imshow("Result", frame)


parser = argparse.ArgumentParser(description="Code for Cascade Classifier tutorial.")
parser.add_argument(
    "--plate_cascade",
    help="Path to number cascade.",
    default="model/haarcascade_russian_plate_number.xml",
)
parser.add_argument("--camera", help="Camera divide number.", type=int, default=0)
args = parser.parse_args()
plate_cascade_name = args.plate_cascade

plate_cascade = cv.CascadeClassifier()

# -- 1. Load the cascades
if not plate_cascade.load(cv.samples.findFile(plate_cascade_name)):
    print("--(!)Error loading face cascade")
    exit(0)

camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
#########################################
# cap = cv.VideoCapture("rtsp://admin:ferpi2024@192.168.1.64:554")

if not cap.isOpened:
    print("--(!)Error opening video capture")
    exit(0)


def main():
    while True:
        time.sleep(0.3)
        _, frame = cap.read()
        if frame is None:
            print("--(!) No captured frame -- Break!")
            break
        close_capture = detextCarNumber(frame)

        if close_capture:
            break

        if cv.waitKey(10) == 27:
            break


if __name__ == "__main__":
    main()
