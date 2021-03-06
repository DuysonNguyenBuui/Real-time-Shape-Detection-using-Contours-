import cv2
import numpy as np

frameWidth = 300
frameHeight = 480

webCamFeed = True
pathImage = "01.jpg"
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass


# create trackbar
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 300, 240)
cv2.createTrackbar("Threshold1", "Parameters", 155, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)


# hien thi tat ca anh dung canh nhau trong cua so
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# tim cac duong bao quanh vat the
def get_contours(img, imgContours):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # loai bo nhieu
    for cnt in contours:
        # dien tich duong bao
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            # ve duong vien len vat the
            cv2.drawContours(imgContours, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContours, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(imgContours, "Point: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(imgContours, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)


while True:
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    # frame, img = cap.read()
    # lam mo anh
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgContours = img.copy()
    # chuyen thanh anh xam
    imGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    # khai bao 2 threshold
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    # su dung canny de do canh
    imgCanny = cv2.Canny(imGray, threshold1, threshold2)
    # kernel
    kernel = np.ones((5, 5))
    # Ph??p to??n gi??n n??? (Dilation), Ph??p to??n n??y c?? t??c d???ng l??m cho ?????i t?????ng ban ?????u trong ???nh t??ng l??n v??? k??ch th?????c (Gi??n n??? ra)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    get_contours(imgDil, imgContours)
    imgStack = stackImages(0.8, ([img, imGray, imgBlur],
                                 [imgCanny, imgDil, imgContours]))
    cv2.imshow("Tracking", imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
