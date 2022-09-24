import cv2
import imutils
import numpy as np
import pandas as pd

# FIRST STEP
# Load iamge, grayscale, adaptive threshold
image = cv2.imread('../img/rambu_belok kiri.jpg')
image = imutils.resize(image, width=1000, height=1000)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-2]

mask = np.zeros(image.shape[:-1], np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

dst = cv2.bitwise_and(image, image, mask=masked)

segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

x,y,w,h = cv2.boundingRect(masked)
cropped = segmented[y-1:y+h+1,x-1:x+w+1]

# SECOND STEP
cropped = imutils.resize(cropped, height=32, width=32)

gray = cv2.cvtColor(cropped,cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(gray, np.mean(gray), 1, cv2.THRESH_BINARY)
feature = np.concatenate((np.sum(thresh, axis=0), np.sum(thresh, axis=1)), axis=None)
print(feature)


print(len(cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)))


cv2.imshow('cropped', cropped)
# cv2.imshow('segmented', cv2.resize(segmented, (400,400)))
# # cv2.imshow('segmented2', cv2.resize(segmented2, (400,400)))
# # cv2.imshow('thresh', cv2.resize(thresh, (400,400)))
# # cv2.imshow('opening', cv2.resize(opening, (400,400)))
# # cv2.imshow('image', cv2.resize(image, (400,400)))
cv2.waitKey()