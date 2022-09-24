import cv2
import imutils
import numpy as np

# Load iamge, grayscale, adaptive threshold
image = cv2.imread('../img/rambu7.jpg')
image = imutils.resize(image, width=1000, height=1000)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-2]

print(cv2.contourArea(cnt))
print(cv2.arcLength(cnt,True))
print(cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True))

mask = np.zeros(image.shape[:-1], np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

dst = cv2.bitwise_and(image, image, mask=masked)

segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

x,y,w,h = cv2.boundingRect(masked)
cropped = segmented[y-1:y+h+1,x-1:x+w+1]

cropped = imutils.resize(cropped, height= 64, width=64)

# # Fill rectangular
# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

# # Morph open
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
# #
# # Draw rectangles
# cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)

cv2.imshow('cropped', cropped)
cv2.imshow('segmented', cv2.resize(segmented, (400,400)))
# cv2.imshow('segmented2', cv2.resize(segmented2, (400,400)))
# cv2.imshow('thresh', cv2.resize(thresh, (400,400)))
# cv2.imshow('opening', cv2.resize(opening, (400,400)))
# cv2.imshow('image', cv2.resize(image, (400,400)))
cv2.waitKey()