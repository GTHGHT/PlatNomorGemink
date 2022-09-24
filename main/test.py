import cv2
import imutils
import numpy as np
import pandas as pd
import os

df = pd.DataFrame(columns=['nama', 'fitur', 'segi'])
folder_dir = "../img"
for namagambar in os.listdir(folder_dir):
    if (namagambar.endswith(".jpg")): 
        # FIRST STEP: 
        # Load image, grayscale, adaptive threshold
        image = cv2.imread('img/'+ namagambar)
        image = imutils.resize(image, width=1000, height=1000)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(
            gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

        cnt = sorted(
            cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], 
            key=cv2.contourArea
            )[-2]

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
        feature = np.concatenate(
            (np.sum(thresh, axis=0), np.sum(thresh, axis=1)), axis=None)
        segi = len(cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True))
        
        new_row = pd.Series({'nama':namagambar, 'fitur':feature, 'segi':segi})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        
        print(df)
        df.to_csv('rambu_dataset.csv')