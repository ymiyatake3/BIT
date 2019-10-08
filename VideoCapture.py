import cv2
import imutils

# 選擇攝影機
cap = cv2.VideoCapture(0)
center = []

while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()

    # 顯示圖片
    #cv2.imshow('frame', frame)
    
    # Grayscale Image
    gscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gscale', gscale)
    
    # Binary Image
    ret, thresh = cv2.threshold(gscale, 250, 255, cv2.THRESH_BINARY)
    #cv2.imshow('threshold', thresh)

    # Detect all contours
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img_contour, contours, -1, (0,255,0), 3)
    #cv2.imshow('img_contour', img_contour)
    
    #img_contour = cv2.cvtColor(cnts[0], cv2.COLOR_GRAY2BGR)
    cnts = imutils.grab_contours(cnts)

    ret, img_record = cv2.threshold(gscale, 255, 255, cv2.THRESH_BINARY)
    centerpoint = False
    for c in cnts:
        area = cv2.contourArea(c)
        if(area > 4000):
            M = cv2.moments(c)
            if not M["m00"] == 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #cv2.drawContours(img_contour, [c], -1, (255, 0, 0), 3)
                
                # Draw a dot on the center point
                cv2.circle(gscale, (cX, cY), 3, (0, 0, 255), -1)
                
                center.append([cX, cY])
                centerpoint = True
                
        #cv2.imshow('img_contour', img_contour)
        cv2.imshow('gscale', gscale)
    if not centerpoint:
        center.clear()
        #ret, img_record = cv2.threshold(gscale, 255, 255, cv2.THRESH_BINARY)

    for row in range(len(center)):
        cv2.circle(img_record, (center[row][0], center[row][1]), 15, (255, 255, 255), -1)
    
    cv2.imshow('img_record', img_record)

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
