import numpy as np
import cv2
# cap = cv.VideoCapture('/Users/rishabparuchuri/Desktop/Camshift/Clips/clips.mp4')
# # take first frame of the video
# ret, frame = cap.read()
# # setup initial location of window
# x, y, width, height = 300, 200, 100, 50
# track_window = (x, y, width, height)
# # set up the ROI for tracking
# roi = frame[y:y+height, x: x+width]
# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255)))
# roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# # Setup the termination criteria, either 10 iteration or move by at least 1 pt
# term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
# cv.imshow('roi', roi)
# while 1:
#     ret, frame = cap.read()
#     if ret:
#
#         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
#         # apply meanshift to get the new location
#         ret, track_window = cv.CamShift(dst, track_window, term_crit)
#
#         # Draw it on image
#         pts = cv.boxPoints(ret)
#         print(pts)
#         pts = np.int0(pts)
#         final_image = cv.polylines(frame, [pts], True, (0, 255, 0), 2)
#         cv.rectangle(frame, (800, 500), (850, 550), 2)
#         # x,y,w,h = track_window
#         # final_image = cv.rectangle(frame, (x,y), (x+w, y+h), 255, 3)
#
#         cv.imshow('dst', dst)
#         cv.imshow('final_image', final_image)
#         k = cv.waitKey(30) & 0xff
#         if k == 27:
#             break
#     else:
#         break

#
# cap = cv2.VideoCapture('/Users/rishabparuchuri/Desktop/Camshift/Clips/skeet-8.mov')
#
# # take first frame of the video
# ret, frame = cap.read()
# l_b = np.array([0, 60, 80])  # lower hsv bound for orange
# u_b = np.array([15, 255, 255])  # upper hsv bound to orange
# while ret == True:
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, l_b, u_b)
#     contours, _= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#     max_contour = contours[0]
#     print(cv2.contourArea(max_contour))
#     for contour in contours:
#         if cv2.contourArea(contour)> 60.0:
#             approx=cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True),True)
#             x,y,w,h=cv2.boundingRect(approx)
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#
#
#
#
#     cv2.imshow("frame", frame)
#     # cv2.imshow("mask", mask)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
# cv2.waitKey(0)
# cv2.destroyAllWindows()





cap = cv2.VideoCapture('/Users/rishabparuchuri/Desktop/Camshift/Clips/skeet-8.mov')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output-8.mp4', fourcc, fps, (frame_width, frame_height))

# take first frame of the video
ret, frame = cap.read()
l_b = np.array([0, 60, 80])  # lower hsv bound for orange
u_b = np.array([15, 255, 255])  # upper hsv bound to orange
while ret == True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, l_b, u_b)
    contours, _= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_contour = contours[0]
    print(cv2.contourArea(max_contour))
    for contour in contours:
        if cv2.contourArea(contour)> 60.0:
            approx=cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True),True)
            x,y,w,h=cv2.boundingRect(approx)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)




    cv2.imshow("frame", frame)
    output.write(frame)
    # cv2.imshow("mask", mask)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
