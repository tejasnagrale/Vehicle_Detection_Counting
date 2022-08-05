import cv2
import numpy as np

cap=cv2.VideoCapture("video.mp4")
count_line_position=380
detect_up=[]
detect_down=[]
offset=5
counter_up=0
counter_down=0

algo=cv2.bgsegm.createBackgroundSubtractorMOG()
while True:
    ret, frame1=cap.read()
    frame1=cv2.resize(frame1,(854,480))
    ##y:y1,x:x1
    roi=frame1[260:383,130:379]
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(5,5),0)
    img_sub = algo.apply(blur)
    dilate=cv2.dilate(img_sub,np.ones((3,3)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape,h= cv2.findContours(dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (450, count_line_position), (800, count_line_position), (255, 0, 0), 2)
    cv2.line(frame1, (70, count_line_position), (450, count_line_position), (167, 150, 240), 2)


    for (i,c) in enumerate(counterShape):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter= (w>=60) and (h>=60)
        if not validate_counter:
            continue

        #Create Center point of bounding box
        cx=int((x+x+w)/2)
        cy=int((y+y+h)/2)

        #detect center in bounding box
        center=(cx,cy)
        detect_up.append(center)
        detect_down.append(center)
        cv2.circle(frame1,(cx,cy),2,(0,0,255),-1)

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x,y) in detect_up:
            if y<(count_line_position+offset) and y>(count_line_position-offset) and x>=460 and x<=800:
                counter_up+=1
            cv2.line(frame1, (450, count_line_position), (800, count_line_position), (0, 128, 255), 3)
            detect_up.remove((x,y))
            print("Vehicle Counter:"+str(counter_up))

        for (x1,y1) in detect_down:
            if y1<(count_line_position+offset) and y1>(count_line_position-offset) and x1>=70 and x1<=400:
                counter_down+=1
            cv2.line(frame1, (70, count_line_position), (400, count_line_position), (0, 128, 255), 3)
            detect_down.remove((x1,y1))
            print("Vehicle Counter1:"+str(counter_down))



    cv2.putText(frame1, "VEHICLE UP :" + str(counter_up), (450, 90), cv2.FONT_ITALIC, 1, (0, 0, 255), 4)
    cv2.putText(frame1, "VEHICLE DOWN :" + str(counter_down), (70, 150), cv2.FONT_ITALIC, 1, (0, 0, 255), 4)
        #center = center_handle(x, y, w, h)
        #detect.append(center)
        #cv2.circle(frame1, center, 4, (0, 0, 255), -1)


    cv2.imshow("Vehicle Video",frame1)
    #cv2.imshow("ROI",roi)
    key = cv2.waitKey(60)
    if key ==27:
        break


cap.release()
cv2.destroyAllWindows()
#print(("Package Imported"))





#cv2.destroyAllWindow()
#  cv2.release()