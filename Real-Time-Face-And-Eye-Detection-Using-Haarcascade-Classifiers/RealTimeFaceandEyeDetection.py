import cv2

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

def face_and_eye_detector(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        x = 0
        y = 0
        w = 0
        h = 0
        hx = [0,0,0,0]
        return x,y,w,h,hx;
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        myeyes = []
        hx = [0,0,0,0]
        if eyes == ():
             hx = [0,0,0,0]
        for eex,eey,eew,eeh in eyes:
            hx = [eex,eey,eew,eeh]
            myeyes.append(hx)
    return x,y,w,h,myeyes;


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    x,y,w,h,myeyes = face_and_eye_detector(frame)
    if len(myeyes) == 2:
        for hx in myeyes:
            cv2.rectangle(frame[y:y+h, x:x+w], (hx[0],hx[1]), (hx[0]+hx[2],hx[1]+hx[3]), (127,0,255), 2)
            cv2.putText(frame[y:y+h, x:x+w], 'Beautiful Eye', (hx[0], hx[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
    cv2.putText(frame, 'Handsome Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('Face Detector', frame )
    cv2.imwrite("detected.jpg",frame)

    if cv2.waitKey(1) == ord("q"):
        break
       
cap.release()
cv2.destroyAllWindows()