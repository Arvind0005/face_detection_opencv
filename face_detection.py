import cv2

# path
eye_cascade = cv2.CascadeClassifier(r'C:\Users\Arvind\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\cv2\data\haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(r"C:\Users\Arvind\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
path = r'C:\Users\Arvind\Desktop\profile\LRM_EXPORT_50689176198511_20190713_083347385.jpeg'

# Using cv2.imread() method
img = cv2.imread(path,1)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.15,minNeighbors=7)
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),15)
    roi_gray = gray_img[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # To draw a rectangle in eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (100, 255, 255), 8)
resized = cv2.resize(img, (int(img.shape[1]/6),int(img.shape[0]/6)))
cv2.imshow("Gray", resized)
cv2.waitKey((0))