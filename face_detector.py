import cv2
from random import randrange

def img_face_detector(img):
    # load some pre-trained data on face frontals from opencv (haar cascade algorithm)
    trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

    # read image to detect
    img = cv2.imread(img)
    greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # [[upper_left point, width, height]]
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

    # Draw rectangle from the color image
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(128,256), randrange(256),0), 2)

    # shows the image
    cv2.imshow("Face Detector", img)
    # Wait here in the code and listen for a key press
    cv2.waitKey()

def webcam_face_detector():
    trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # defual webcam
    webcam = cv2.VideoCapture(0)
    # iterate forever over frame
    while True:
        # returns a tuple (boolean, actual image)
        successfully_read, cur_frame = webcam.read()
        greyscaled_img = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

        # Draw rectangle from the color image
        for (x,y,w,h) in face_coordinates:
            cv2.rectangle(cur_frame, (x,y), (x+w,y+h), (128,255,0), 2)

        # show the frame
        cv2.imshow("Face Detector: ", cur_frame)
        key = cv2.waitKey(1)

        #Stops if esc is pressed
        if key == 27:
            break
    webcam.release()
#img_face_detector("snowman_member.jpg")
webcam_face_detector()