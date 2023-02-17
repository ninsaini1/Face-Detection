import cv2


facec = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        rval, fr = self.video.read()

        while rval:
            rval, fr = self.video.read()
            gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(fr,(x,y),(x+w, y+h),(127, 255,0),3)
            cv2.imshow("preview", fr)
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        self.video.release()
        cv2.destroyWindow("preview")


cv2.namedWindow("preview")
vd=VideoCamera()
vd.get_frame()



