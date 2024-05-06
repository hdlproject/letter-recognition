import cv2


class VideoReader(object):
    def __init__(self):
        self.vid = cv2.VideoCapture(0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exit()

    def exit(self):
        self.vid.release()
        cv2.destroyAllWindows()

    @staticmethod
    def exit_condition():
        return cv2.waitKey(1) & 0xFF == ord('q')

    def read(self):
        while True:
            ret, frame = self.vid.read()
            cv2.imshow('frame', frame)
            yield ret, frame
