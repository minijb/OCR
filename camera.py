from picamera import PiCamera
import time
camera = PiCamera()
camera.resolution = (1920,1080)
camera.start_preview()
time.sleep(2)
camera.capture('test.jpg')
camera.stop_preview()
camera.close()


