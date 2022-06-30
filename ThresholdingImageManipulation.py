import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def loadingImages():
    img = cv.imread('car.jpg', cv.IMREAD_COLOR)
    # choose for the car to be in color, can also try GRAYSCALE instead
    # if image is not in same folder as script, use filepath as variable
    cv.imshow('Car', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # pass through title and image. waitKey waits for key to be pressed before continuing, delay of 0 ms
    # try and show the image using matplotlib
    plt.imshow(img)
    plt.show()
    # matplotlib uses BGR color scheme as opposed to RGB, so image must be converted
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    plt.imshow(img)
    plt.show()
    # now we can edit our images
    cv.line(img, (50, 50), (250, 250), (255, 255, 0), 15)
    cv.rectangle(img, (350, 450), (500, 350), (0, 255, 0), 5)
    cv.circle(img, (500, 200), 100, (255, 0, 0), 7)
    # for line and rectagle, first parameters are starting and end point (x, y), them pass a triple with RGB values for color
    # lastly specify the thickness of the line
    x_values = np.linspace(100, 900, 50)
    y_values = np.sin(x_values) * 100 + 300

    plt.imshow(img, cmap='gray')
    plt.plot(x_values, y_values, 'c', linewidth=5)
    plt.show()
    # here we plotted a sine function onto our car for no apparent reason
    # we can also copy and cut parts of our image
    img[0:200, 0:300] = [0, 0, 0]
    # replace all pixels on x axis from 0 to 200 and y axis from 0 to 300
    copypart = img[300:500, 300:700]
    img[100:300, 100:500] = copypart
    # store part of the image and display on another part of image
    img[300:500, 300:700] = [0, 0, 0]
    plt.imshow(img)
    plt.show()
    # if you want to save your image after processing
    cv.imwrite('car_new.jpg', img)

def loadingVideo():
    # lets load a video, if not in same folder pass filepath
    video = cv.VideoCapture('video.mp4')
    # start an endless loop, reading one frame after the other using read, show it using imshow, if x key is pressed script terminates
    # one frame every 30 m/s is 33 frames per second, can change all values
    # if we don't terminate script manually, once finished script will get error, use if statement
    while True:
        ret, frame = video.read()
        if ret:
            cv.imshow('Video', frame)

            if cv.waitKey(30) == ord('x'):
                break

        else:
            video = cv.VideoCapture('video.mp4') # will continuously play
            #break # will break out once done

    video.release()
    cv.destroyAllWindows()
    # if you want to save your video, define a few variables
    capture = cv.VideoCapture(0) # 0 is used to access primary camera, use 1 for another, or sub 0 with filepath
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    writer = cv.VideoWriter('video.avi', fourcc, 60.0, (640, 480))
    # fourcc is encoding used for video, save file name, codec, frame rate (60 fps), and resolution
    # use loop to write frames into file
    while True:
        ret, frame = capture.read()

        writer.write(frame)

        cv.imshow('Cam', frame)

        if cv.waitKey(1) == ord('x'):
            break

    capture.release()
    writer.release()
    cv.destroyAllWindows()

def logoOntoImage():
    # creating a semi-transparent logo to put on image
    img1 = cv.imread('laptop.jpg')
    img2 = cv.imread('logo.png')
    # convert logo to grayscale because we are only interested in white color
    # threshold function gets passed grayscale logo, and which color value we are changing to which
    # 180 for light gray, 255 for white
    logo_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    ret, mask = cv.threshold(logo_gray, 180, 255, cv.THRESH_BINARY_INV)
    cv.imshow('Mask', mask)
    cv.waitKey(0)
    # invert to get white letter and black background
    #mask_inv = cv.bitwise_not(mask) #alternative way to invert
    mask_inv = np.invert(mask)
    # insert logo onto image
    rows, columns, channels = img2.shape
    area = img1[0:rows, 0:columns]
    # shape attribute gets resolution and channels, then save into variable
    img1_bg = cv.bitwise_and(area, area, mask=mask_inv)
    img2_fg = cv.bitwise_and(img2, img2, mask=mask)
    # define 2 parts in upper left corner, define background of first image, then add to second image in next step
    result = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:columns] = result
    cv.imshow('Result', img1)
    cv.waitKey(0)
    cv.destroyAllWindows()

def makeImagesReadable():
    # convert img to grayscale and apply binary thresholding (try it)
    img = cv.imread('bookpage.jpg')
    cv.namedWindow("output", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions
    img = cv.imread("bookpage.jpg")                    # Read image
    img = cv.resize(img, (960, 540))                # Resize image
    cv.imshow("output", img)                       # Show image
    cv.waitKey(0)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, threshold = cv.threshold(img_gray, 32, 255, cv.THRESH_BINARY)
    # every pixel that is whiter than 32(dark gray) will be converted to 255(white), everything else to 0(black)
    cv.imshow('Image', threshold)
    cv.waitKey(0)

    # now try adaptive Gaussian Thresholding
    gaus = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 81, 4)
    # pass gray picture,parameters passed to threshold are block size specifying how large in pixels
    # the larger the value the less details the image will have, value needs to be odd
    # nect value sharpens the image, may need to experiment with
    cv.imshow('Gaus', gaus)
    cv.waitKey(0)
    cv.destroyAllWindows()

def filtering():
    img = cv.imread('parrot.jpg')
    # resize image
    cv.namedWindow("output", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions
    img = cv.imread("parrot.jpg")                    # Read image
    img = cv.resize(img, (960, 540))                # Resize image
    cv.imshow("output", img)                       # Show image
    cv.waitKey(0)
    # convert to hsv
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # hue, saturation, value (color, strong, bright)
    minimum = np.array([100, 60, 0])
    maximum = np.array([255, 255, 255])
    # define minimum and maximum colors we are looking for, we want everything inbetween
    # for us between 100 and 255 gives orange hues to red, saturation of 60 to 255 to avoid gray values
    # ignore brightness, 0-255
    mask = cv.inRange(hsv, minimum, maximum)
    result = cv.bitwise_and(img, img, mask = mask)
    # sets all pixels in our range to white, all rest to black
    cv.imshow('Mask', mask)
    cv.waitKey(0)
    cv.imshow('Result', result)
    cv.waitKey(0)
    # optimize result by implementing blurring and smoothing
    # make less sharp but reduce background noise
    averages = np.ones((15, 15), np.float32) / 255
    smoothed = cv.filter2D(result, -1, averages)
    cv.imshow('Smoothed', smoothed)
    cv.waitKey(0)
    # try reverse order
    smoothed2 = cv.filter2D(mask, -1, averages)
    smoothed2 = cv.bitwise_and(img, img, mask=smoothed2)
    cv.imshow('Smoothed2', smoothed2)
    cv.waitKey(0)
    # try another blur method
    blur = cv.GaussianBlur(result, (15, 15), 0)
    median = cv.medianBlur(result, 15)
    cv.imshow('Median', median)
    cv.waitKey(0)
    cv.destroyAllWindows()

def filteringCameraData():
    camera = cv.VideoCapture(0)

    while True:
        _, img = camera.read()
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        minimum = np.array([100, 60, 0])
        maximum = np.array([255, 255, 255])

        mask = cv.inRange(hsv, minimum, maximum)

        median = cv.medianBlur(mask, 15)
        median = cv.bitwise_and(img, img, mask=median)

        cv.imshow('Median', median)

        if cv.waitKey(5) == ord('x'):
            break

    cv.destroyAllWindows()
    camera.release()

def edgeDetection():
    img = cv.imread('room.jpg')
    edges = cv.Canny(img, 100, 100)
    cv.imshow('Edges', edges)

    cv.waitKey(0)
    cv.destroyAllWindows()

def motionDetection():
    video = cv.VideoCapture(0)
    subtractor = cv.createBackgroundSubtractorMOG2(20, 50)
    # first parameter is length of history, or how far back we look for movement, second one is threshold
    # run endless loop to alter frames
    while True:
        _, frame = video.read()
        mask = subtractor.apply(frame)

        cv.imshow('Mask', mask)

        if cv.waitKey(5) == ord('x'):
            break

    cv.destroyAllWindows()
    video.release()

def facialRecognition():
    # use this as a way to classify your chosen picture
    faces_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv.imread('people.jpg')
    img = cv.resize(img, (1400, 900))

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    faces = faces_cascade.detectMultiScale(gray, 1.3, 5)
    # convert to grayscale and use detectmultiselect to help identify faces along with the cascade
    # first parameter is the scaling factor, which will be higher the higher quality image you have
    # second is the minimum amount of neighbor classification for a match
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(img, 'FACE', (x,y+h+30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv.imshow('FACES',img)
        cv.waitKey(0)

    cv.destroyAllWindows()

    # iterate over each face and get coordinates, the width and the height. We use those to draw box around faces

loadingImages()
loadingVideo()
logoOntoImage()
makeImagesReadable()
filtering()
filteringCameraData()
edgeDetection()
motionDetection()
facialRecognition()
