import os
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


ht = 480
wd = 640
smallRegionTh = 0.001*ht*wd  # 0.01% of total pixels in image
largeRegionTh = 0.3*ht*wd   # 50% of total pixels in image
objectDimTh = 0.2*ht

# background image
bgImg = cv2.imread('IMG_2043.JPG')
bgImgResized = imutils.resize(bgImg,wd,ht)
bgImgGray = cv2.cvtColor(bgImgResized, cv2.COLOR_BGR2GRAY)

strFolderName = "../Images"
fileList = os.listdir(strFolderName)

for idxFileName in fileList:
    strFileName = strFolderName + "/" + idxFileName
    print strFileName

    # Input image
    Img = cv2.imread(strFileName)
    ImgResized = imutils.resize(Img,wd,ht)
    ImgGray = cv2.cvtColor(ImgResized, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Input Image',ImgResized)
    # cv2.imshow('Input Gray Image',ImgGray)

    subImg = cv2.absdiff(ImgGray, bgImgGray)
    # cv2.imshow('Subtracted Image',subImg)

    # Otsu's thresholding after Gaussian filtering
    blurredImg = cv2.GaussianBlur(subImg, (5, 5), 0)
    ret3, thresholdImgRGB = cv2.threshold(blurredImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('Thresholded RGB Image',thresholdImgRGB)


    ImgHSV = cv2.cvtColor(ImgResized, cv2.COLOR_BGR2HSV)  # ,ImgS, ImgV
    ImgH, ImgS, ImgV = ImgHSV[:, :, 0], ImgHSV[:, :, 1], ImgHSV[:, :, 2]
    # cv2.imshow('Hue Image',ImgH)
    # cv2.imshow('Sat Image',ImgS)
    # cv2.imshow('Val Image',ImgV)

    # Otsu's thresholding after Gaussian filtering
    blurredImgH = cv2.GaussianBlur(ImgH, (5, 5), 0)
    ret3H, thresholdImgH = cv2.threshold(blurredImgH, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholdImgHinv = cv2.bitwise_not(thresholdImgH)
    # cv2.imshow('Thresholded Hue Image',thresholdImgHinv)

    # Otsu's thresholding after Gaussian filtering
    blurredImgS = cv2.GaussianBlur(ImgS, (5, 5), 0)
    ret3H, thresholdImgS = cv2.threshold(blurredImgS, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('Thresholded Sat Image',thresholdImgS)

    # Otsu's thresholding after Gaussian filtering
    blurredImgV = cv2.GaussianBlur(ImgV, (5, 5), 0)
    ret3H, thresholdImgV = cv2.threshold(blurredImgV, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('Thresholded Val Image',thresholdImgV)

    # AND Thresholded Hue Image and Thresholded Sat Image
    thresholdImgHandS = cv2.bitwise_and(thresholdImgHinv, thresholdImgS)
    # cv2.imshow('Segmented Image 1',thresholdImgHandS)

    # OR Thresholded Image and Thresholded Value Image
    thresholdImgOrV = cv2.bitwise_or(thresholdImgRGB, thresholdImgV)
    # cv2.imshow('Segmented Image 2',thresholdImgOrV)

    # OR Thresholded Hue and Sat Image and Thresholded RGB image
    segmentedImg = cv2.bitwise_or(thresholdImgHandS, thresholdImgOrV)
    # cv2.imshow('Segmented Image',segmentedImg)
    # cv2.waitKey(0)


    # Find connected components
    output = cv2.connectedComponentsWithStats(segmentedImg, 4, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    for labelIdx, stat in zip(range(0, num_labels), stats):
        if 0 in stat[0:4] or ht in stat[0:4] or wd in stat[0:4]:
            segmentedImg[labels == labelIdx] = 0

    # Find connected components
    output = cv2.connectedComponentsWithStats(segmentedImg, 4, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    modStats = []
    modCentroids = []
    # Remove too small and too big connected components
    for labelIdx, stat, centroid in zip(range(0, num_labels), stats, centroids):
        if stat[4] <= smallRegionTh or stat[4] >= largeRegionTh:
            segmentedImg[labels == labelIdx] = 0
            continue
        else:
            modStats.append(stat)
            modCentroids.append(centroid)

    # Plot centroids and draw bounding box
    # outImg = ImgResized.copy()
    # for stat,centroid in zip(modStats,modCentroids):
    #     cv2.rectangle(outImg, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), [0, 0, 255])
    #     cv2.circle(outImg, (int(centroid[0]),int(centroid[1])) , 2, [0,255,0])
    #
    # cv2.imshow('Output Image',outImg)
    # cv2.waitKey(0)

    # print centroids

    ################ Ellipse Fitting Start
    # outImgElps = ImgResized.copy()
    # # contours,hierarchy = cv2.findContours(segmentedImg, cv2.CV_RETR_EXTERNAL)
    # im2, contours, hierarchy = cv2.findContours(segmentedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # for cnt in contours:
    #     if len(cnt) > 10:
    #         ellipse = cv2.fitEllipse(cnt)
    #         if (ellipse[1][0] / ellipse[1][1]) > 5 or (ellipse[1][1] / ellipse[1][0]) > 5 or ellipse[1][
    #             0] > objectDimTh or ellipse[1][1] > objectDimTh:
    #             continue
    #
    #         cv2.ellipse(outImgElps, ellipse, (0, 255, 0), 2)
    #         # print ellipse
    ################ Ellipse Fitting End

    outImgMarker = ImgResized.copy()
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(segmentedImg, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # cv2.imshow('FG',sure_fg)
    # cv2.imshow('BG',sure_bg)
    # cv2.waitKey(0)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # ret, markers = cv2.connectedComponentsWithStats(sure_fg, 4, cv2.CV_32S)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # plt.figure()
    # plt.imshow(markers, cmap=cm.jet, alpha=0.75)
    # plt.show()

    markers = cv2.watershed(outImgMarker, markers)
    outImgMarker[markers == -1] = [0, 255, 0]

    plt.figure(1,figsize=(17,10))
    plt.subplot(121)
    # plt.imshow(ImgResized)
    plt.imshow(cv2.cvtColor(ImgResized, cv2.COLOR_BGR2RGB) )
    plt.subplot(122)
    # plt.imshow(outImgMarker)
    plt.imshow(cv2.cvtColor(outImgMarker, cv2.COLOR_BGR2RGB))
    # plt.imshow(outImgMarker, cmap=cm.jet, alpha=0.75)
    # plt.show()
    plt.draw()
    plt.pause(1)

    filenameDir, file_name = os.path.split(strFileName)
    filename, ext = os.path.splitext(file_name)
    filenamePar = os.path.dirname(filenameDir)
    strFileNameStore = filenamePar + '/'+ filename + '_R' + ext
    cv2.imwrite(strFileNameStore,outImgMarker)
