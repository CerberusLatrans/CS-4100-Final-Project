#delete later

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def apply_canny(img, type, id, display=True):
    """Applies canny filter to image"""
    img = cv.imread('./images/neu.png',0)
    img = cv.medianBlur(img,5)
    if type == 'dirty':
        # ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
        # th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
        #     cv.THRESH_BINARY,11,2)
        # th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #     cv.THRESH_BINARY,11,2)
        # edges = cv.Canny(th1, 200, 400)
        # attempt 2
        
        ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        blur = cv.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #edges = cv.Canny(th3, 200, 400)
        # blur = cv.bilateralFilter(img,9,75,75)
        # ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #edges = cv.Canny(th3, 200, 400)
        
        #avging: 
        
        kernel = np.ones((5,5),np.float32)/25
        dst = cv.filter2D(img,-1,kernel)
        edges = cv.Canny(th3, 50, 250)
        edgesOriginal = cv.Canny(img, 150, 300)
    elif type == 'clean':
        edges = cv.Canny(img, 5, 10)
    else:
        raise Exception("type should be either 'clean' or 'dirty'")
    
    #bw, _ = cv.threshold(edges, 200, 255, cv.THRESH_BINARY)
    #bw = cv.cvtColor(edges, )
    if display:
        plt.imshow(th3)
        plt.show()

    file_name = f'images/canny/{type}/{id}.jpg'
    cv.imwrite(file_name, edges)

    return file_name



if __name__ == "__main__":
    apply_canny('./images/neu.png', 'dirty', 2)