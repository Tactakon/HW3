import cv2
import matplotlib.pyplot as plt


def objectDetectionFromScratch(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    
    dilate = cv2.dilate(edged, kernel, iterations=1)

   
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
   
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print("objects were found in this image.")
    plt.axis('off')
    plt.imshow(image_copy,cmap='gray', vmin=0, vmax=255)
    plt.show()
    
img1 = "Supf/der1.png"
img2 = "Supf/der2.png"
objectDetectionFromScratch(img1)
objectDetectionFromScratch(img2)