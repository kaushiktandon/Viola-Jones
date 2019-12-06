import cv2

# from: https://www.datacamp.com/community/tutorials/face-detection-python-opencv
def detect_faces(cascade, test_image, scaleFactor=1.1):
  # create a copy of the image to prevent any changes to the original one.
  image_copy = test_image.copy()

  #convert the test image to gray scale as opencv face detector expects gray images
  gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

  # Applying the haar classifier to detect faces
  faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

  for (x, y, w, h) in faces_rect:
      cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)

  return image_copy

def main():
  cascade_path = "train/classifier/cascade.xml"
  img_path     = "test_images/test_img.jpg"

  # read image as grayscale, generated OpenCV classifier
  test_image = cv2.imread(img_path)
  cascade    = cv2.CascadeClassifier(cascade_path)

  # return image with rectangles around detected faces
  face_img = detect_faces(cascade, test_image)
  cv2.imshow("", face_img)
  cv2.waitKey(10000)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()