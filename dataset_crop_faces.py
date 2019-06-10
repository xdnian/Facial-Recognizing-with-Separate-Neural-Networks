import glob, os
from face_detect import *

os.chdir("./images/")

counter = 0

for filename in glob.glob("*"):
    impath = filename
    print(impath)
    image = cv2.imread(impath)

    faceRects = detect_faces(image)
    faces = crop_rects(image, faceRects)

    folder = '../dataset/'
    for f in faces:
        cv2.imwrite(folder+str(counter)+'.png', f)
        counter += 1