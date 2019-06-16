from os import listdir
from os.path import isdir
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # reading image
    image = cv2.imread(filename)
    # convert to RGB since cv2 read as BGR
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # using mtcnn
    detector = MTCNN()
    # using detect_faces fucntion - it will return box, confidence and keypoints
    results = detector.detect_faces(image)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix - altering negative values to positive
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face using region
    face = image[y1:y2, x1:x2]
    # face size should be 160*160 as per FaceNet model
    face_array = cv2.resize(face,(required_size))
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# load train dataset
trainX, trainy = load_dataset('dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('dataset/val/')
# save arrays to one file in compressed format
savez_compressed('face_detection.npz', trainX, trainy, testX, testy)