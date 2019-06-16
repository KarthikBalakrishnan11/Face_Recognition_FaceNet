# develop a classifier for Faces Dataset
from numpy import expand_dims
import pickle
import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

names=["Dr. A.P.J.Abdul Kalam","Karthik","Rajinikanth"]
model = load_model('facenet_keras.h5')
print("FaceNet Model Loaded")

for i in range(1,10):
    imageName="Test_Images/test"+str(i)+".jpg"
    print (imageName)
    image = cv2.imread(imageName)
    #Resizing image to fit window
    scale_percent = 80 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
    #using mtcnn for face detection
    detector = MTCNN()
    #using detect faces function to retrive box, confidence and landmarks of faces
    results = detector.detect_faces(image)
    #if face not detected just skip the image
    if (results==[]):
        continue
    print("1. Face Detected form image")
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = image[y1:y2, x1:x2]
    print("2. Face extracted")
    #resized for FaceNet model
    face_pixels = cv2.resize(face,(160,160))
    face_pixels = face_pixels.astype('float32')
    
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    
    samples = expand_dims(face_pixels, axis=0)
    #Face embeddings collected
    yhat = model.predict(samples)
    print("3. Face Embeddings Collected")
    #Loading FaceEmbedding model file
    filename = 'finalized_model.sav'
    prediction_model = pickle.load(open(filename, 'rb'))
    
    #comparing the embeddings
    yhat_class = prediction_model.predict(yhat)
    #Retrieving the probability of the prediction
    yhat_prob = prediction_model.predict_proba(yhat)
    #print("4. Predicting class and probability done")
    
    class_index = yhat_class[0]
    #print("Index",class_index)
    class_probability = yhat_prob[0,class_index] * 100
    
    print('Prediction Probablity:%.3f' %(class_probability))
    #setting threshold based on probability
    if(class_probability>99.5):
        #print("Name:",names[class_index])
        cv2.putText(resized_image ,names[class_index],(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Output",resized_image)
        cv2.waitKey(0)
    else:
        #print("Person not matched")
        cv2.putText(resized_image,"unknown",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Output",resized_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()