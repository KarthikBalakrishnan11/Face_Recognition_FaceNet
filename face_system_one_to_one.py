from numpy import expand_dims
import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import numpy as np

model = load_model('facenet_keras.h5')
print("FaceNet Model Loaded")
emb=[]

for i in range(1,3):
    imageName="Test_Images/Faces/"+str(i)+".jpg"
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
        print("Face Not Detected")
        
    print("1. Face Detected form image")
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = image[y1:y2, x1:x2]
    print("2. Face extracted")
    #resized for FaceNet model
    face = cv2.resize(face,(160,160))
    face = face.astype('float32')
    
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    
    face = expand_dims(face, axis=0)
    
    #Face embeddings collected
    embed = model.predict(face)
    emb.append(embed)
    #print("Embeds",emb)
    #print("3. Face Embeddings Collected")    
    #comparing the embeddings

output=np.sum(np.square(emb[0] - emb[1]))
print("Output",output)

src = cv2.imread("Test_Images/Faces/1.jpg")
cv2.putText(src,"Source Image",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
cv2.imshow("Source Image",src)
cv2.waitKey(0)

if(output<150):
        #print("Name:",names[class_index])
        cv2.putText(image,"Face Matched",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Output",image)
        cv2.waitKey(0)
else:
     cv2.putText(image,"Not Matching",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
     cv2.imshow("Output",image)
     cv2.waitKey(0)
    
cv2.destroyAllWindows()