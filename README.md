# Face_Recognition_FaceNet
Face Recognition System using FaceNet

Face Recognition:
Face recognition is the general task of identifying and verifying people from photographs of their face.

Face Verification:
A one-to-one mapping of a given face against a known identity (e.g. is this the person?)

Face Identification:
A one-to-many mapping for a given face against a database of known faces (e.g. who is this person?)
In this project we will focus on the face identification task.

FaceNet:
FaceNet is a face recognition system that was described by Florian Schroff, et al. at Google in their 2015 paper titled “FaceNet: A Unified Embedding for Face Recognition and Clustering.”

Face Embeddings:
It is a system that, given a picture of a face, will extract high-quality features from the face and predict a 128 element vector representation these features, called a face embedding.


Pre-trained Keras FaceNet model:
In this project we will use the pre-trained Keras FaceNet model provided by Hiroki Taniai. It was trained on MS-Celeb-1M dataset and expects input images to be color, to have their pixel values whitened (standardized across all three channels), and to have a square shape of 160×160 pixels.

https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn?usp=drive_open

Detect Faces for Face Recognition using MTCNN:
we will also use the Multi-Task Cascaded Convolutional Neural Network, or MTCNN, for face detection, e.g. finding and extracting faces from photos.

To install mtcnn:
sudo pip install mtcnn

Dataset:
We are going to use our small dataset with the photos of Rajin and AbdulKalam.

Detecting Faces:
The first step is to detect the face in each photograph and reduce the dataset to a series of faces only.

https://github.com/KarthikBalakrishnan11/Face_Recognition_FaceNet/blob/master/face_detection.py

Create Face Embeddings:
The next step is to create a face embedding. 

A face embedding is a vector that represents the features extracted from the face. This can then be compared with the vectors generated for other faces. For example, another vector that is close (by some measure) may be the same person, whereas another vector that is far (by some measure) may be a different person.

https://github.com/KarthikBalakrishnan11/Face_Recognition_FaceNet/blob/master/face_embeddings.py

Perform Face Classification:
Now we need to develop a model to classify face embeddings.

https://github.com/KarthikBalakrishnan11/Face_Recognition_FaceNet/blob/master/face_classification.py

Prediction for a given unseen photo:

https://github.com/KarthikBalakrishnan11/Face_Recognition_FaceNet/blob/master/face_system.py

From this project we learned:
About the FaceNet face recognition system.
Open source implementations and pre-trained models of FaceNet.
Extracting faces via a face detection system.
Extracting face features via face embeddings.
Fit, evaluate, and demonstrate an SVM model to predict identities from faces embeddings.

Useful links:
For Detailed Blog: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
FaceNet Paper: https://arxiv.org/abs/1503.03832
Keras OpenFace: https://github.com/iwantooxxoox/Keras-OpenFace
FaceNet by David Sandberg: https://github.com/davidsandberg/facenet
FaceNet by Hiroki Taniai: https://github.com/nyoki-mtl/keras-facenet
MS-Celeb-1M dataset: https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/
Keras FaceNet Pre-Trained Model: https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn
MTCNN: https://arxiv.org/abs/1604.02878
Linear Support Vector Machine (SVM): https://machinelearningmastery.com/support-vector-machines-for-machine-learning/
5 Celebrity Faces Dataset, Kaggle: https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset
savez_compressed() function: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html
load() NumPy function: https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html
Vector normalization: https://machinelearningmastery.com/vector-norms-machine-learning/
 Normalizer class in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
LabelEncoder class in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
SVC class in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
