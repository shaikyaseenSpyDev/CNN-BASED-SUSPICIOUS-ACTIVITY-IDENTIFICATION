from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import numpy as np
import pickle
from django.core.files.storage import FileSystemStorage
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, LSTM
from keras.layers import Conv2D
from keras.models import Sequential, load_model, Model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


global uname, X, Y
global X_train, X_test, y_train, y_test
accuracy, precision, recall, fscore = [], [], [], []
labels = []
path = "Dataset"
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())


def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)


X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')
print(X.shape)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (1, 32, 32, 3)))
model.add(TimeDistributed(MaxPooling2D((4, 4))))
model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
model.add(TimeDistributed(MaxPooling2D((4, 4))))
model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
model.add(TimeDistributed(MaxPooling2D((1, 1))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(32))
model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = model.fit(X_train, y_train, batch_size = 32, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    model.load_weights("model/cnn_weights.hdf5")
predict = model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
calculateMetrics("CNN", predict, y_test1)

def DetectActivityAction(request):
    if request.method == 'POST':
        global model
        labels = ['Accident', 'Burglary', 'Fighting', 'Fire', 'Normal', 'Shooting']
        model = load_model("model/cnn_weights.hdf5")
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("ActivityApp/static/"+fname):
            os.remove("ActivityApp/static/"+fname)
        with open("ActivityApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        cap = cv2.VideoCapture("ActivityApp/static/"+fname)
        option = 0
        msg = "none"
        while True:
            ret, frame = cap.read()
            if ret == True:
                img = cv2.resize(frame, (32, 32))
                img = img.astype('float32')
                img = img/255
                data = []
                data.append(img)
                data = np.asarray(data)
                temp = []
                temp.append(data)
                data = np.asarray(temp)
                predict = model.predict(data)
                pred = np.argmax(predict)
                score = np.amax(predict)
                print(str(pred)+" "+str(score))
                frame = cv2.resize(frame, (500, 500)) 
                if score > 0.80 and option == 0:
                    option = 1
                    msg = labels[pred]+" "+str(score)
                    cv2.putText(frame, labels[pred]+" "+str(score), (100, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, msg, (100, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
                cv2.imshow('Video Output', frame)
                if cv2.waitKey(70) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        return render(request, 'DetectActivity.html', {})


def DetectActivity(request):
    if request.method == 'GET':
       return render(request, 'DetectActivity.html', {})

def TrainCNN(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th>'
        output+='<th>Recall</th><th>FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['CNN Algorithm']
        for i in range(len(algorithms)):
            output+='<td>'+algorithms[i]+'</td><td>'+str(accuracy[i])+'</td><td>'+str(precision[i])+'</td><td>'+str(recall[i])+'</td><td>'+str(fscore[i])+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)
        
def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})    

def AdminLoginAction(request):
    if request.method == 'POST':
        global userid
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == "admin" and password == "admin":
            context= {'data':'Welcome '+user}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid Login'}
            return render(request, 'AdminLogin.html', context)

