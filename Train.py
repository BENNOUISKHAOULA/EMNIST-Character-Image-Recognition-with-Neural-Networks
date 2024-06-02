# Importer les bibliothèques
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2
from keras.models import Sequential 
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense,Dropout,Flatten 
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import to_categorical
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

# Charger les données d'entraînement et de test depuis les fichiers CSV
train=pd.read_csv("emnist-balanced-train.csv",delimiter=',')
test=pd.read_csv("emnist-balanced-test.csv",delimiter=',')
mapp = pd.read_csv("emnist-balanced-mapping.txt", delimiter=' ', index_col=0, header=None)

# definition du Fonction de rotation et Supprimé les variables non utilisables
#print("Train:%s,Test:%s,Map:%s"   %(train.shape,test.shape,mapp.shape))
print("Train:%s,,Map:%s"   %(train.shape,mapp.shape))

# Extraire les labels et les images d'entraînement et de test
HEIGHT =28
WIDTH =28
train_x=train.iloc[:,1:]
train_y=train.iloc[:,0]
del train
test_x=test.iloc[:,1:]
test_y=test.iloc[:,0]
del test
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
print(train_x.shape,train_y.shape)

#definir la fonction rotate
def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

#Flip and rotate image
train_x=np.asarray(train_x)
train_x=np.apply_along_axis(rotate,1,train_x)
print("train_x:",train_x.shape)
test_x=np.asarray(test_x)
test_x=np.apply_along_axis(rotate,1,test_x)
#print("test_x",test_x.shape)

#Normalisation
train_x=train_x.astype('float32')
train_x  /=255
test_x=test_x.astype('float32')
test_x /=255
#plot image
for i in range(100,109):
    plt.subplot(330+(i +1))
    plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))
    plt.title(chr(mapp.iloc[train_y.iloc[i]].values[0]))


# Préparer nos images pour le modéle (suite) ....

#number of classes 
num_classes =train_y.nunique()
    
   
print(train_y.shape)
#one hot encoding
train_y=to_categorical(train_y, num_classes)
test_y=to_categorical(test_y,num_classes)
print("train_y",train_y.shape)
print("test_y",test_y.shape)
#reshape image for cnn
train_x=train_x.reshape(-1,HEIGHT,WIDTH,1)
test_x=test_x.reshape(-1,HEIGHT,WIDTH,1)
#partition to train and val
print("train_y",train_y.shape)
train_x,val_x,train_y,val_y=train_test_split(train_x,train_y,test_size=0.1,random_state=7)
print("train_y",train_y.shape)
print("val_y",val_y.shape)

# Creation du modèle
EPOCHS=20
BATCH_SIZE=128
model=Sequential()
model.add(Conv2D(filters=128,kernel_size=(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=47,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
#model.compile(loss='mean_squared_error')//Adam(Adaptative Moment Estimation) default value 0.001 SGD:stochastic gradient descent
history=model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,validation_data=(val_x,val_y))

num_iteration =(len(train_x) /8)*EPOCHS
print(num_iteration)
model.evaluate(test_x,test_y)

def plotgraph(epochs,acc,val_acc, name ):
    #plot training &validation accuracy values
    plt.plot(epochs,acc,'b')
    plt.plot(epochs,val_acc,'r')
    plt.title('Model '+name)
    plt.ylabel(name)
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'],loc='upper left')
    plt.show()

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
model.save('ModelPresentation.h5')
print("Sauvegard le model sous nom ModelPresentation.h5")

# la courbe de précision d'apprentissage
#accuracy curve
plotgraph(epochs,acc,val_acc,'Accuracy')

#loss curve
plotgraph(epochs,loss, val_loss,'loss')
score=model.evaluate(test_x,test_y,verbose=0)
print("Test loss:",score[0])
print("Test accuracy:",score[1])
y_pred=model.predict(test_x)
y_pred=(y_pred>0.5)
cm=metrics.confusion_matrix(test_y.argmax(axis=1),y_pred.argmax(axis=1))

print (val_x.shape)
print(val_y.shape)

