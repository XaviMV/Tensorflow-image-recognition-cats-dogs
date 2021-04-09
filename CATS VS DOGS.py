#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras


# # Carregar la data
# ### Es descarrega la data necessària per a fer el model

# ### Calculo la relació d'aspecte mitjana de totes les imatges, per només escollir les imatges que són com a màxim un 20% més estirades tant verticalment com horitzontalment, ja que al entrar a la xarxa neuronal totes les imatges han de tenir la mateixa relació d'aspecte i els mateixos píxels, i imatges molt distorsionades afectarien al rendiment

# In[2]:


DATADIR = r"C:\Users\Xavi\Desktop\cat-and-dog"
size_x = 0
size_y = 0
acumulador = 0
class_names = ["dogs", "cats"]


for category in class_names:  # etiquetes de "gat" i "gos" per entrar a les carpetes adequades
    path = os.path.join(DATADIR,category)  # direcció dels arxius
    
    for img in os.listdir(path):  # per cada imatge de gat i gos
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # converteixo a una array (com una llista)
        size_y += img_array.shape[0]
        size_x += img_array.shape[1]
        plt.imshow(img_array,cmap='gray')
        plt.show()
        acumulador += 1
        break


average_ratio_yx =  (size_y/acumulador)/(size_x/acumulador)
    
print('RATIO Y:X: ', average_ratio_yx)


# ### Calculo la mida que tindran totes les imatges

# In[3]:


SIZE_Y = 100
SIZE_X = int(SIZE_Y/average_ratio_yx)


# # Creo els grups d'imatges per entrenar i per posar a prova la xarxa neuronal

# ### Creo les imatges i les etiquetes per l'entrenament de la xarxa neuronal

# In[4]:


training_images = []
training_labels = []

def create_training_data():
    for category in class_names:  # per gats i gossos

        path = os.path.join(DATADIR,category)  # direcció dels arxius
        class_num = class_names.index(category)  # classifico 0 = gos i 1 = gat

        for img in os.listdir(path):  # per cada imatge de gat i gos
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # converteixo a array
                
                if (img_array.shape[0]/img_array.shape[1]) > average_ratio_yx*0.7 and (img_array.shape[0]/img_array.shape[1]) < average_ratio_yx*1.3:
                    new_array = cv2.resize(img_array, (SIZE_Y, SIZE_X))  # redimensiono les imatges
                    training_images.append(new_array)  # afegeixo la imatge redimensionada a el set d'imatges
                    training_labels.append(class_num)

            except Exception as e:  # per si hi ha algun arxiu corrupte o que no sigui una imatge
                pass

create_training_data()


# ### Canvio el format en el que està per un que tensorflow accepta i també cambio l'escala de grisos de un valor entre 0 i 255 a un valor entre 0 i 1

# In[5]:


training_labels = np.asarray(training_labels)
training_images = np.asarray(training_images)

training_images = training_images/255

print(training_images[0])
print(training_labels[0])
print(type(training_images))

plt.imshow(training_images[0], cmap='gray')
plt.show()


# ### Barrejo totes les imatges per tal de que sigui el més aleatori possible

# In[6]:


training_data = list(zip(training_images,training_labels))

random.shuffle(training_data)

training_images, training_labels = zip(*training_data)

training_labels = np.asarray(training_labels)
training_images = np.asarray(training_images)


# ### Divideixo totes les imatges en 90% per entrenar la xarxa neuronal i 10% reservades per posar-la a prova amb imatges que no ha vist mai

# In[7]:


all_images = training_images
all_labels = training_labels

training_labels = []
training_images = []

test_labels = []
test_images = []

acc = 0

for i in all_images:
    acc += 1
    if acc > len(all_images)*0.9:
        test_images.append(i)
    else:
        training_images.append(i)

acc = 0
        
for i in all_labels:
    acc += 1
    if acc > len(all_labels)*0.9:
        test_labels.append(i)
    else:
        training_labels.append(i)
           

training_labels = np.asarray(training_labels)
training_images = np.asarray(training_images)

test_labels = np.asarray(test_labels)
test_images = np.asarray(test_images)


# # Crear el model

# ### I create the neural network structure

# In[8]:


model = keras.Sequential([
    keras.layers.Conv2D(128, (10,10), activation = "relu", input_shape=(SIZE_X, SIZE_Y, 1), padding='same'),
    keras.layers.MaxPooling2D((3, 3)),
    keras.layers.Conv2D(64, (10,10), activation = "relu", padding='same'),
    keras.layers.MaxPooling2D((5, 5)),
    keras.layers.Conv2D(64, (5, 5), activation = "relu", padding='same'),

    keras.layers.Flatten(),
    keras.layers.Dense(32, activation = "relu"),
    keras.layers.Dense(32, activation = "relu"),
    keras.layers.Dense(2, activation = "softmax")])

model.compile(optimizer = 'sgd', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[9]:


model.summary()


# # Entrenar el model

# In[ ]:


training_images_copia = training_images
test_images_copia = test_images

training_images = np.expand_dims(training_images, -1)
test_images = np.expand_dims(test_images, -1)


# In[ ]:


model.fit(training_images, training_labels, batch_size = 2, epochs = 7, shuffle = True)
test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[ ]:


print('Test accuarcy:', test_acc)


# ### Miro quines són les 10 primeres imatges que la xarxa neuronal ha fallat

# In[ ]:


prediction = model.predict(test_images)


# In[ ]:


acumulador = 0
i = 0
while True:
    if acumulador >= 10:
        break
    if class_names[int(test_labels[i])] == class_names[np.argmax(prediction[i])]:
        pass
    else:
        print(class_names[int(test_labels[i])], '<-- CORRECTE')
        print(class_names[np.argmax(prediction[i])])
        plt.imshow(test_images_copia[i], cmap='gray')
        plt.show()
        acumulador += 1
    i += 1
        


# In[38]:


for i in range(3):
    plt.imshow(test_images[random.randint(1, len(test_images))], cmap='gray')
    plt.show()


# In[ ]:




