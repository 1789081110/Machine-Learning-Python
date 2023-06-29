#!/usr/bin/env python
# coding: utf-8

# # Name: Vishnu Kumar S.R

# # USN:20BTRCL030

# In[32]:


from keras.preprocessing.image import ImageDataGenerator


# In[86]:


train=('C:/Users/vknsr/Downloads/training')
test=('C:/Users/vknsr/Downloads/testing')


# In[87]:


inimg=ImageDataGenerator(rescale=(1/255),rotation_range=0.2,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2,horizontal_flip=True,vertical_flip=True)


# In[88]:


timg=ImageDataGenerator(rescale=(1/255))


# In[89]:


train_images=inimg.flow_from_directory(directory=train,target_size=(200,200),class_mode='binary',batch_size=5)


# In[90]:


test_images=timg.flow_from_directory(directory=test,target_size=(200,200),class_mode='binary')


# In[91]:


from tensorflow.keras.models import Sequential 


# In[92]:


model=Sequential()


# In[93]:


from tensorflow.keras.layers import Conv2D,MaxPooling2D, Flatten,Dense


# In[94]:


model.add(Conv2D(32,(3,3),activation='elu',kernel_initializer='he_uniform',input_shape=(200,200,3)))


# In[95]:


model.add(MaxPooling2D(2,2))


# In[96]:


model.add(Conv2D(16,(3,3),activation='elu',kernel_initializer='he_uniform'))


# In[97]:


model.add(Flatten())


# In[98]:


model.add(Dense(32,activation='elu',kernel_initializer='he_uniform'))


# In[99]:


model.add(Dense(1,activation='sigmoid'))


# In[100]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[101]:


cnn=model.fit_generator(train_images,epochs=5,validation_data=test_images)


# In[102]:


import matplotlib.pyplot as plt


# In[103]:


plt.plot(cnn.history['accuracy'],label='accuracy')
plt.plot(cnn.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()


# In[104]:


plt.plot(cnn.history['loss'],label='accuracy')
plt.plot(cnn.history['val_loss'],label='val_accuracy')
plt.legend()
plt.show()


# In[109]:


path='C:/Users/vknsr/Downloads/archive (1)/train'
import matplotlib.pyplot as plt


# In[110]:


import os
import numpy as np
for i in os.listdir(path):
    print(i)
    img=image.load_img(path+"//"+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)#3d we add 1 dim=4dim
    images=np.vstack([x])
    val=model.predict(images)
    print(val)
    if(val>=0.5):
        print('cat')
    else:
        print('dog')


# In[5]:


import tensorflow.keras.utils as image


# In[41]:


from tensorflow.keras.utils import load_img,array_to_img,img_to_array


# In[55]:


train1='C:/Users/vknsr/Downloads/archive (1)/train'


# In[3]:


test='C:/Users/vknsr/Downloads/archive (1)/test'


# In[28]:


from keras.models import Model


# In[ ]:




