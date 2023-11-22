#!/usr/bin/env python
# coding: utf-8

# # Homework #9

# In[194]:


get_ipython().system(' wget https://github.com/alexeygrigorev/large-datasets/releases/download/wasps-bees/bees-wasps.h5')


# In[195]:


import numpy as np

import tensorflow as tf
from tensorflow import keras

tf.__version__


# In[196]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input


# In[197]:


model = keras.models.load_model('bees-wasps.h5', compile=False)


# In[198]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('bees-wasps.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# ## Question 1

# In[199]:


get_ipython().system('ls -lh')


# In[200]:


import tensorflow.lite as tflite


# In[201]:


interpreter = tflite.Interpreter(model_path='bees-wasps.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# ## Question 2

# In[202]:


output_index


# In[203]:


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


# In[204]:


url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'


# In[205]:


img = download_image(url)
#img


# In[206]:


target_size = (150, 150)
img = prepare_image(img, target_size=target_size)


# In[207]:


x = np.array(img, dtype='float32')
X = np.array([x])


# In[210]:


X = X/255.


# ## Question 3

# In[214]:


X[0, 0, 0, :]


# ## Question 4

# In[211]:


preds = model.predict(X)
preds


# In[ ]:




