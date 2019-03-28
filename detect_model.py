#!/usr/bin/env python
# coding: utf-8

# **Import các thư viện cần thiết**

# In[1]:


import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.utils.generic_utils import CustomObjectScope


# **Load mô hình**

# In[3]:


with CustomObjectScope({'relu6': keras.layers.ReLU,'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = load_model('model_save.h5')


# **Xử lý ảnh giống quá trình train**

# In[4]:


def prepare_image(img):
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# **Hàm dự đoán ảnh**

# In[6]:


def predict(img):
    """
    Predict face crop from frame
    :param img:
    :return: If boss is appear when open the code IDE
    """
    try:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        probs = model.predict(prepare_image(img))
        is_boss = np.argmax(probs[0])
        return is_boss
    except:
        return False

