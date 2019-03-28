{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Import các thư viện cần thiết**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import keras\n",
    "from keras.models import load_model\n",
    "from keras.preprocessing import image\n",
    "from keras.utils.generic_utils import CustomObjectScope"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Load mô hình**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\ProgramData\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Colocations handled automatically by placer.\n",
      "WARNING:tensorflow:From C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n",
      "WARNING:tensorflow:From C:\\ProgramData\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\ops\\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.cast instead.\n",
      "WARNING:tensorflow:From C:\\ProgramData\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\ops\\math_grad.py:102: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Deprecated in favor of operator or tf.math.divide.\n"
     ]
    }
   ],
   "source": [
    "with CustomObjectScope({'relu6': keras.layers.ReLU,'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):\n",
    "    model = load_model('model_save.h5')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Xử lý ảnh giống quá trình train**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def prepare_image(img):\n",
    "    img_array = image.img_to_array(img)\n",
    "    img_array_expanded_dims = np.expand_dims(img_array, axis=0)\n",
    "    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Hàm dự đoán ảnh**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(img):\n",
    "    \"\"\"\n",
    "    Predict face crop from frame\n",
    "    :param img:\n",
    "    :return: If boss is appear when open the code IDE\n",
    "    \"\"\"\n",
    "    try:\n",
    "        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)\n",
    "        probs = model.predict(prepare_image(img))\n",
    "        is_boss = np.argmax(probs[0])\n",
    "        return is_boss\n",
    "    except:\n",
    "        return False"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
