import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image

if __name__ == '__main__':
   model = keras.Sequential(name='Sequential')
   # 第一层需定义输入尺寸
   model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(10, activation='softmax'))