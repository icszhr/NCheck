import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 标准化数据
train_images = train_images / 255.0
test_images = test_images / 255.0

# 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5)

# 评估模型
model.evaluate(test_images, test_labels)

