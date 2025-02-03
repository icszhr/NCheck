import tensorflow as tf
from tensorflow.keras.applications import VGG19
import numpy as np
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
#tf.config.experimental.set_visible_devices([], 'GPU')

# 创建虚拟数据集
num_samples = 1000
input_shape = (224, 224, 3)

x_train = np.random.randn(num_samples, *input_shape).astype(np.float32)
y_train = np.random.randint(0, 1000, size=num_samples)

# 构建VGG19模型
base_model = VGG19(weights=None, input_shape=input_shape, include_top=True, classes=1000)

# 编译模型
base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
batch_size = 32
epochs = 10

start_time = time.time()
base_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
end_time = time.time()

print(f"Training took {end_time - start_time:.2f} seconds")

# 性能评估
num_samples_eval = 500
x_eval = np.random.randn(num_samples_eval, *input_shape).astype(np.float32)

start_time = time.time()
predictions = base_model.predict(x_eval)
end_time = time.time()

print(f"Inference on {num_samples_eval} samples took {end_time - start_time:.2f} seconds")

