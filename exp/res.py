import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# 生成随机数据和标签
num_classes = 10
images = np.random.randint(0, 256, size=(60000, 224, 224, 3), dtype=np.uint8)
labels = np.random.randint(0, num_classes, size=(60000, 1), dtype=np.int32)
labels = tf.keras.utils.to_categorical(labels, num_classes)

# 加载 ResNet50 模型，不包括顶部的全连接层
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

# 添加自定义层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 构建并编译模型
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 设置 TensorBoard 日志目录
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置回调
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
progbar_logger = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)

# 训练模型
model.fit(images, labels, batch_size=1, epochs=1, callbacks=[tensorboard_callback, progbar_logger])