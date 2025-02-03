import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
import os
import time

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 定义生成随机图像数据的函数
def generate_random_image(height=224, width=224, channels=3):
    return np.random.rand(height, width, channels).astype(np.float32)

def generate_training_data(batch_size=32, height=224, width=224, channels=3):
    images = [generate_random_image(height, width, channels) for _ in range(batch_size)]
    labels = np.random.randint(0, 1000, size=(batch_size,))
    return np.array(images), labels

# 设置超参数
batch_size = 1
num_samples = 1000

# 生成训练数据
train_images, train_labels = generate_training_data(batch_size=batch_size)

# 转换为 TensorFlow 张量
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)

# 构建 ResNet50 模型
model = ResNet50(weights=None, input_shape=(224, 224, 3), classes=1000)

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 定义训练步骤
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 训练模型一个迭代
for inputs, labels in train_dataset.take(2):  # 只进行一个 iteration
    # 前向传播
    forward_pass_start_time = time.time()
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
    forward_pass_end_time = time.time()

    # 反向传播
    backward_pass_start_time = time.time()
    gradients = tape.gradient(loss, model.trainable_variables)
    backward_pass_end_time = time.time()

    # 权重更新
    weight_update_start_time = time.time()
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    weight_update_end_time = time.time()

    # 检查点从GPU snapshot到CPU
    snapshot_to_cpu_start_time = time.time()
    model_weights_cpu = [var.numpy() for var in model.trainable_variables]
    snapshot_to_cpu_end_time = time.time()

    # 模拟将模型从 CPU 序列化到内存
    serialization_start_time = time.time()
    serialized_data = [tf.io.serialize_tensor(var) for var in model_weights_cpu]
    serialization_end_time = time.time()

    # 计算总的检查点保存时间
    checkpoint_save_start_time = time.time()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save('/tmp/resnet_checkpoint')
    checkpoint_save_end_time = time.time()

    # 计算写入文件系统和保存到SSD的时间
    total_write_and_ssd_copy_time = checkpoint_save_end_time - checkpoint_save_start_time
    write_and_ssd_copy_time = total_write_and_ssd_copy_time - (snapshot_to_cpu_end_time - snapshot_to_cpu_start_time) - (serialization_end_time - serialization_start_time)

# 计算时间
forward_pass_time = forward_pass_end_time - forward_pass_start_time
backward_pass_time = backward_pass_end_time - backward_pass_start_time
weight_update_time = weight_update_end_time - weight_update_start_time
snapshot_to_cpu_time = snapshot_to_cpu_end_time - snapshot_to_cpu_start_time
serialization_time = serialization_end_time - serialization_start_time

# 打印时间
print(f"Time taken for forward pass: {forward_pass_time:.2f} seconds")
print(f"Time taken for backward pass: {backward_pass_time:.2f} seconds")
print(f"Time taken for weight update: {weight_update_time:.2f} seconds")
print(f"Time taken for snapshot to CPU: {snapshot_to_cpu_time:.2f} seconds")
print(f"Time taken for serialization: {serialization_time:.2f} seconds")
print(f"Time taken for write and SSD copy: {write_and_ssd_copy_time:.2f} seconds")

# 恢复检查点性能分析
recovery_start_time = time.time()
new_checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
new_checkpoint.restore(tf.train.latest_checkpoint('/tmp'))
recovery_end_time = time.time()

# 从SSD复制到DRAM
copy_to_dram_start_time = time.time()
restored_model_weights_cpu = [var.numpy() for var in model.trainable_variables]
copy_to_dram_end_time = time.time()

# 反序列化
deserialization_start_time = time.time()
restored_model_weights = [tf.io.parse_tensor(var, out_type=tf.float32) for var in serialized_data]
deserialization_end_time = time.time()

# 计算恢复时间
recovery_time = recovery_end_time - recovery_start_time
copy_to_dram_time = copy_to_dram_end_time - copy_to_dram_start_time
deserialization_time = deserialization_end_time - deserialization_start_time

# 打印恢复时间
print(f"Time taken for recovery from SSD: {recovery_time:.2f} seconds")
print(f"Time taken for copy to DRAM: {copy_to_dram_time:.2f} seconds")
print(f"Time taken for deserialization: {deserialization_time:.2f} seconds")

# 保存结果到文件
total_times = {
    "forward_pass_time": forward_pass_time,
    "backward_pass_time": backward_pass_time,
    "weight_update_time": weight_update_time,
    "snapshot_to_cpu_time": snapshot_to_cpu_time,
    "serialization_time": serialization_time,
    "write_and_ssd_copy_time": write_and_ssd_copy_time,
    "recovery_time": recovery_time,
    "copy_to_dram_time": copy_to_dram_time,
    "deserialization_time": deserialization_time
}

with open("ResNet50_performance.txt", "w") as f:
    for key, value in total_times.items():
        f.write(f"{key}: {value:.2f} seconds\n")

print("Benchmark results saved to ResNet50_performance.txt")
