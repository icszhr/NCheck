import tensorflow as tf
import time
import os
import numpy as np

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.get_logger().setLevel('ERROR')

# 定义生成随机图像数据的函数
def generate_random_image_data(num_samples, img_height, img_width, num_classes):
    images = np.random.rand(num_samples, img_height, img_width, 3).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)
    return images, labels

def benchmark_resnet50(batch_size=1, iterations_per_epoch=10):
    print("Running benchmark for ResNet50...")
    iteration_times = []

    img_height = 224
    img_width = 224
    num_classes = 1000
    num_samples = 1000

    # 生成训练数据
    train_images, train_labels = generate_random_image_data(num_samples, img_height, img_width, num_classes)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)

    # 构建 ResNet50 模型
    model = tf.keras.applications.ResNet50(
        input_shape=(img_height, img_width, 3),
        include_top=True,
        weights=None,
        classes=num_classes
    )

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    start_time = time.time()
    print(f"Starting training for 1 epoch with {iterations_per_epoch} iterations")

    for iteration in range(iterations_per_epoch):
        # 获取一个批次的训练数据
        for images, labels in train_dataset.take(1):
            if iteration == 0:
                continue  # 跳过第一个 iteration

            # 记录整个 iteration 的开始时间
            iteration_start_time = time.time()

            # 前向传播和计算损失
            forward_pass_start_time = time.time()
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            forward_pass_end_time = time.time()

            # 反向传播
            backward_pass_start_time = forward_pass_end_time
            gradients = tape.gradient(loss, model.trainable_variables)
            backward_pass_end_time = time.time()

            # 权重更新
            weight_update_start_time = backward_pass_end_time
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            weight_update_end_time = time.time()

            iteration_end_time = time.time()

            # 计算时间
            forward_pass_time = forward_pass_end_time - forward_pass_start_time
            backward_pass_time = backward_pass_end_time - backward_pass_start_time
            weight_update_time = weight_update_end_time - weight_update_start_time
            iteration_time = iteration_end_time - iteration_start_time

            print(f"Iteration {iteration + 1} - Time taken for forward pass: {forward_pass_time:.2f} seconds")
            print(f"Iteration {iteration + 1} - Time taken for backward pass: {backward_pass_time:.2f} seconds")
            print(f"Iteration {iteration + 1} - Time taken for weight update: {weight_update_time:.2f} seconds")
            print(f"Iteration {iteration + 1} - Total time taken for iteration: {iteration_time:.2f} seconds")

            iteration_times.append(iteration_time)

    end_time = time.time()

    # 计算总的训练时间
    training_time = end_time - start_time
    print(f"Total training time for 1 epoch with ResNet50 (excluding first iteration):", sum(iteration_times))

    return sum(iteration_times)

# 运行基准测试并保存结果
total_time = benchmark_resnet50()
with open("ResNet50_performance.txt", "w") as f:
    f.write(f"Total Time (excluding first iteration): {total_time:.2f} seconds\n")

print("Benchmark results saved to ResNet50_performance.txt")
