import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel, TFBertModel, BertTokenizer, AlbertTokenizer, TFAlbertModel
import time
import random
import string
import os
import numpy as np
import threading
import json

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 定义生成随机文本的函数
def generate_random_text(min_length=5, max_length=1000):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# 生成训练数据
def generate_training_data(tokenizer, min_length=5, max_length=10, batch_size=32):
    texts = [generate_random_text(min_length, max_length) for _ in range(batch_size)]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="tf")
    labels = inputs["input_ids"]
    return inputs, labels

# 编译模型
def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model, optimizer, loss_fn

# 创建保存 checkpoint 的目录
checkpoint_dir = "./checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 定义一个回调函数以定期保存模型并记录时间
class CheckpointAndTimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, checkpoint_dir, save_checkpoints=True):
        super().__init__()
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints
        self.total_checkpoint_blocking_time = 0
        self.snapshot_thread = None
        self.persist_thread = None
        self.snapshot_done = threading.Event()
        self.persist_done = threading.Event()

    def on_epoch_end(self, epoch, logs=None):
        if not self.save_checkpoints:
            return
        
        self.save_checkpoint(epoch)

    def save_checkpoint(self, iteration):
        self.snapshot_done.clear()
        self.persist_done.clear()

        # Step 1: Copy weights from GPU to CPU
        def snapshot_weights():
            gpu_to_cpu_start = time.time()
            cpu_weights = {var.name: var.value().numpy() for var in self.model.variables}
            gpu_to_cpu_time = time.time() - gpu_to_cpu_start
            print(f"Iteration {iteration + 1} - Time to transfer weights from GPU to CPU: {gpu_to_cpu_time:.2f} seconds")
            self.snapshot_done.set()
            return cpu_weights

        self.snapshot_thread = threading.Thread(target=snapshot_weights)
        self.snapshot_thread.start()

        # Step 2: Save weights from CPU to disk
        def persist_weights(cpu_weights):
            self.snapshot_done.wait()
            persist_start = time.time()
            checkpoint_name = f"{self.checkpoint_dir}/iteration_{iteration + 1}.ckpt"
            # Manually saving the weights to simulate disk write
            np.savez(checkpoint_name, **cpu_weights)
            persist_time = time.time() - persist_start
            print(f"Iteration {iteration + 1} - Time to persist weights from CPU to disk: {persist_time:.2f} seconds")
            self.persist_done.set()

        def save_weights():
            cpu_weights = snapshot_weights()
            self.persist_thread = threading.Thread(target=persist_weights, args=(cpu_weights,))
            self.persist_thread.start()

        # 执行 CPU 到 SSD 的持久化，并行权重更新
        if self.persist_thread and self.persist_thread.is_alive():
            self.persist_thread.join()
        save_weights()

# 定义一个回调函数以实时显示训练进度和计时
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        print(f"Epoch {epoch + 1} - loss: {logs['loss']:.4f}, Time taken: {epoch_time:.2f} seconds")

# 检查是否有以前的checkpoint，并计算恢复时间
def load_checkpoint(model, checkpoint_dir):
    load_start_time = time.time()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading checkpoint from:", latest_checkpoint)
        model.load_weights(latest_checkpoint)
        load_time = time.time() - load_start_time
        print(f"Time taken to load checkpoint: {load_time:.2f} seconds")
    else:
        print("No checkpoint found, starting from scratch.")

# 定义生成随机图像数据的函数
def generate_random_image_data(num_samples, img_height, img_width, num_classes):
    images = np.random.rand(num_samples, img_height, img_width, 3).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)
    return images, labels

# 模型配置
models_config = [
    ("ResNet50", None, tf.keras.applications.ResNet50, {"weights": None, "input_shape": (224, 224, 3), "include_top": True, "classes": 1000}),
    ("ResNet101", None, tf.keras.applications.ResNet101, {"weights": None, "input_shape": (224, 224, 3), "include_top": True, "classes": 1000}),
    ("ResNet152", None, tf.keras.applications.ResNet152, {"weights": None, "input_shape": (224, 224, 3), "include_top": True, "classes": 1000}),
    ("ResNet50V2", None, tf.keras.applications.ResNet50V2, {"weights": None, "input_shape": (224, 224, 3), "include_top": True, "classes": 1000}),
    ("albert-base-v2", AlbertTokenizer, TFAlbertModel, "albert-base-v2"),
    ("distilbert-base-uncased", DistilBertTokenizer, TFDistilBertModel, "distilbert-base-uncased"),
    ("bert-base-uncased", BertTokenizer, TFBertModel, "bert-base-uncased"),
    ("bert-large-uncased", BertTokenizer, TFBertModel, "bert-large-uncased")
]

def benchmark_model(model_name, tokenizer_class, model_class, model_params, batch_size=1, iterations_per_epoch=21, checkpoint_interval=10):
    print(f"Running benchmark for {model_name} with batch size {batch_size}...")

    iteration_times = []
    checkpoint_callback = CheckpointAndTimingCallback(None, checkpoint_dir, save_checkpoints=True)

    if model_name.startswith("ResNet"):
        img_height = 224
        img_width = 224
        num_classes = 1000
        num_samples = 1000

        train_images, train_labels = generate_random_image_data(num_samples, img_height, img_width, num_classes)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)

        model = model_class(**model_params)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        start_time = time.time()

        for iteration in range(iterations_per_epoch):
            iteration_start_time = time.time()

            for images, labels in train_dataset.take(1):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time

            print(f"Iteration {iteration + 1} - Batch size {batch_size} - Time taken: {iteration_time:.2f} seconds")
            iteration_times.append(iteration_time)

            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_callback.model = model
                checkpoint_callback.save_checkpoint(iteration)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Total training time for {model_name} with batch size {batch_size}: {training_time:.2f} seconds")

    else:
        tokenizer = tokenizer_class.from_pretrained(model_params)
        model = model_class.from_pretrained(model_params)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model, optimizer, loss_fn = compile_model(model)

        start_time = time.time()

        for iteration in range(iterations_per_epoch):
            iteration_start_time = time.time()

            train_inputs, train_labels = generate_training_data(tokenizer, batch_size=batch_size)
            train_inputs = {key: tf.constant(value) for key, value in train_inputs.items()}
            train_labels = tf.constant(train_labels)

            with tf.GradientTape() as tape:
                outputs = model(train_inputs, training=True)
                logits = outputs.last_hidden_state if "last_hidden_state" in outputs else outputs.logits
                loss_value = loss_fn(train_labels, logits)

            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time

            print(f"Iteration {iteration + 1} - Batch size {batch_size} - Time taken: {iteration_time:.2f} seconds")
            iteration_times.append(iteration_time)

            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_callback.model = model
                checkpoint_callback.save_checkpoint(iteration)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Total training time for {model_name} with batch size {batch_size}: {training_time:.2f} seconds")

    return sum(iteration_times), checkpoint_callback.total_checkpoint_time

# 运行基准测试并汇总结果
results = []

# 使用不同版本的 ResNet 模拟不同 batch size 的 ResNet50
for model_config in models_config[:4]:
    model_name, _, model_class, model_params = model_config
    total_time, total_checkpoint_time = benchmark_model(model_name, None, model_class, model_params, batch_size=1, checkpoint_interval=10)
    results.append({
        "original_model_name": model_name,
        "total_time": total_time,
        "total_checkpoint_time": total_checkpoint_time
    })

# 使用不同版本的 BERT 模拟不同 batch size 的 BERT
for model_config in models_config[4:]:
    model_name, tokenizer_class, model_class, model_params = model_config
    total_time, total_checkpoint_time = benchmark_model(model_name, tokenizer_class, model_class, model_params, batch_size=1, checkpoint_interval=10)
    results.append({
        "original_model_name": model_name,
        "total_time": total_time,
        "total_checkpoint_time": total_checkpoint_time
    })

# 输出结果到JSON文件
with open("result3_2.json", "w") as f:
    json.dump(results, f, indent=4)

print("Benchmark results saved to result3_2.json")
