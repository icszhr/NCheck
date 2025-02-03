import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetLMHeadModel, BertTokenizer, TFBertModel, RobertaTokenizer, TFRobertaModel, T5Tokenizer, TFT5ForConditionalGeneration, DistilBertTokenizer, TFDistilBertModel
import time
import random
import string
import os
import numpy as np

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
        self.checkpoint_times = []
        self.total_checkpoint_time = 0

    def on_epoch_end(self, epoch, logs=None):
        if not self.save_checkpoints:
            return
        
        self.save_checkpoint(epoch)

    def save_checkpoint(self, iteration):
        # 开始记录checkpoint保存时间
        total_checkpoint_time_start = time.time()

        # Step 1: Copy weights from GPU to CPU
        gpu_to_cpu_start = time.time()
        cpu_weights = {var.name: var.value().numpy() for var in self.model.variables}
        gpu_to_cpu_time = time.time() - gpu_to_cpu_start
        print(f"Iteration {iteration + 1} - Time to transfer weights from GPU to CPU: {gpu_to_cpu_time:.2f} seconds")

        # Step 2: Save weights from CPU to disk
        persist_start = time.time()
        checkpoint_name = f"{self.checkpoint_dir}/iteration_{iteration + 1}.ckpt"
        # Manually saving the weights to simulate disk write
        np.savez(checkpoint_name, **cpu_weights)
        persist_time = time.time() - persist_start
        print(f"Iteration {iteration + 1} - Time to persist weights from CPU to disk: {persist_time:.2f} seconds")

        # Total checkpoint time
        total_checkpoint_time = time.time() - total_checkpoint_time_start
        self.total_checkpoint_time += total_checkpoint_time
        print(f"Iteration {iteration + 1} - Total Time taken to save checkpoint: {total_checkpoint_time:.2f} seconds")

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
    ("ResNet50", None, tf.keras.applications.ResNet50, None),
    ("GPT-2", XLNetTokenizer, TFXLNetLMHeadModel, "xlnet-base-cased"),
    ("BERT", BertTokenizer, TFBertModel, "bert-base-uncased"),
    ("RoBERTa", RobertaTokenizer, TFRobertaModel, "roberta-base"),
    ("T5", T5Tokenizer, TFT5ForConditionalGeneration, "t5-small"),
    ("DistilBERT", DistilBertTokenizer, TFDistilBertModel, "distilbert-base-uncased")
]

def benchmark_model(model_name, tokenizer_class, model_class, pretrained_model_name, batch_size=1, total_iterations=21):
    print(f"Running benchmark for {model_name}...")
    iteration_times = []
    checkpoint_callback = None

    if model_name == "ResNet50":
        # ResNet50 特殊处理
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

        # 定义训练步骤
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, gradients

        start_time = time.time()
        print(f"Starting training for 1 epoch with {total_iterations} iterations")

        for iteration in range(total_iterations):
            # 获取一个批次的训练数据
            for images, labels in train_dataset.take(1):  # 只进行一个 iteration
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

                # 执行 checkpoint 每10个迭代执行一次
                if iteration % 10 == 0:
                    if checkpoint_callback is None:
                        checkpoint_callback = CheckpointAndTimingCallback(model, checkpoint_dir, save_checkpoints=True)
                    checkpoint_callback.save_checkpoint(iteration)

        end_time = time.time()

        # 计算总的训练时间
        training_time = end_time - start_time
        print(f"Total training time for 1 epoch with {model_name} (excluding first iteration):", sum(iteration_times))

    else:
        # 初始化 tokenizer 和 model
        tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
        model = model_class.from_pretrained(pretrained_model_name)

        # 添加 padding token 如果没有的话
        if model_name in ["BERT", "DistilBERT"]:
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.eos_token

        model, optimizer, loss_fn = compile_model(model)

        # 训练模型，执行多个 iteration
        start_time = time.time()
        print(f"Starting training for 1 epoch with {total_iterations} iterations")

        for iteration in range(total_iterations):
            # 生成一个批次的训练数据
            train_inputs, train_labels = generate_training_data(tokenizer, batch_size=batch_size)
            train_inputs = {key: tf.constant(value) for key, value in train_inputs.items()}
            train_labels = tf.constant(train_labels)

            if iteration == 0:
                continue  # 跳过第一个 iteration

            # 记录整个 iteration 的开始时间
            iteration_start_time = time.time()

            # 前向传播和计算损失
            forward_pass_start_time = time.time()
            with tf.GradientTape() as tape:
                if model_name == "T5":
                    # T5 特殊处理
                    decoder_input_ids = tf.roll(train_labels, shift=1, axis=1)
                    decoder_input_ids = tf.where(decoder_input_ids == 0, tf.fill(tf.shape(decoder_input_ids), -100), decoder_input_ids)
                    outputs = model(train_inputs, decoder_input_ids=decoder_input_ids, training=True)
                    logits = outputs.logits
                elif model_name in ["BERT", "DistilBERT", "RoBERTa"]:
                    outputs = model(train_inputs, training=True)
                    logits = outputs.last_hidden_state
                else:
                    outputs = model(train_inputs, training=True)
                    logits = outputs.logits

                if model_name == "T5":
                    loss_value = loss_fn(train_labels[:, 1:], logits[:, :-1])
                else:
                    loss_value = loss_fn(train_labels, logits)

            forward_pass_end_time = time.time()

            # 反向传播
            backward_pass_start_time = time.time()
            gradients = tape.gradient(loss_value, model.trainable_variables)
            backward_pass_end_time = time.time()

            # 权重更新
            weight_update_start_time = time.time()
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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

            # 执行 checkpoint 每10个迭代执行一次
            if iteration % 10 == 0:
                if checkpoint_callback is None:
                    checkpoint_callback = CheckpointAndTimingCallback(model, checkpoint_dir, save_checkpoints=True)
                checkpoint_callback.save_checkpoint(iteration)

        end_time = time.time()

        # 计算总的训练时间
        training_time = end_time - start_time
        print(f"Total training time for 1 epoch with {model_name} (excluding first iteration):", sum(iteration_times))

    return sum(iteration_times) * 5, checkpoint_callback.total_checkpoint_time * 5

# 运行基准测试并汇总结果
results = []

for model_config in models_config:
    model_name, tokenizer_class, model_class, pretrained_model_name = model_config
    total_time, total_checkpoint_time = benchmark_model(model_name, tokenizer_class, model_class, pretrained_model_name)
    results.append({
        "model_name": model_name,
        "total_time": total_time,
        "total_checkpoint_time": total_checkpoint_time
    })

# 输出结果到文件
with open("result1.txt", "w") as f:
    for result in results:
        f.write(f"Model: {result['model_name']}\n")
        f.write(f"Total Time (excluding first iteration): {result['total_time']:.2f} seconds\n")
        f.write(f"Total Checkpoint Time: {result['total_checkpoint_time']:.2f} seconds\n")
        f.write("\n")

print("Benchmark results saved to result1.txt")
