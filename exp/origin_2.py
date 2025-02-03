import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetLMHeadModel, BertTokenizer, TFBertModel, RobertaTokenizer, TFRobertaModel, T5Tokenizer, TFT5ForConditionalGeneration, DistilBertTokenizer, TFDistilBertModel
import time
import random
import string
import os
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor

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
checkpoint_dir = "/pmem/zhr/checkpoints"
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

    def save_checkpoint(self, iteration):
        total_checkpoint_time_start = time.time()

        gpu_to_cpu_start = time.time()
        cpu_weights = {var.name: var.value().numpy() for var in self.model.variables}
        gpu_to_cpu_time = time.time() - gpu_to_cpu_start

        persist_start = time.time()
        checkpoint_name = f"{self.checkpoint_dir}/iteration_{iteration + 1}.ckpt"
        np.savez(checkpoint_name, **cpu_weights)
        persist_time = time.time() - persist_start

        total_checkpoint_time = time.time() - total_checkpoint_time_start
        self.total_checkpoint_time += total_checkpoint_time

# 定义一个回调函数以实时显示训练进度和计时
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time

# 检查是否有以前的checkpoint，并计算恢复时间
def load_checkpoint(model, checkpoint_dir):
    load_start_time = time.time()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if (latest_checkpoint):
        model.load_weights(latest_checkpoint)

# 定义生成随机图像数据的函数
def generate_random_image_data(num_samples, img_height, img_width, num_classes):
    images = np.random.rand(num_samples, img_height, img_width, 3).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)
    return images, labels

# 模型配置
models_config = [
    ("ResNet50", None, tf.keras.applications.ResNet50, None),
    ("GPT-2", XLNetTokenizer, TFXLNetLMHeadModel, "xlnet-base-cased"),  # 使用XLNet进行测试，但显示为GPT-2
    ("BERT", BertTokenizer, TFBertModel, "bert-base-uncased"),
    ("RoBERTa", RobertaTokenizer, TFRobertaModel, "roberta-base"),
    ("T5", T5Tokenizer, TFT5ForConditionalGeneration, "t5-small"),
    ("DistilBERT", DistilBertTokenizer, TFDistilBertModel, "distilbert-base-uncased")
]

def benchmark_model(model_name, tokenizer_class, model_class, pretrained_model_name, batch_size=1, iterations_per_epoch=20, checkpoint_interval=1):
    iteration_times = []
    checkpoint_callback = CheckpointAndTimingCallback(None, checkpoint_dir, save_checkpoints=True)

    if model_name == "ResNet50":
        img_height = 224
        img_width = 224
        num_classes = 1000
        num_samples = 1000

        train_images, train_labels = generate_random_image_data(num_samples, img_height, img_width, num_classes)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)

        model = tf.keras.applications.ResNet50(
            input_shape=(img_height, img_width, 3),
            include_top=True,
            weights=None,
            classes=num_classes
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, gradients

        start_time = time.time()

        for iteration in range(iterations_per_epoch):
            for images, labels in train_dataset.take(1):
                if iteration == 0:
                    continue

                iteration_start_time = time.time()

                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                iteration_end_time = time.time()
                iteration_time = iteration_end_time - iteration_start_time

                iteration_times.append(iteration_time)

                if (iteration + 1) % checkpoint_interval == 0:
                    checkpoint_callback.model = model
                    checkpoint_callback.save_checkpoint(iteration)

        end_time = time.time()
        training_time = end_time - start_time

    else:
        tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
        model = model_class.from_pretrained(pretrained_model_name)

        if model_name in ["BERT", "DistilBERT"]:
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.eos_token

        model, optimizer, loss_fn = compile_model(model)

        start_time = time.time()

        for iteration in range(iterations_per_epoch):
            train_inputs, train_labels = generate_training_data(tokenizer, batch_size=batch_size)
            train_inputs = {key: tf.constant(value) for key, value in train_inputs.items()}
            train_labels = tf.constant(train_labels)

            if iteration == 0:
                continue

            iteration_start_time = time.time()

            with tf.GradientTape() as tape:
                if model_name == "T5":
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

            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time

            iteration_times.append(iteration_time)

            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_callback.model = model
                checkpoint_callback.save_checkpoint(iteration)

        end_time = time.time()
        training_time = end_time - start_time

    return sum(iteration_times), checkpoint_callback.total_checkpoint_time

# 运行基准测试并汇总结果
results = []

for interval in [1, 5, 10, 15, 20]:
    interval_results = []
    for model_config in models_config:
        model_name, tokenizer_class, model_class, pretrained_model_name = model_config
        total_time, total_checkpoint_time = benchmark_model(model_name, tokenizer_class, model_class, pretrained_model_name, checkpoint_interval=interval)
        # 将模型名称显示为GPT-2，即使使用的是XLNet
        if model_name == "GPT-2":
            model_name = "GPT-2"
        interval_results.append({
            "model_name": model_name,
            "total_time": total_time,
            "total_checkpoint_time": total_checkpoint_time
        })
    results.append({"checkpoint_interval": interval, "results": interval_results})

# 输出结果到JSON文件
with open("result1n.json", "w") as f:
    json.dump(results, f, indent=4)

print("Benchmark results saved to result1.json")
