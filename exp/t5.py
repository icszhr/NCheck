import tensorflow as tf
from transformers import T5Tokenizer, TFT5Model
import time
import random
import string
import os
import numpy as np
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def generate_random_text(min_length=5, max_length=1000):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_training_data(tokenizer, min_length=5, max_length=10, batch_size=32):
    texts = [generate_random_text(min_length, max_length) for _ in range(batch_size)]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="tf")
    decoder_inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="tf")
    labels = inputs["input_ids"]
    return inputs, decoder_inputs, labels

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model, optimizer, loss_fn

def benchmark_t5(batch_size=1):
    print("Running benchmark for T5...")

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 添加填充token

    model = TFT5Model.from_pretrained("t5-base")

    model, optimizer, loss_fn = compile_model(model)

    train_inputs, decoder_inputs, train_labels = generate_training_data(tokenizer, batch_size=batch_size)
    train_inputs = {key: tf.constant(value) for key, value in train_inputs.items()}
    decoder_inputs = {key: tf.constant(value) for key, value in decoder_inputs.items()}
    train_labels = tf.constant(train_labels)

    # 前向传播和计算损失
    forward_pass_start_time = time.time()
    with tf.GradientTape() as tape:
        outputs = model(input_ids=train_inputs['input_ids'], decoder_input_ids=decoder_inputs['input_ids'], training=True)
        logits = outputs.last_hidden_state
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

    # 创建检查点对象
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # 模拟 GPU 到 CPU 的快照时间
    snapshot_start_time = time.time()
    model_weights_cpu = [w.numpy() for w in model.trainable_variables]  # 模拟快照到 CPU
    with tf.device('/GPU:0'):
        data = tf.random.normal([sum([tf.reduce_prod(w.shape) for w in model.trainable_variables])])
    with tf.device('/CPU:0'):
        data_cpu = tf.identity(data)
    snapshot_end_time = time.time()

    # 模拟将模型从 CPU 序列化到内存
    serialization_start_time = time.time()
    serialized_data = tf.io.serialize_tensor(data_cpu)
    serialization_end_time = time.time()

    # 计算总的检查点保存时间
    checkpoint_save_start_time = time.time()
    checkpoint.save("t5_checkpoint")  # 保存完整的检查点
    checkpoint_save_end_time = time.time()

    # 模拟文件系统管理时间
    filesystem_start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
    filesystem_end_time = time.time()

    # 模拟将序列化的数据写入文件系统并保存到 SSD
    total_write_and_ssd_copy_time = checkpoint_save_end_time - checkpoint_save_start_time
    write_and_ssd_copy_time = total_write_and_ssd_copy_time - (snapshot_end_time - snapshot_start_time) - (serialization_end_time - serialization_start_time)

    # 计算时间
    forward_pass_time = forward_pass_end_time - forward_pass_start_time
    backward_pass_time = backward_pass_end_time - backward_pass_start_time
    weight_update_time = weight_update_end_time - weight_update_start_time
    snapshot_time = snapshot_end_time - snapshot_start_time
    serialization_time = serialization_end_time - serialization_start_time
    filesystem_time = filesystem_end_time - filesystem_start_time

    print(f"Time taken for forward pass: {forward_pass_time:.2f} seconds")
    print(f"Time taken for backward pass: {backward_pass_time:.2f} seconds")
    print(f"Time taken for weight update: {weight_update_time:.2f} seconds")
    print(f"Time taken for snapshot (GPU to CPU): {snapshot_time:.2f} seconds")
    print(f"Time taken for serialization (CPU to memory): {serialization_time:.2f} seconds")
    print(f"Time taken for filesystem management: {filesystem_time:.2f} seconds")
    print(f"Time taken for write and SSD copy: {write_and_ssd_copy_time:.2f} seconds")

benchmark_t5()
