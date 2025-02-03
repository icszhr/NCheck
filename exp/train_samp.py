import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
import time
import os

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.get_logger().setLevel('ERROR')

# 加载模型和分词器
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

# 准备数据的函数
def generate_training_data(batch_size=10):
    sample_text = ["Hello world"] * batch_size  # 简单的重复样本
    inputs = tokenizer(sample_text, return_tensors="tf", padding=True, truncation=True)
    return inputs["input_ids"], inputs["attention_mask"]

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(input_ids, attention_mask):
    with tf.GradientTape() as tape:
        logits = model(input_ids, attention_mask=attention_mask, training=True)[0]
        loss = loss_fn(input_ids, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 执行多个训练迭代
num_iterations = 10  # 设定迭代次数为10
for i in range(num_iterations):
    input_ids, attention_mask = generate_training_data(batch_size=1)  # 每次迭代重新生成数据

    # 前向传播的时间测量
    forward_start_time = time.time()
    logits = model(input_ids, attention_mask=attention_mask, training=True)[0]
    forward_time = time.time() - forward_start_time

    # 损失计算
    loss = loss_fn(input_ids, logits)

    # 反向传播的时间测量
    backward_start_time = time.time()
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(input_ids, attention_mask=attention_mask, training=True)[0]
        loss = loss_fn(input_ids, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    backward_time = time.time() - backward_start_time

    # 权重更新的时间测量
    update_start_time = time.time()
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    update_time = time.time() - update_start_time

    print(f"Iteration {i}: Loss = {loss.numpy():.4f}, "
          f"Forward Time = {forward_time:.4f}s, "
          f"Backward Time = {backward_time:.4f}s, "
          f"Update Time = {update_time:.4f}s")
