import os
import time
import pickle
import json
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, TFGPT2Model, AutoTokenizer

# 设置环境变量以减少日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建随机数据集
batch_size = 8
seq_length = 128

def create_random_dataset(batch_size, seq_length):
    input_ids = tf.random.uniform([batch_size, seq_length], minval=0, maxval=30522, dtype=tf.int32)
    attention_mask = tf.ones([batch_size, seq_length], dtype=tf.int32)
    labels = tf.random.uniform([batch_size], minval=0, maxval=2, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_mask}, labels))
    return dataset.batch(batch_size)

# 检查点保存路径
ssd_checkpoint_dir = "./checkpoints"

# 保存检查点到SSD并测量时间
def save_ssd_checkpoint(model, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')
    start_time = time.time()
    weights = model.get_weights()
    serialize_start_time = time.time()
    data = pickle.dumps(weights)
    serialize_time = time.time() - serialize_start_time
    with open(checkpoint_path, 'wb') as f:
        f.write(data)
    total_time = time.time() - start_time
    return checkpoint_path, total_time, serialize_time

# 从SSD检查点恢复并测量时间
def restore_from_ssd_checkpoint(model, checkpoint_path):
    start_time = time.time()
    with open(checkpoint_path, 'rb') as f:
        data = f.read()
    deserialize_start_time = time.time()
    weights = pickle.loads(data)
    deserialize_time = time.time() - deserialize_start_time
    model.set_weights(weights)
    total_time = time.time() - start_time
    return total_time, deserialize_time

# 保存检查点到NVM（模拟）并测量时间
nvm_storage = {}

def save_nvm_checkpoint(model):
    start_time = time.time()
    weights = model.get_weights()
    global nvm_storage
    nvm_storage = {i: weight for i, weight in enumerate(weights)}
    total_time = time.time() - start_time
    return total_time

# 从NVM（模拟）检查点恢复并测量时间
def restore_from_nvm_checkpoint(model):
    start_time = time.time()
    global nvm_storage
    weights = [tf.convert_to_tensor(nvm_storage[i]) for i in range(len(nvm_storage))]
    model.set_weights(weights)
    total_time = time.time() - start_time
    return total_time

# 结果保存路径
results_file = "checkpoint_results.json"

# 测试代码
model_names = {
    "bert-base-uncased": TFAutoModelForSequenceClassification,
    "roberta-base": TFAutoModelForSequenceClassification,
    "distilbert-base-uncased": TFAutoModelForSequenceClassification,
    "gpt2": TFGPT2Model
}

results = {}

for model_name, model_class in model_names.items():
    print(f"Testing checkpoint performance for {model_name}...")

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 初始化模型和数据集
    model = model_class.from_pretrained(model_name, num_labels=2)
    dataset = create_random_dataset(batch_size, seq_length)

    # 配置优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # 保存和恢复SSD检查点
    print(f"Saving {model_name} checkpoint to SSD...")
    ssd_checkpoint_path, ssd_save_time, ssd_serialize_time = save_ssd_checkpoint(model, ssd_checkpoint_dir)
    print(f"SSD checkpoint save time: {ssd_save_time} seconds")
    print(f"SSD serialization time: {ssd_serialize_time} seconds")

    print(f"Saving {model_name} checkpoint to NVM...")
    nvm_save_time = save_nvm_checkpoint(model)
    print(f"NVM checkpoint save time: {nvm_save_time} seconds")

    print(f"Restoring {model_name} from SSD checkpoint...")
    ssd_restore_time, ssd_deserialize_time = restore_from_ssd_checkpoint(model, ssd_checkpoint_path)
    print(f"SSD checkpoint restore time: {ssd_restore_time} seconds")
    print(f"SSD deserialization time: {ssd_deserialize_time} seconds")

    print(f"Restoring {model_name} from NVM checkpoint...")
    nvm_restore_time = restore_from_nvm_checkpoint(model)
    print(f"NVM checkpoint restore time: {nvm_restore_time} seconds")

    results[model_name] = {
        "ssd_save_time": ssd_save_time,
        "ssd_serialize_time": ssd_serialize_time,
        "nvm_save_time": nvm_save_time,
        "ssd_restore_time": ssd_restore_time,
        "ssd_deserialize_time": ssd_deserialize_time,
        "nvm_restore_time": nvm_restore_time
    }

# 将结果保存到文件中
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {results_file}")
