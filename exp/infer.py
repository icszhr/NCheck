import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import time
import random
import string



# 设置 TensorFlow 日志级别
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

# 加载 T5 模型和分词器
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = TFT5ForConditionalGeneration.from_pretrained("t5-large")

# 生成随机的推理数据
def generate_random_text(min_length=5, max_length=20):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# 生成大量推理数据
batch_size = 450
texts = [generate_random_text() for _ in range(batch_size)]

# 对所有推理数据进行推理
input_ids = tokenizer(texts, return_tensors="tf", padding=True, truncation=True)['input_ids']
attention_mask = tokenizer(texts, return_tensors="tf", padding=True, truncation=True)['attention_mask']

start_time = time.time()
outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
end_time = time.time()

# 计算总的推理时间
inference_time = end_time - start_time
print("Total inference time:", inference_time)
