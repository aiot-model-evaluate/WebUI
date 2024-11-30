from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.utils.data import DataLoader, SequentialSampler
import random
from tqdm import tqdm
from mypack.datasets import read_squad_examples, convert_examples_to_features, SquadFeaturesDataset

from monitor import GPUMonitor
import time

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 监控GPU性能的函数 (负责间隔时间采样，可以用于计算"能耗"+"能效比")
filename = 'infer_Result.log'
monitor = GPUMonitor(device_index=1, interval = 1, filename = filename)

# 训练参数
eval_file = "./Datasets/SQuAD/dev-v2.0.json"
eval_batch_size = 8
num_eval_epochs = 1
warmup_proportion = 0.1
max_seq_length = 512
doc_stride = 128
max_query_length = 64

# 加载模型与tokenizer

tokenizer = BertTokenizer.from_pretrained('./model/Bert-Tiny/')
start_load = time.time()
model = BertForQuestionAnswering.from_pretrained("./model/Bert-Tiny/")
model.to(device)
finish_load = time.time()
model.eval()    # 将模型转换为eval模式

# 准备数据
eval_examples = read_squad_examples(input_file=eval_file, is_training=False)
num_eval_steps = int(len(eval_examples) / eval_batch_size * num_eval_epochs)
num_warmup_steps = int(num_eval_steps * warmup_proportion)
rng = random.Random(12345)
rng.shuffle(eval_examples)

# 对数据进行tokenize
eval_features = convert_examples_to_features(
    examples=eval_examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=False
)

# 创建DataLoader
eval_dataset = SquadFeaturesDataset(eval_features)
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

# 执行测试
#---------------------------------------
inference_start = time.time()
sum_forward_time = 0
forward_count = 0
batch_num = len(eval_dataloader)
#---------------------------------------

# ---------------------------------------
monitor.start_monitoring()
# ---------------------------------------

# 进行推理
all_results = []
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)

    with torch.no_grad():
        forward_start = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        forward_end = time.time()
        sum_forward_time += forward_end - forward_start
        forward_count += 1
        
#---------------------------------------
indference_end = time.time()
inference_time = indference_end - inference_start
#---------------------------------------
#-----------------------------------------------------------------------------
# 结束监控线程
monitor.stop_monitoring()

#-----------------------------------------------------------------------------
monitor.write(f'Batch num: {batch_num}')
monitor.write(f'Load time: {finish_load - start_load}')
monitor.write(f'Average Forward delay: {sum_forward_time / forward_count}')
monitor.write(f'Total Inference time: {inference_time}')
monitor.cleanup()

# 程序结束标志
print("Program finished.")