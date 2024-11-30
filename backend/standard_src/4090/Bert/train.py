import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering, AdamW
from datasets import load_dataset
import numpy as np

from monitor import GPUMonitor
import time

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 启动监控线程
filename = 'train_Result.log'
monitor = GPUMonitor(device_index = 1, interval = 1, filename = filename)
#-----------------------------------------------------------------------------

# 加载数据集
dataset = load_dataset("./Datasets/squad_v2/")

# 加载BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("./model/Bert-Tiny/")


def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["question"],
        examples["context"],
        padding="max_length",
        truncation=True,
        max_length=384,
        return_offsets_mapping=True,
    )
    
    # 获取答案的起始位置和文本
    start_positions = []
    end_positions = []
    for answer in examples["answers"]:
        if len(answer["answer_start"]) == 0:
            # 如果没有答案，标记为0
            start_positions.append(0)
            end_positions.append(0)
        else:
            start = answer["answer_start"][0]
            text = answer["text"][0]
            start_positions.append(start)
            end_positions.append(start + len(text))
    
    tokenized_inputs["start_positions"] = start_positions
    tokenized_inputs["end_positions"] = end_positions
    return tokenized_inputs

# 数据集tokenize
tokenized_datasets = dataset.map(tokenize_function, batched=True)

def custom_collate_fn(batch):
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
    start_positions = torch.tensor([example['start_positions'] for example in batch])
    end_positions = torch.tensor([example['end_positions'] for example in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions
    }

# 数据加载器
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=custom_collate_fn)

# 加载BERT模型并移动到GPU
model = BertForQuestionAnswering.from_pretrained("./model/Bert-Tiny/")
model.to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

#---------------------------------------
training_start = time.time()
batch_num = len(train_dataloader)
monitor.start_monitoring()
#---------------------------------------

# 训练
model.train()
for epoch in range(1):  # 训练3个epoch
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")

#---------------------------------------
training_end = time.time()
training_time = training_end - training_start
#---------------------------------------
monitor.stop_monitoring()
monitor.write(f'Batch num: {batch_num}')
monitor.write(f'Total training time: {training_time}')
monitor.cleanup()

# 程序结束标志
print("Program finished.")

