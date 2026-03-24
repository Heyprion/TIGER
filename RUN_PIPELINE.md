# TIGER 运行文档

本文档整理了这个仓库的完整运行链路，包括：

1. 第一步：数据预处理
2. 第二步：训练 RQVAE
3. 第三步：用 RQVAE 生成离散 code
4. 第四步：训练最终的 T5 推荐模型


## 0. 环境准备

默认假设：

- 仓库目录：`/root/autodl-tmp/TIGER`
- conda 环境：`rec-env`
- 数据集已经放到 `data/` 下

建议先执行：

```bash
cd /root/autodl-tmp/TIGER
conda activate rec-env
export HF_HUB_ENABLE_HF_TRANSFER=0
```

参数说明：

- `conda activate rec-env`
  - 使用项目运行环境
- `HF_HUB_ENABLE_HF_TRANSFER=0`
  - 避免 `huggingface_hub` 在没装或不兼容 `hf_transfer` 时下载失败


## 1. 第一步：数据预处理

### 1.1 官方仓库原始方式

这个仓库的第一步没有单独封装成 Python 脚本，原始入口是 notebook：

```bash
cd /root/autodl-tmp/TIGER/data
jupyter notebook process.ipynb
```

然后按顺序执行 `process.ipynb` 里的关键代码单元，生成：

- `Beauty.json`
- `Beauty_metadata.json`
- `train.parquet`
- `valid.parquet`
- `test.parquet`
- `item_emb.parquet`

### 1.2 第一步里的可改参数

在 `data/process.ipynb` 里主要改这些：

- `dataset_name = "Beauty"`
  - 想换数据集时改成别的名字，例如 `Sports`、`Toys`
- `SentenceTransformer('./sentence-t5-base')`
  - 如果本地模型目录不同，可以改成本地路径或 Hugging Face 模型名

### 1.3 第一步成功后的关键输出

输出目录：

```bash
data/Beauty/
```

关键文件：

- `data/Beauty/item_emb.parquet`
- `data/Beauty/train.parquet`
- `data/Beauty/valid.parquet`
- `data/Beauty/test.parquet`


## 2. 第二步：训练 RQVAE

### 2.1 运行命令

```bash
cd /root/autodl-tmp/TIGER/rqvae
python main.py \
  --data_path ../data/Beauty/item_emb.parquet \
  --ckpt_dir ./ckpt/Beauty \
  --device cuda:0
```

### 2.2 参数说明

- `--data_path`
  - 第一步生成的 item embedding 文件
  - 默认是 `../data/Beauty/item_emb.parquet`
- `--ckpt_dir`
  - 第二步权重保存目录
  - 实际会保存到 `./ckpt/Beauty/<时间戳>/`
- `--device`
  - 训练设备，常用 `cuda:0` 或 `cpu`

### 2.3 常用可调参数

下面这些都可以直接在命令行里改：

```bash
cd /root/autodl-tmp/TIGER/rqvae
python main.py \
  --data_path ../data/Beauty/item_emb.parquet \
  --ckpt_dir ./ckpt/Beauty \
  --device cuda:0 \
  --epochs 3000 \
  --batch_size 1024 \
  --lr 1e-3 \
  --eval_step 50 \
  --num_emb_list 256 256 256 \
  --layers 512 256 128 64 \
  --e_dim 32
```

参数说明：

- `--epochs`
  - 总训练轮数
- `--batch_size`
  - 训练 batch size
- `--lr`
  - 学习率
- `--eval_step`
  - 每多少个 epoch 做一次 collision evaluation 和 checkpoint 保存
- `--num_emb_list`
  - 每一级 codebook 的大小
- `--layers`
  - MLP 编码器/解码器隐藏层维度
- `--e_dim`
  - 量化后的 embedding 维度

### 2.4 第二步输出的权重命名格式

输出目录格式：

```text
rqvae/ckpt/Beauty/<时间戳>/
```

时间戳格式：

```text
Mon-DD-YYYY_HH-MM-SS
```

例如：

```text
rqvae/ckpt/Beauty/Mar-24-2026_01-32-24/
```

里面常见文件：

- `best_loss_model.pth`
  - 当前训练中 `train loss` 最优的模型
- `best_collision_model.pth`
  - 当前训练中 `collision_rate` 最优的模型
- `epoch_<epoch>_collision_<rate>_model.pth`
  - 每次评估时额外保存的阶段性 checkpoint


## 3. 第三步：生成离散 code

第三步现在支持通过配置文件控制加载哪个 RQVAE 模型。

### 3.1 推荐方式：用配置文件运行

先编辑配置文件：

```bash
cd /root/autodl-tmp/TIGER
sed -n '1,120p' rqvae/generate_code.example.yaml
```

示例配置文件内容：

```yaml
dataset: Beauty
ckpt_path: ./ckpt/Beauty/Mar-24-2026_01-32-24/best_collision_model.pth
output_file: ../data/Beauty/Beauty_t5_rqvae.npy
device: cuda:0
batch_size: 64
```

运行命令：

```bash
cd /root/autodl-tmp/TIGER/rqvae
python generate_code.py --config generate_code.example.yaml
```

### 3.2 不用配置文件，直接命令行运行

```bash
cd /root/autodl-tmp/TIGER/rqvae
python generate_code.py \
  --ckpt_path ./ckpt/Beauty/Mar-24-2026_01-32-24/best_collision_model.pth \
  --output_file ../data/Beauty/Beauty_t5_rqvae.npy \
  --device cuda:0 \
  --batch_size 64
```

### 3.3 参数说明

- `--config`
  - JSON 或 YAML 配置文件路径
- `--ckpt_path`
  - 第二步训练得到的 RQVAE checkpoint
  - 推荐优先用 `best_collision_model.pth`
- `--output_file`
  - 第三步生成的离散 code 输出文件
- `--device`
  - 编码设备
- `--batch_size`
  - 第三步生成 code 时的 batch size

### 3.4 如果想改第三步配置参数，怎么改

推荐改 `rqvae/generate_code.example.yaml` 里的值：

- 想换 checkpoint：
  - 改 `ckpt_path`
- 想换输出文件：
  - 改 `output_file`
- 想换设备：
  - 改 `device`
- 想换 batch size：
  - 改 `batch_size`

也可以命令行覆盖配置文件里的值，命令行优先级更高。

### 3.5 第三步输出

```bash
data/Beauty/Beauty_t5_rqvae.npy
```

这个文件的作用：

- 它把每个 item id 映射成一个离散 code 序列
- 第四步训练推荐模型时会读取它，把历史 item 和目标 item 转成 token 序列


## 4. 第四步：训练最终推荐模型

真正的推荐器在 `model/` 目录，不在 `rqvae/` 目录。

### 4.1 运行命令

```bash
cd /root/autodl-tmp/TIGER/model
python main.py \
  --dataset_path ../data/Beauty \
  --code_path ../data/Beauty/Beauty_t5_rqvae.npy \
  --device cuda \
  --mode train \
  --save_path ./ckpt/tiger_beauty.pth \
  --log_path ./logs/tiger_beauty.log
```

### 4.2 参数说明

- `--dataset_path`
  - 第一阶段生成的 parquet 数据目录
- `--code_path`
  - 第三阶段生成的 item-to-code 文件
- `--device`
  - 训练设备
- `--mode`
  - 目前主要用 `train`
- `--save_path`
  - 最优推荐模型保存路径
- `--log_path`
  - 训练日志保存路径

### 4.3 常用可调参数

```bash
cd /root/autodl-tmp/TIGER/model
python main.py \
  --dataset_path ../data/Beauty \
  --code_path ../data/Beauty/Beauty_t5_rqvae.npy \
  --device cuda \
  --batch_size 256 \
  --infer_size 96 \
  --num_epochs 200 \
  --lr 1e-4 \
  --num_layers 4 \
  --num_decoder_layers 4 \
  --d_model 128 \
  --d_ff 1024 \
  --num_heads 6 \
  --d_kv 64 \
  --dropout_rate 0.1 \
  --max_len 20 \
  --beam_size 30 \
  --ndcg20_eval_interval 1 \
  --full_eval_interval 20 \
  --save_path ./ckpt/tiger_beauty.pth \
  --log_path ./logs/tiger_beauty.log
```

参数说明：

- `--batch_size`
  - 训练 batch size
- `--infer_size`
  - 验证/测试时的 batch size
- `--num_epochs`
  - 最终推荐器训练轮数
- `--lr`
  - 学习率
- `--num_layers`
  - T5 encoder 层数
- `--num_decoder_layers`
  - T5 decoder 层数
- `--d_model`
  - 隐层维度
- `--d_ff`
  - FFN 维度
- `--num_heads`
  - 注意力头数
- `--d_kv`
  - key/value 维度
- `--dropout_rate`
  - dropout
- `--max_len`
  - 历史序列最大长度
- `--beam_size`
  - 生成推荐时的 beam search 大小
- `--ndcg20_eval_interval`
  - 每多少个 epoch 做一次最基本的验证集 `NDCG@20`
  - 这个指标用于 early stop 和保存最好模型
- `--full_eval_interval`
  - 每多少个 epoch 才做一次完整的 `Recall@5/10/20` 和 `NDCG@5/10/20`
  - 非完整评估 epoch 只计算 `NDCG@20`，用于 early stop 和保存最好模型


## 5. 各步骤之间的依赖关系

依赖顺序如下：

1. 第一步输出 `item_emb.parquet`
2. 第二步读取 `item_emb.parquet`，输出 RQVAE checkpoint
3. 第三步读取第二步 checkpoint，输出 `Beauty_t5_rqvae.npy`
4. 第四步同时读取：
   - `train/valid/test.parquet`
   - `Beauty_t5_rqvae.npy`


## 6. 最短运行清单

如果你已经完成第一步，并且只想顺着跑下去：

### 第二步

```bash
cd /root/autodl-tmp/TIGER/rqvae
conda run --no-capture-output -n rec-env python main.py \
  --data_path ../data/Beauty/item_emb.parquet \
  --ckpt_dir ./ckpt/Beauty \
  --device cuda:0
```

### 第三步

```bash
cd /root/autodl-tmp/TIGER/rqvae
conda run -n rec-env python generate_code.py \
  --ckpt_path ./ckpt/Beauty/Mar-24-2026_01-32-24/best_collision_model.pth \
  --output_file ../data/Beauty/Beauty_t5_rqvae.npy \
  --device cuda:0
```

### 第四步训练

```bash
cd /root/autodl-tmp/TIGER/model
conda run --no-capture-output -n rec-env python main.py \
  --dataset_path ../data/Beauty \
  --code_path ../data/Beauty/Beauty_t5_rqvae.npy \
  --device cuda \
  --save_path ./ckpt/tiger_beauty.pth \
  --log_path ./logs/tiger_beauty.log
```

### 第四步训练后测试最佳权重

```bash
cd /root/autodl-tmp/TIGER/model
python main.py \
  --mode evaluation \
  --dataset_path ../data/Beauty \
  --code_path ../data/Beauty/Beauty_t5_rqvae.npy \
  --device cuda \
  --save_path ./ckpt/tiger_beauty.pth \
  --log_path ./logs/tiger_beauty_eval.log
```

说明：

- `--mode evaluation`
  - 只加载 `--save_path` 指定的权重并在测试集上评估，不再进入训练循环
- `--save_path`
  - 指向第四步训练保存的最佳模型权重
- 如果训练时改过模型结构参数，例如 `--num_layers`、`--d_model`、`--num_heads`、`--max_len`，那么测试时也必须传同样的值，否则会加载失败
