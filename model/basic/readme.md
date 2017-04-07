# Basic Model
最基本的Encoder-Decoder结构对话生成模型。不考虑语料中的情感（情绪）信息。
### Encoder
采用标准的单层LSTM网络，对于输入文本[x_1, x_2, ..., x_n]，使用LSTM网络对输入文本序列进行读取，得到LSTM隐藏层状态[c_1, c_2, ..., c_n]和LSTM输出[h_1, h_2, ..., h_n]。  
该模型不使用attention，因此只使用最后一个LSTM隐藏层状态c_n用于Decoder文本生成。LSTM输出被直接抛弃。

在实际实现中，采用mini-batch方式每次读入多个文本，因此输入的tensor形状为：[max_length, batch_size]。注意这里采用Time_Major形式的tensor来加快训练速度。因此输入形状例如：

[[今] [我] [晚] ...]  
[[天] [们] [上] ...]  
[[天] [一] [吃] ...]  
[[气] [起] [什] ...]  
[ ... ... ...  ...]  

### Decoder
采用标准的单层LSTM网络，对于输入的字符依次预测下一个输出。该模型在训练过程中使用LSTM网络上一个输出目标target作为输入，而不是用网络的上一个输出predict作为输入.
#### 2017.4.7  
更新输出部分增加了beam_search机制，得到了比单纯预测最高概率的单词更好的结果。

### 模型方法
1. __init__(): 模型构建和定义
2. variables_init(): 模型参数初始化
3. train(): 对模型进行训练
4. next_feed(): 自动从预料中获取下一个训练数据mini-batch
5. generate(): 从模型预测下一个文本
6. __print_result(): 训练过程中使用的打印函数（内部）
7. save(): 保存模型
8. restore(): 载入模型


