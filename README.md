# Deep Learning Homework

## 项目结构与内容说明

本项目为 CS231n 深度学习课程的作业实现,也是广东工业大学课程《深度学习》的实验，作业实现，与原版的不同的是，使用了uv包管理工具组织项目，并将原本每个notebook的source统一到dl_utils目录。具体结构与内容如下：

### 1. 作业 Notebook 说明

- **1, 2, 3, 4, 5, 6, 7**  
  分别对应课程的各次作业，每一个 notebook 文件实现和演示了特定的深度学习内容。
  - **1.softmax**：实现了softmax loss函数，以及softmax分类器,loss函数使用单元素循环和向量计算两种方式实现。
  - **2.two_layer_net**：实现全连接层和relu激活函数的，并搭建成两层mlp，包含前向传播、反向传播。
  - **3.optimal**：实现多层全连接网络，支持批归一化、dropout 等进阶技巧，并对多种优化器进行实验。
  - **4.BatchNormalization+dropout**：实现batch 、layer normalization、dropout 以及初始化超参数对训练的影响。
  - **5.cnn**：实现卷积神经网络（CNN），包括卷积、池化、全连接层，搭建三层卷积网络，实现对CIFAR-10数据集的分类并可视化了卷积核。
  - **6.rnn**：实现循环神经网络（RNN），包括RNN单步实现，以及RNN完整的网络的搭建，最后使用RNN实现一个给图片添加字幕的简易多模态模型。
  - **7.lstm**：实现LSTM网络，包括LSTM单步实现，以及LSTM完整的网络的搭建，最后使用LSTM实现一个给图片添加字幕的简易多模态模型。

### 2. `dl_utils` 目录说明

所有神经网络层的实现、优化器、数据集处理等 Python 文件都集中放在 `dl_utils` 目录下，方便复用和模块化开发。主要包含：

- **神经网络相关**
  - `fc_net.py`、`fc_net_with_norm_dropout.py`：全连接网络的实现，支持 batchnorm、layernorm、dropout 等模块。
  - `cnn.py`：卷积神经网络（如 ThreeLayerConvNet）的实现。
  - `layers.py`、`layers_cnn.py`：基础层（如 affine、relu、softmax、卷积、池化等）和它们的前向/反向传播实现。
  - `rnn_layers.py`：循环神经网络（如 LSTM 单步等）实现。
  - `fast_layers.py`：高效版的卷积和池化实现，通常基于 Cython 等加速。

- **优化器相关**
  - `optim.py`：实现了常见优化算法，如 SGD、SGD+Momentum、RMSProp、Adam 等，供训练网络时选择。

- **工具与数据集**
  - `data_utils.py`：提供了数据加载、预处理等工具方法（如 `get_CIFAR10_data`）。
  - `gradient_check.py`：用于数值梯度检查，确保反向传播实现的正确性。
  - 其它工具类函数用于调试、误差计算等。

### 3. 实验流程

- 每个 notebook 通过调用 `dl_utils` 目录下的实现文件，完成指定的实验内容。
- 支持自定义网络结构、不同优化器、正则化方式等参数配置，便于实验对比和性能分析。
- 推荐先阅读并理解 `dl_utils` 下各模块的 API 和实现，再在 notebook 中进行实验和调试。

## 如何运行

0. 确保已经安装了uv包管理工具
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install python3-dev
```

1. 安装依赖
```bash
uv sync
uv pip install -e .
```

2. 运行 Jupyter Notebook
```bash
source .venv/bin/activate
jupyter notebook
```
3. 或Visual Studio Code安装Jupyter插件，打开.ipynb后选择.venv即可。
---
