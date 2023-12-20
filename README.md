# Weibo-of-China-EPB-content-analysis

## 简介
本仓库是对于对外经济贸易大学《TODO》课程的代码开源。

同时，是论文《TODO》的代码开源。

## 结构

.
├── bin/                    # 存放需要的二进制文件
│   └── wiki_word2vec_50.bin
├── data/                   # 部分原始数据集示例
│   ├── merged_data_2013.csv
│   ├── merged_data_2014.csv
│   └── TODO
├── src/                    # 存放源代码
│   ├── classify.py         # 调用模型进行分类
│   ├── cnn.py              # 训练cnn模型
│   └── split.py            # 根据原始数据创建数据集
├── README.md               # 项目文档
├── requirements.txt        # 依赖项文件
├── LICENSE                 # 许可证文件
└── .gitignore              # Git忽略文件

### 数据集

OfficialMicroblogMLDataset是目前最大的环境相关政务微博数据集。我们对爬取的25000条微博数据进行了标注，按照其与政务的相关程度逻辑回归标注为二值文本。

如果只需要直接引用OfficialMicroblogMLDataset，可以值关注model数据下的三个txt文件。

train.txt:训练集，包含20000条数据

validation.txt:验证集，包含2500条数据

test.txt:测试集，包含2500条数据

可以参考src/cnn.py对该数据集的使用方法。

如果关注OfficialMicroblogMLDataset的其他特征，需要对原始数据重新标注，可以参考/data下的部分数据。

全部数据链接：[TODO]

### model训练

本repo利用tensorflow创建了cnn模型。如需复现，可直接执行

```bash
TODO
```

## 引用

TODO:给出引用格式




