# Sentiment classifier for financial news based on bert series pre-trained models
## Training data
The training data is from https://xueqiu.com and is in Chinese language.
正样本（pos）：6873条，负样本（neg）：3591条；
## 一些测试实验
模型结构为：原始预训练语言模型后接两层dense；
* 使用albert_chinese_small，参数全调，epoch=10, lr=0.001, 准确率97%；
* 使用albert_chinese_small，只调节原始模型的后面的一层dense层与后接的两层sense，epoch=30, lr=0.004, 准确率92%；


