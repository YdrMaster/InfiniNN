# 神经网络

[![CI](https://github.com/YdrMaster/InfiniNN/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/YdrMaster/InfiniNN/actions)
[![license](https://img.shields.io/github/license/YdrMaster/InfiniNN)](https://mit-license.org/)
[![GitHub Issues](https://img.shields.io/github/issues/YdrMaster/InfiniNN)](https://github.com/YdrMaster/InfiniNN/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/YdrMaster/InfiniNN)](https://github.com/YdrMaster/InfiniNN/pulls)
![GitHub repo size](https://img.shields.io/github/repo-size/YdrMaster/InfiniNN)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/YdrMaster/InfiniNN)
![GitHub contributors](https://img.shields.io/github/contributors/YdrMaster/InfiniNN)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/YdrMaster/InfiniNN)

> - ⚠️ 本项目还处在🚧早期探索和💡设计阶段，API 可能激烈变动。
> - 💬 关于本项目的🧠思路、📃方案以及其他渠道收集的❓意见将发布在[🗪讨论区](https://github.com/YdrMaster/InfiniNN/discussions)，也欢迎直接[发帖](https://github.com/YdrMaster/InfiniNN/discussions/new/choose)讨论！

神经网络结构是**人工智能知识**到**人工智能系统**的映射，旨在向上隔离系统复杂性，支持 AI 模型开发者用相对简单的 API 描述神经网络。一个完整的结构定义由下列部分组成：

1. 超参数/元信息：神经网络先验知识，隐含着有关网络的功能和特性的知识。元信息一定是在模型训练时确定的；
2. 执行信息：包含不属于元信息，但对于一轮执行所必要的其他所有信息的结构，通常是一系列张量布局。执行信息与网络中传递的数据无关；
3. 执行环境：包含运行时的动态结构，执行时可从中获得算子列表、队列、各个张量和中间状态的地址等运行时数对

通过下列示例了解详情：

- [Activation](/src/nn/activation.rs);
- [Normalization](/src/nn/normalization.rs);
- [Attention](/src/nn/attention.rs);
- [Mlp](/src/nn/mlp.rs);
