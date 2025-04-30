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

## 静态化工作流

1. 计算逻辑
   1. 逻辑图变换 1
   2. 形状锁定
   3. 逻辑图变换 2
2. 存储管理
   1. 复合类型分析（生成内存块重边）
   2. 区分内部外部存储（所有权决定）
   3. 元信息变换生成（存储布局约束）
   4. 可变性分析
   5. 算子原地化（内存块合并）
3. 地址锁定
   1. 内存块生命周期分析
   2. 偏移计算
   3. 地址锁定（虚存分配，不支持虚存则分配物理内存）
   4. 硬件计算图（地址相关计算图）生成
4. 调度执行
