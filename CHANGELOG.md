# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2025.03.14

### Changed

- 单个 crate 划分为 3 个独立的 crate；
- 扩展 test-vm 功能，整理结构测试；
- 完善和规范化栈追踪实现，作为核心组件提供栈追踪器；

### Fixed

- 改正 `attention` 中 softmax 使用的张量布局；
- 为各个结构元信息实现 `Clone`；
- 所有结构和参数字段全部 `pub`，允许直接构造；

## [0.0.1] - 2025.03.12

### Changed

- 完整重构，提出虚拟机概念；

### Added

- 支持比较方便的、结构化的参数加载和推理执行；
- 完成 linear、linear residual、normalization、mlp、attention、self attention、transformer blk 结构；
- 为每个结构添加测试；

## [0.0.0] - 2025.03.09

### Added

- 创建项目；
- 初步实现了分离的 `LayoutManage` 和 `MemManage` 用于管理结构定义中的张量布局信息和结构执行中的张量分配释放过程；
- 定义了 8 个主要算子，并搭建了**激活**、**归一**、**注意力**和**感知机**结构；
- 定义了一个简单的“测试录制器”以验证运行时功能；

[0.0.2]: https://github.com/YdrMaster/InfiniNN/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/YdrMaster/InfiniNN/compare/v0.0.0...v0.0.1
[0.0.0]: https://github.com/YdrMaster/InfiniNN/releases/tag/v0.0.0
