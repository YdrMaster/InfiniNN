//! 网络结构是后端系统与人工神经网络先验知识结合的范式，由下列部分组成：
//!
//! 1. 超参数/元信息：神经网络先验知识，隐含着网络的功能和特性，同时元信息设计决定了执行接口的形式；
//! 2. 执行副本：包含不属于元信息，但对于一轮执行所必要的其他所有信息的结构，
//!    主要是 batch size、request info，以及每个张量的 strides 和 offset；
//! 3. 执行环境：包含运行时的动态结构，执行时可从中获得算子列表、队列、各个张量和中间状态的地址等运行时数对象；

pub mod activation;
pub mod attention;
pub mod mlp;
pub mod normalization;
