use std::{
    collections::{HashMap, hash_map::Entry},
    sync::RwLock,
};

/// 结构 ID，通过中心化的注册系统获得
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct NNId(usize);

/// 参数所有权
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ownership {
    /// 输入，外部构造内部释放
    In,
    /// 输出，内部构造外部释放
    Out,
    /// 输入输出，外部构造外部释放
    InOut { mutable: bool },
}

/// 参数信息，结构的每一个参数的所有权和可变性
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArgInfo {
    pub name: &'static str,
    pub ownership: Ownership,
}

/// 权重信息，结构的每一个权重的名称和是否必然存在
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WgtInfo {
    pub name: &'static str,
    pub optional: bool,
}

/// 结构信息的动态形式，由注册中心存档
pub struct NNInfo {
    /// 结构 ID
    pub id: NNId,
    /// 参数信息
    pub args: Box<[ArgInfo]>,
    /// 权重数量
    pub wgts: Box<[WgtInfo]>,
}

/// 注册中心，维护结构信息的注册和分配
#[repr(transparent)]
pub struct NNRegister(RwLock<(HashMap<String, NNId>, Vec<NNInfo>)>);

impl NNRegister {
    pub fn register<T: NuralNetwork>(&self) -> NNId {
        let mut guard = self.0.write().unwrap();
        let (map, vec) = &mut *guard;
        let id = NNId(map.len());
        let info = NNInfo {
            id,
            args: T::ARGS.to_vec().into(),
            wgts: T::WGTS.to_vec().into(),
        };
        match map.entry(T::NAME.into()) {
            Entry::Occupied(entry) => {
                let info_ = &vec[entry.get().0];
                assert_eq!(info.args, info_.args);
                assert_eq!(info.wgts, info_.wgts);
                info_.id
            }
            Entry::Vacant(entry) => {
                entry.insert(id);
                vec.push(info);
                id
            }
        }
    }
}

/// 结构定义
pub trait NuralNetwork {
    /// 结构名称，由注册中心检查不重复
    const NAME: &str;
    /// 参数信息，参数的数量、所有权和可变性
    const ARGS: &[ArgInfo];
    /// 权重数量，此结构中直接使用的权重数量，不包括子结构权重
    const WGTS: &[WgtInfo];

    /// 结构 ID，由注册中心分配
    fn id(&self) -> NNId;
    /// 子结构数量和类型，可能由超参决定
    fn subs(&self) -> Box<[NNId]>;
}

pub struct Normalization {
    id: NNId,
    ty: Type,
}

#[derive(Clone, Copy, Debug)]
pub enum Type {
    RmsNorm { epsilon: f32 },
    LayerNorm,
}

impl NuralNetwork for Normalization {
    const NAME: &str = "normalization";

    const ARGS: &[ArgInfo] = &[
        ArgInfo {
            name: "y",
            ownership: Ownership::Out,
        },
        ArgInfo {
            name: "x",
            ownership: Ownership::In,
        },
    ];

    const WGTS: &[WgtInfo] = &[
        WgtInfo {
            name: "scale",
            optional: false,
        },
        WgtInfo {
            name: "bias",
            optional: true,
        },
    ];

    fn id(&self) -> NNId {
        self.id
    }

    fn subs(&self) -> Box<[NNId]> {
        Box::new([])
    }
}

pub enum ArgType {
    Struct(&'static [(&'static str, ArgType)]),
    Array(&'static ArgType),
    Optional(&'static ArgType),
    Tensor,
}

const ATTENTION: ArgType = ArgType::Struct(&[
    ("q", ArgType::Tensor),
    ("k", ArgType::Tensor),
    ("v", ArgType::Tensor),
    ("o", ArgType::Tensor),
    (
        "cache",
        ArgType::Optional(&ArgType::Struct(&[
            ("k_cache", ArgType::Tensor),
            ("v_cache", ArgType::Tensor),
            ("pos", ArgType::Tensor),
        ])),
    ),
]);

const SELF_ATTN: ArgType = ArgType::Struct(&[
    ("y", ArgType::Tensor),
    ("x", ArgType::Tensor),
    ("pos", ArgType::Tensor),
    ("n_sin", ArgType::Tensor),
    ("n_cos", ArgType::Tensor),
    ("reqs", ArgType::Array(&REQUEST)),
]);

const REQUEST: ArgType = ArgType::Struct(&[
    ("k_cache", ArgType::Tensor),
    ("v_cache", ArgType::Tensor),
    ("n_seq", ArgType::Tensor),
    ("pos", ArgType::Tensor),
]);
