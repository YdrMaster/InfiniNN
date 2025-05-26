use crate::Dim;
use std::collections::HashMap;

/// 神经网络标量参数
#[derive(Clone, Debug)]
pub enum Arg {
    Dim(Dim),
    Bool(bool),
    Int(u64),
    Float(f64),
    Str(&'static str),
    Arr(Box<[Self]>),
    Dict(HashMap<String, Self>),
}

macro_rules! impl_from {
    ($( $ty:ty => $variant:ident )+) => {
        $(
            impl From<$ty> for Arg {
                fn from(value: $ty) -> Self {
                    Self::$variant(value)
                }
            }
        )+
    };
}

impl_from! {
    Dim  => Dim
    bool => Bool
    u64  => Int
    f64  => Float
    &'static str           => Str
        Box<       [Self]> => Arr
    HashMap<String, Self > => Dict
}

impl Arg {
    pub fn dim(value: impl Into<Dim>) -> Self {
        value.into().into()
    }

    pub fn bool(value: bool) -> Self {
        value.into()
    }

    pub fn int(value: usize) -> Self {
        (value as u64).into()
    }

    pub fn float(value: f64) -> Self {
        value.into()
    }

    pub fn arr(value: impl IntoIterator<Item = Self>) -> Self {
        Self::Arr(value.into_iter().collect())
    }

    pub fn dict(value: impl IntoIterator<Item = (String, Self)>) -> Self {
        Self::Dict(value.into_iter().collect())
    }

    pub fn substitute(self, value: &HashMap<&str, usize>) -> Self {
        match self {
            Self::Dim(dim) => Self::Int(dim.substitute(value).unwrap() as _),
            Self::Arr(args) => Self::Arr(args.into_iter().map(|a| a.substitute(value)).collect()),
            Self::Dict(map) => Self::Dict(
                map.into_iter()
                    .map(|(k, v)| (k, v.substitute(value)))
                    .collect(),
            ),
            primitive => primitive,
        }
    }

    pub fn to_usize(&self) -> usize {
        match self {
            Self::Dim(dim) => dim.to_usize(),
            Self::Int(val) => *val as _,
            _ => panic!(),
        }
    }
}
