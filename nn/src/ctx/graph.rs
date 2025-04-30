use super::TensorMeta;
use crate::Arg;

pub struct Node {
    pub name: String,
    pub op: String,
    pub arg: Option<Arg>,
}

pub struct Edge<T> {
    pub meta: TensorMeta,
    pub external: Option<External<T>>,
}

pub struct External<T> {
    pub name: String,
    pub item: T,
}
