use arg::Arg;
use smallvec::SmallVec;
use tensor::Tensor;

#[repr(transparent)]
pub struct Graph<T>(pub graph::Graph<Node, Edge<T>>);

pub struct Node {
    pub name: String,
    pub op: String,
    pub arg: Option<Arg>,
}

pub struct Edge<T>(pub SmallVec<[Tensor<Info<T>, 2>; 1]>);

pub enum Info<T> {
    Internal(usize),
    External(External<T>),
}

pub struct External<T> {
    pub name: String,
    pub item: T,
}
