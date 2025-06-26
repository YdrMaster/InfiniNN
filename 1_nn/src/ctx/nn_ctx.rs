use super::{GraphBuilder, Tensor, TensorMeta, internal::GraphContext, stack::Stack};
use crate::{Arg, Dim, NNError, NNGraph, NuralNetwork};
use std::fmt::Display;
use tensor::digit_layout::DigitLayout;

impl GraphBuilder {
    pub fn build<T, NN: NuralNetwork<T>>(
        &self,
        nn: NN,
        inputs: impl IntoIterator<Item = TensorMeta>,
    ) -> Result<NNGraph<T>, NNError> {
        let (context, inputs) = self.new_context(inputs);
        let outputs = nn
            .launch(
                inputs,
                Context {
                    graph: &context,
                    stack: Stack::new("Ω"),
                },
            )
            .map(|(_, outputs)| outputs)?;
        Ok(context.take().into_graph(outputs))
    }
}

pub struct Context<'g, T> {
    graph: &'g GraphContext<T>,
    stack: Stack,
}

impl<T> Context<'_, T> {
    pub fn path(&self) -> String {
        self.stack.top().borrow().path().to_string()
    }

    pub fn trap<NN: NuralNetwork<T>>(
        &mut self,
        name: impl ToString,
        nn: NN,
        inputs: impl IntoIterator<Item = Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, NNError> {
        self.stack.push(name);
        let outputs = nn
            .launch(
                inputs,
                Context {
                    graph: self.graph,
                    stack: self.stack.clone(),
                },
            )
            .map(|(_, outputs)| outputs);
        self.stack.pop();
        outputs
    }

    pub fn load_external(
        &mut self,
        name: impl Display,
        dt: DigitLayout,
        shape: impl IntoIterator<Item = Dim>,
        item: T,
    ) -> Tensor<T> {
        let top = self.stack.top();
        let mut top = top.borrow_mut();
        assert!(top.tensor.check(&name));
        let path = format!("{}.{}", top.path(), name);
        self.graph.load_external(path, dt, shape, item)
    }

    pub fn save_external(&mut self, name: impl Display, tensor: Tensor<T>, item: T) {
        let top = self.stack.top();
        let mut top = top.borrow_mut();
        assert!(top.tensor.check(&name));
        let path = format!("{}.{}", top.path(), name);
        self.graph.save_external(path, tensor, item)
    }

    pub fn call(
        &mut self,
        name: impl Display,
        op: impl AsRef<str>,
        arg: Option<Arg>,
        inputs: impl IntoIterator<Item = Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, NNError> {
        let top = self.stack.top();
        let mut top = top.borrow_mut();
        // 没有设置名字的，使用 op 名作为名字
        let mut name = name.to_string();
        if name.is_empty() {
            name = op.as_ref().into()
        }
        // 加序号去重
        let name = top.operator.decorate(name.clone());
        let path = format!("{}:{}", top.path(), name);
        // 连接到图
        self.graph
            .call(path.clone(), op, inputs, arg)
            .map_err(|err| NNError { name: path, err })
    }
}
