mod macros;
mod nn;
mod op;
mod tensor;

use ::tensor::digit_layout::DigitLayout;
use rwrc::RwRc;
use std::{any::Any, collections::HashMap};
use tensor::{BackwardTensorOf, KernelTensorOf, NNTensorId};

pub use nn::NeuralNetwork;
pub use op::Operator;

pub type NNTensor<VM> = tensor::NNTensor<<VM as VirtualMachine>::Memory>;
pub type TenosrOf<VM> = tensor::Tensor<RwRc<<VM as VirtualMachine>::Memory>>;

pub trait VirtualMachine: Sized + 'static {
    type Domain: Domain<Self>;
    type Memory;

    fn process(&self) -> usize;
    fn domain(&self, process: usize, device: usize) -> Self::Domain;
}

pub trait Domain<VM: VirtualMachine> {
    fn tensor(&self, dt: DigitLayout, shape: &[usize]) -> TenosrOf<VM>;
    fn launch<'a>(
        &self,
        name: impl AsRef<str>,
        tensors: impl IntoIterator<Item = KernelTensorOf<'a, VM>>,
        attributes: impl IntoIterator<Item = &'static dyn Any>,
    );
}

pub struct Context<VM: VirtualMachine> {
    path: String,
    domain: VM::Domain,
    weights: HashMap<String, (NNTensor<VM>, Option<NNTensor<VM>>)>,
    forward: Vec<ForwardNode<VM>>,
    backward: Option<BackwardBuilder<VM>>,
}

impl<VM: VirtualMachine> Context<VM> {
    pub fn save_weight(&mut self, name: &str, weight: NNTensor<VM>) {
        let name = format!("{}.{name}", self.path);
        assert!(self.weights.insert(name, (weight, None)).is_none())
    }

    pub fn load_weight(&self, name: &str) -> Option<NNTensor<VM>> {
        let name = format!("{}.{name}", self.path);
        let (weight, _gradient) = self.weights.get(&name)?;
        Some(weight.clone())
    }

    pub fn load_gradient(&mut self, name: &str) -> NNTensor<VM> {
        let name = format!("{}.{name}", self.path);
        let (weight, gradient) = self.weights.get_mut(&name).unwrap();
        if let Some(gradient) = gradient {
            gradient.clone()
        } else {
            let tensor = self.domain.tensor(weight.dt(), weight.shape());
            let mut tensor = tensor::NNTensor::from(tensor);
            tensor.unlock();
            let _ = gradient.insert(tensor.clone());
            tensor
        }
    }

    pub fn init<NN: NeuralNetwork<VM>>(&mut self, name: impl AsRef<str>, init: NN::Init) {
        let name = name.as_ref();

        self.path.push('.');
        self.path.push_str(name);

        NN::init(init, self);

        self.path.truncate(self.path.len() - 1 - name.len());
    }

    pub fn forward<NN: NeuralNetwork<VM>>(
        &mut self,
        name: impl AsRef<str>,
        args: NN::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
    ) -> Vec<NNTensor<VM>> {
        let name = name.as_ref();

        self.path.push('.');
        self.path.push_str(name);

        let outputs = NN::forward(&args, inputs, self);

        self.path.truncate(self.path.len() - 1 - name.len());
        outputs
    }

    pub fn call<Op: Operator<VM>>(
        &mut self,
        args: Op::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
    ) -> Vec<NNTensor<VM>> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        let inputs_id = inputs.iter().map(tensor::NNTensor::id).collect::<Vec<_>>();

        let outputs = Op::call(&args, inputs, &self.domain);
        let outputs_id = outputs.iter().map(tensor::NNTensor::id).collect::<Vec<_>>();

        self.forward.push(ForwardNode {
            node: Box::new(OpNode::new::<Op>(args)),
            inputs: inputs_id,
            outptus: outputs_id,
        });

        outputs
    }

    pub fn take_backward_builder(&mut self) -> Option<BackwardBuilder<VM>> {
        self.backward.take()
    }

    pub fn put_backward_builder(&mut self, bb: BackwardBuilder<VM>) {
        assert!(self.backward.replace(bb).is_none())
    }
}

pub trait Node<VM: VirtualMachine> {
    fn compute(&self, inputs: Vec<NNTensor<VM>>, domain: &VM::Domain) -> Vec<NNTensor<VM>>;
}

pub struct OpNode<VM: VirtualMachine>(
    Box<dyn Fn(Vec<NNTensor<VM>>, &VM::Domain) -> Vec<NNTensor<VM>>>,
);

impl<VM: VirtualMachine> Node<VM> for OpNode<VM> {
    fn compute(&self, inputs: Vec<NNTensor<VM>>, domain: &VM::Domain) -> Vec<NNTensor<VM>> {
        (self.0)(inputs, domain)
    }
}

impl<VM: VirtualMachine> OpNode<VM> {
    pub fn new<Op: Operator<VM>>(args: Op::Args) -> Self {
        Self(Box::new(move |inputs, domain| {
            Op::call(&args, inputs, domain)
        }))
    }
}

#[repr(transparent)]
pub struct BackwardBuilder<VM: VirtualMachine>(Vec<BackwardNode<VM>>);

impl<VM: VirtualMachine> BackwardBuilder<VM> {
    pub fn call<Op: Operator<VM>>(
        &mut self,
        args: Op::Args,
        inputs: impl IntoIterator<Item = BackwardTensorOf<VM>>,
        outputs: impl IntoIterator<Item = BackwardTensorOf<VM>>,
    ) {
        self.0.push(BackwardNode {
            node: Box::new(OpNode::new::<Op>(args)),
            inputs: inputs.into_iter().collect(),
            outptus: outputs.into_iter().collect(),
        });
    }
}

pub struct ForwardNode<VM: VirtualMachine> {
    node: Box<dyn Node<VM>>,
    inputs: Vec<NNTensorId>,
    outptus: Vec<NNTensorId>,
}

pub struct BackwardNode<VM: VirtualMachine> {
    node: Box<dyn Node<VM>>,
    inputs: Vec<BackwardTensorOf<VM>>,
    outptus: Vec<BackwardTensorOf<VM>>,
}
