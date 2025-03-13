use vm::{
    ObjId, Tensor,
    op::{AttnMask, Softmax},
};

impl Softmax for crate::TestVM {
    fn softmax(&self, stack: ObjId, att: &mut Tensor<Self>, mask: AttnMask) {
        assert_eq!(att.shape().len(), 3);

        let mask = match mask {
            AttnMask::None => "",
            AttnMask::Causal => ", causal",
        };
        self.launch(stack, format!("softmax(mut %{}{mask})", att.blob().id()))
    }
}
