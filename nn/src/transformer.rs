use crate::{
    Context, Mapping, NuralNetwork, self_attn,
    transformer_blk::{self, TransformerBlk},
};
use digit_layout::types;
use std::borrow::Cow;
use vm::{Id, Tensor, VirtualMachine};

#[derive(Clone)]
#[repr(transparent)]
pub struct Transformer<T>(pub T);

impl Transformer<Repeat<TransformerBlk>> {
    pub fn repeat(blk: TransformerBlk, n_blk: usize) -> Self {
        Self(Repeat(blk, n_blk))
    }
}

#[derive(Clone)]
pub struct Repeat<T>(T, usize);
pub struct RepeatIter<T>(T, usize, usize);

impl<T: Clone> IntoIterator for Repeat<T> {
    type Item = T;
    type IntoIter = RepeatIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        RepeatIter(self.0.clone(), self.1, 0)
    }
}

impl<T: Clone> Iterator for RepeatIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        let Self(item, count, idx) = self;
        if *idx < *count {
            *idx += 1;
            Some(item.clone())
        } else {
            None
        }
    }
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub embed: Tensor<'vm, VM>, // [n, d]
    pub n_sin: usize,
    pub n_cos: usize,
    pub reqs: Vec<Request<'vm, VM>>,
}

pub struct Request<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub kv_cache: Tensor<'vm, VM>, // [nbuf, nblk, 2, nkvh, dh]
    pub n_seq: usize,
    pub pos: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Sub(usize);

impl Id for Sub {
    fn name(&self) -> Cow<str> {
        "blk".into()
    }

    fn idx(&self) -> Option<usize> {
        Some(self.0)
    }
}

impl<VM, T> NuralNetwork<VM> for Transformer<T>
where
    VM: VirtualMachine + ?Sized + transformer_blk::Ops,
    T: IntoIterator<Item = TransformerBlk> + Clone,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Data = Vec<transformer_blk::Data>;
    type Obj = ();
    type Sub = Sub;

    fn init(data: Self::Data, mut mapping: Mapping<VM, Self>) {
        for (i, blk) in data.into_iter().enumerate() {
            mapping.trap::<TransformerBlk>(Sub(i), blk);
        }
    }

    fn forward(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let Args {
            embed,
            n_sin,
            n_cos,
            reqs,
        } = args;

        let mut pos = vec![];
        for req in &reqs {
            for i in 0..req.n_seq {
                pos.push((req.pos + i) as u32)
            }
        }
        let mut pos_ = vec![0u8; size_of_val(pos.as_slice())];
        unsafe { std::ptr::copy_nonoverlapping(pos.as_ptr().cast(), pos_.as_mut_ptr(), pos_.len()) }
        let pos = ctx.map_host(types::U32, &[pos.len()], Box::new(pos_));

        for (i, blk) in self.0.clone().into_iter().enumerate() {
            ctx.trap(
                Sub(i),
                &blk,
                transformer_blk::Args {
                    embed: embed.clone(),
                    pos: pos.clone(),
                    n_sin,
                    n_cos,
                    reqs: reqs
                        .iter()
                        .map(|req| {
                            let kv_cache = req.kv_cache.clone().index(1, i);
                            self_attn::Request {
                                k_cache: kv_cache.clone().index(1, 0),
                                v_cache: kv_cache.index(1, 1),
                                n_seq: req.n_seq,
                                pos: req.pos,
                            }
                        })
                        .collect(),
                },
            );
        }
    }
}
