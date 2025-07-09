use crate::macros::destruct;

use super::{Context, NNError, NuralNetwork, TPTensor, Tensor};
use tensor::digit_layout::DigitLayout;

#[derive(Clone)]
pub struct Embedding<T: Clone> {
    pub dt: DigitLayout,
    pub d: usize,
    pub wte: Table<T>,
    pub wpe: Option<Table<T>>,
}

#[derive(Clone)]
pub struct Table<T> {
    pub row: usize,
    pub weight: T,
}

impl<T: Clone> Embedding<T> {
    pub fn tensor_parallel(self) -> Embedding<TPTensor<T>> {
        let Self { dt, d, wte, wpe } = self;
        Embedding {
            dt,
            d,
            wte: Table {
                row: wte.row,
                weight: wte.weight.into(),
            },
            wpe: wpe.map(|Table { row, weight }| Table {
                row,
                weight: weight.into(),
            }),
        }
    }
}

impl<T: Clone> NuralNetwork<T> for Embedding<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { dt, d, wte, wpe } = self;
        let mut inputs = inputs.into_iter();

        let Table { row, weight } = wte;

        let tokens = inputs.next().unwrap();

        let outputs = if dt.group_size() > 1 {
            let w = ctx.load_external("weight", dt, [row.into(), d.into()], weight)?;
            match wpe {
                Some(_) => {
                    todo!()
                }
                None => {
                    let inputs = w.into_iter().chain([tokens]).collect::<Vec<_>>();
                    ctx.call("", "quant-embedding", Some(false.into()), inputs)
                }
            }
        } else {
            destruct!([wte] = ctx.load_external("wte", dt, [row.into(), d.into()], weight)?);

            match wpe {
                Some(wpe) => {
                    let Table { row, weight } = wpe;
                    destruct!(
                        [wpe] = ctx.load_external("wpe", dt, [row.into(), d.into()], weight)?
                    );
                    let pos = inputs.next().unwrap();
                    ctx.call("", "embedding", None, [wte, tokens, wpe, pos])
                }
                None => {
                    // format
                    ctx.call("", "embedding", None, [wte, tokens])
                }
            }
        };

        Ok((ctx, outputs?))
    }
}
