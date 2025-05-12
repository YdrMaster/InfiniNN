use super::{Context, NNError, NuralNetwork, TPTensor, Tensor};
use digit_layout::DigitLayout;

#[derive(Clone)]
pub struct Embedding<T> {
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

impl<T> Embedding<T> {
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

impl<T> NuralNetwork<T> for Embedding<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { dt, d, wte, wpe } = self;
        let mut inputs = inputs.into_iter();

        let Table { row, weight } = wte;
        let wte = ctx.load_external("wte", dt, [row.into(), d.into()], weight.into());
        let tokens = inputs.next().unwrap();

        let outputs = match wpe {
            Some(wpe) => {
                let Table { row, weight } = wpe;
                let wpe = ctx.load_external("wpe", dt, [row.into(), d.into()], weight.into());
                let pos = inputs.next().unwrap();
                ctx.call("", "embedding", None, [wte, tokens, wpe, pos])
            }
            None => {
                // format
                ctx.call("", "embedding", None, [wte, tokens])
            }
        };

        Ok((ctx, outputs?))
    }
}
