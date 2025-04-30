use super::{Context, Linear, NNError, NuralNetwork, Tensor, macros::*};
use crate::{Arg, Dim};
use itertools::izip;

pub struct Attention<T> {
    pub nh: Dim,
    pub nkvh: Dim,
    pub qkv: Linear<T>,
    pub rope: Option<RoPE<T>>,
    pub sessions: Box<[Session<T>]>,
    pub output: Linear<T>,
}

pub struct RoPE<T> {
    pub nctx: Dim,
    pub sin: T,
    pub cos: T,
}

pub struct Session<T> {
    pub seq: Dim,
    pub cache: Option<Cache<T>>,
}

pub struct Cache<T> {
    pub pos: Dim,
    pub items: [T; 2],
}

impl<T> NuralNetwork<T> for Attention<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x, pos] = inputs);

        let Self {
            nh,
            nkvh,
            qkv,
            rope,
            sessions,
            output,
        } = self;
        let residual = x.clone();

        destruct!([x] = ctx.trap("attn-qkv", qkv, [x])?);
        dims!([_, dqkv] = x);
        let dt = x.dt();
        let dh = dqkv.clone() / (nh.clone() + nkvh.clone() + nkvh.clone());

        destruct!(
            [q, k, v] = ctx.call(
                "split-qkv",
                "split",
                Some(Arg::dict([
                    ("axis".into(), Arg::int(1)),
                    (
                        "parts".into(),
                        Arg::arr([nh, nkvh.clone(), nkvh.clone()].map(Arg::from))
                    )
                ])),
                [x],
            )?
        );

        let [q, k] = match rope {
            Some(RoPE { nctx, sin, cos }) => {
                let shape = [nctx, dh.clone() / 2];
                let sin = ctx.load_external("rope.sin", dt, shape.clone(), sin);
                let cos = ctx.load_external("rope.cos", dt, shape.clone(), cos);
                destruct!(
                    [q_] = ctx.call(
                        "attn-q-rope",
                        "rope",
                        None,
                        [q, pos.clone(), sin.clone(), cos.clone()]
                    )?
                );
                destruct!([k_] = ctx.call("attn-k-rope", "rope", None, [k, pos, sin, cos])?);
                [q_, k_]
            }
            None => [q, k],
        };

        let session_split = Some(Arg::dict([
            ("axis".into(), Arg::int(0)),
            (
                "parts".into(),
                Arg::arr(sessions.iter().map(|s| Arg::Dim(s.seq.clone()))),
            ),
        ]));

        let q = ctx.call("", "split", session_split.clone(), [q])?;
        let k = ctx.call("", "split", session_split.clone(), [k])?;
        let v = ctx.call("", "split", session_split, [v])?;
        let o = izip!(q, k, v, sessions)
            .enumerate()
            .map(|(i, (q, k, v, s))| -> Result<Tensor<T>, NNError> {
                match s.cache {
                    Some(Cache { pos, items }) => {
                        let name_in = format!("session[{i}].kv-cache-in");
                        let [input, output] = items;
                        let shape_in = [pos + s.seq, nkvh.clone() * dh.clone()];
                        let cache = ctx.load_external(name_in, q.dt(), shape_in, input);
                        destruct!(
                            [o, cache] = ctx.call(
                                "",
                                "attention",
                                Some(dh.clone().into()),
                                [q, k, v, cache]
                            )?
                        );
                        let name_out = format!("session[{i}].kv-cache-out");
                        ctx.save_external(name_out, cache, output);
                        Ok(o)
                    }
                    None => {
                        destruct!(
                            [o] = ctx.call("", "attention", Some(dh.clone().into()), [q, k, v])?
                        );
                        Ok(o)
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        destruct!([o] = ctx.call("", "concat", Some(Arg::int(0)), o)?);

        let outputs = ctx.trap("attn-output", output, [o, residual]);

        Ok((ctx, outputs?))
    }
}
