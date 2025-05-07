mod blob;
mod gguf;
mod model;

use gguf::{GGufModel, map_files};
use ggus::ggml_quants::digit_layout::types;
use nn::{Dim, Exec, GraphBuilder, Node, TensorMeta, op};
use std::time::Instant;

// cargo run --release -- ../TinyStory-5M-v0.0-F32.gguf
fn main() {
    let mut timer = TimeCollector::default();

    let path = std::env::args_os().nth(1).unwrap();
    let maps = map_files(path);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    let model = model::init(&mut gguf);
    timer.push("init");

    let graph = GraphBuilder::default()
        .register_op("embedding", op::embedding::Embedding)
        .register_op("rms-norm", op::normalization::RmsNorm)
        .register_op("layer-norm", op::normalization::LayerNorm)
        .register_op("attention", op::attention::Attention)
        .register_op("split", op::split::Split)
        .register_op("swiglu", op::activation::SwiGLU)
        .register_op("gelu", op::activation::GeLU)
        .register_op("linear", op::linear::Linear)
        .register_op("rope", op::rope::Rope)
        .register_op("concat", op::concat::Concat)
        .build(
            model,
            [
                TensorMeta::new(types::U32, [Dim::var("n")]),
                TensorMeta::new(types::U32, [Dim::var("n")]),
                TensorMeta::new(types::U32, [1.into()]),
            ],
        )
        .unwrap();
    timer.push("build");

    let graph = graph.lower(&[("n", 5)].into(), |name| gguf.tensors[&*name].as_ref());
    timer.push("fix shape");

    let mem_range_map = graph.mem_range_map(20 << 30, 512);
    timer.push("alloc");

    let mut _workspace = vec![0u8; mem_range_map.range.len()];
    let exec = graph
        .lower(
            |key| unsafe { _workspace.as_ptr().byte_add(mem_range_map.map[&key].start) },
            |data| data.as_ptr(),
        )
        .into_exec();
    timer.push("into exec");

    println!("{timer}");

    for Exec { node, .. } in exec {
        let Node { op, .. } = node;
        match op {
            _ => {}
        }
    }
}

#[derive(Default)]
#[repr(transparent)]
struct TimeCollector(Vec<(String, Instant)>);

impl TimeCollector {
    pub fn push(&mut self, name: impl std::fmt::Display) {
        self.0.push((name.to_string(), Instant::now()))
    }
}

impl std::fmt::Display for TimeCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name_width = self.0.iter().map(|(name, _)| name.len()).max().unwrap_or(0) + 2;
        for i in 1..self.0.len() {
            writeln!(
                f,
                "{:·<name_width$}{:?}",
                self.0[i].0,
                self.0[i].1 - self.0[i - 1].1
            )?
        }
        Ok(())
    }
}
