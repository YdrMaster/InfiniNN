use crate::{
    blob::{Blob, Data},
    gguf::GGufModel,
    meta,
};
use ggus::GGufMetaMapExt;
use nn::Tensor;
use tensor::digit_layout::types;

pub fn init(gguf: &mut GGufModel) -> nn::LLaMA<String> {
    let arch = meta![gguf => general_architecture];
    let dt_bias = match arch {
        "llama" => None,
        "qwen2" => Some(gguf.tensors["blk.0.attn_qkv.bias"].dt()),
        arch => panic!("unsupported arch {arch}"),
    };

    let nvoc = meta![gguf => tokenizer_ggml_tokens].len();
    let nctx = meta![gguf => llm_context_length];
    let nblk = meta![gguf => llm_block_count];
    let d = meta![gguf => llm_embedding_length];
    let nh = meta![gguf => llm_attention_head_count];
    let nkvh = meta![gguf => llm_attention_head_count_kv; nh];
    let dh = meta![gguf => llm_rope_dimension_count; d / nh];
    let di = meta![gguf => llm_feed_forward_length];
    let epsilon = meta![gguf => llm_attention_layer_norm_rms_epsilon; 1e-5];
    let dt_embd = gguf.tensors["token_embd.weight"].dt();
    let dt_norm = gguf.tensors["output_norm.weight"].dt();
    let dt_linear = gguf.tensors["blk.0.attn_qkv.weight"].dt();
    let theta = meta![gguf => llm_rope_freq_base; 1e4];

    let [sin, cos] = build_sin_cos(nctx, dh, theta);
    gguf.tensors.insert("sin_table", sin);
    gguf.tensors.insert("cos_table", cos);

    ::nn::LLaMA {
        embedding: ::nn::Embedding {
            dt: dt_embd,
            d: d.into(),
            wte: ::nn::Table {
                row: nvoc.into(),
                weight: "token_embd.weight".to_string(),
            },
            wpe: None,
        },
        blks: (0..nblk)
            .map(|iblk| ::nn::TransformerBlk {
                attn_norm: ::nn::Normalization {
                    d: d.into(),
                    epsilon: epsilon as _,
                    items: ::nn::NormType::RmsNorm {
                        dt: dt_norm,
                        scale: format!("blk.{iblk}.attn_norm.weight"),
                    },
                },
                attn: ::nn::Attention {
                    nh: nh.into(),
                    nkvh: nkvh.into(),
                    qkv: ::nn::Linear {
                        dt: dt_linear,
                        shape: [((nh + nkvh + nkvh) * dh).into(), d.into()],
                        weight: format!("blk.{iblk}.attn_qkv.weight"),
                        bias: dt_bias
                            .clone()
                            .map(|dt| (dt, format!("blk.{iblk}.attn_qkv.bias"))),
                    },
                    rope: Some(::nn::RoPE {
                        nctx: nctx.into(),
                        sin: "sin_table".into(),
                        cos: "cos_table".into(),
                    }),
                    output: ::nn::Linear {
                        dt: dt_linear,
                        shape: [d.into(), (nh * dh).into()],
                        weight: format!("blk.{iblk}.attn_output.weight"),
                        bias: None,
                    },
                },
                ffn_norm: ::nn::Normalization {
                    d: d.into(),
                    epsilon: epsilon as _,
                    items: ::nn::NormType::RmsNorm {
                        dt: dt_norm,
                        scale: format!("blk.{iblk}.ffn_norm.weight"),
                    },
                },
                ffn: ::nn::Mlp {
                    up: ::nn::Linear {
                        dt: dt_linear,
                        shape: [(di * 2).into(), d.into()],
                        weight: format!("blk.{iblk}.ffn_gate_up.weight"),
                        bias: None,
                    },
                    act: ::nn::Activation::SwiGLU,
                    down: ::nn::Linear {
                        dt: dt_linear,
                        shape: [d.into(), di.into()],
                        weight: format!("blk.{iblk}.ffn_down.weight"),
                        bias: None,
                    },
                },
            })
            .collect(),
        out_norm: ::nn::Normalization {
            d: d.into(),
            epsilon: epsilon as _,
            items: ::nn::NormType::RmsNorm {
                dt: dt_norm,
                scale: "output_norm.weight".into(),
            },
        },
        lm_head: ::nn::Linear {
            dt: dt_linear,
            shape: [nvoc.into(), d.into()],
            weight: if gguf.tensors.contains_key("output.weight") {
                "output.weight"
            } else {
                "token_embd.weight"
            }
            .into(),
            bias: None,
        },
    }
}

/// 构造 sin cos 表张量
fn build_sin_cos<'a, const N: usize>(
    nctx: usize,
    dh: usize,
    theta: f32,
) -> [Tensor<Data<'a>, N>; 2] {
    let ty = types::F32;
    let mut sin = Blob::new(nctx * dh / 2 * ty.nbytes());
    let mut cos = Blob::new(nctx * dh / 2 * ty.nbytes());

    {
        let ([], sin, []) = (unsafe { sin.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        let ([], cos, []) = (unsafe { cos.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        for pos in 0..nctx {
            for i in 0..dh / 2 {
                let theta = theta.powf(-((2 * i) as f32 / dh as f32));
                let freq = pos as f32 * theta;
                let (sin_, cos_) = freq.sin_cos();
                sin[pos * dh / 2 + i] = sin_;
                cos[pos * dh / 2 + i] = cos_;
            }
        }
    }

    let tensor = |data: Blob| {
        Tensor::from_dim_slice(ty, [nctx, dh / 2]).map(|len| {
            assert_eq!(len, data.len());
            data.into()
        })
    };
    [tensor(sin), tensor(cos)]
}
