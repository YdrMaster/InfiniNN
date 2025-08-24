use crate::{
    blob::{Blob, Data},
    gguf::GGufModel,
    meta,
};
use ggus::GGufMetaMapExt;
use nn::{SelectiveSSM, Tensor};
use tensor::digit_layout::types;

pub fn init(gguf: &mut GGufModel) -> nn::LLaMA<String> {
    let arch = meta![gguf => general_architecture];
    let dt_bias = match arch {
        "llama" => None,
        "qwen2" => Some(gguf.tensors["blk.0.attn_qkv.bias"].dt()),
        "qwen3" => None,
        arch => panic!("unsupported arch {arch}"),
    };

    let nvoc = meta![gguf => tokenizer_ggml_tokens].len();
    let nctx = meta![gguf => llm_context_length];
    let nblk = meta![gguf => llm_block_count];
    let d = meta![gguf => llm_embedding_length];
    let nh = meta![gguf => llm_attention_head_count];
    let nkvh = meta![gguf => llm_attention_head_count_kv; nh];
    let dh = match arch {
        "qwen3" => gguf.tensors["blk.0.attn_qkv.weight"].shape()[0]
            .checked_div(nh + nkvh + nkvh)
            .unwrap(),
        _ => meta![gguf => llm_rope_dimension_count; d / nh],
    };
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
            d,
            wte: ::nn::Table {
                row: nvoc,
                weight: "token_embd.weight".to_string(),
            },
            wpe: None,
        },
        blks: (0..nblk)
            .map(|iblk| {
                ::nn::TransformerBlk::new(
                    ::nn::Normalization {
                        d,
                        epsilon: epsilon as _,
                        items: ::nn::NormType::RmsNorm {
                            dt: dt_norm,
                            scale: format!("blk.{iblk}.attn_norm.weight"),
                        },
                    },
                    ::nn::Attention {
                        nh,
                        nkvh,
                        qkv: ::nn::Linear::new(
                            dt_linear,
                            [(nh + nkvh + nkvh) * dh, d],
                            format!("blk.{iblk}.attn_qkv.weight"),
                            dt_bias.map(|dt| (dt, format!("blk.{iblk}.attn_qkv.bias"))),
                        ),
                        q_norm: if gguf
                            .tensors
                            .contains_key(format!("blk.{iblk}.attn_q_norm.weight").as_str())
                        {
                            Some(::nn::Normalization {
                                d: dh,
                                epsilon: epsilon as _,
                                items: ::nn::NormType::RmsNorm {
                                    dt: dt_norm,
                                    scale: format!("blk.{iblk}.attn_q_norm.weight"),
                                },
                            })
                        } else {
                            None
                        },
                        k_norm: if gguf
                            .tensors
                            .contains_key(format!("blk.{iblk}.attn_k_norm.weight").as_str())
                        {
                            Some(::nn::Normalization {
                                d: dh,
                                epsilon: epsilon as _,
                                items: ::nn::NormType::RmsNorm {
                                    dt: dt_norm,
                                    scale: format!("blk.{iblk}.attn_k_norm.weight"),
                                },
                            })
                        } else {
                            None
                        },
                        rope: Some(::nn::RoPE {
                            multimodal: false,
                            nctx,
                            sin: "sin_table".into(),
                            cos: "cos_table".into(),
                        }),
                        output: ::nn::Linear::new(
                            dt_linear,
                            [d, nh * dh],
                            format!("blk.{iblk}.attn_output.weight"),
                            None,
                        ),
                    },
                    ::nn::Normalization {
                        d,
                        epsilon: epsilon as _,
                        items: ::nn::NormType::RmsNorm {
                            dt: dt_norm,
                            scale: format!("blk.{iblk}.ffn_norm.weight"),
                        },
                    },
                    ::nn::Mlp {
                        up: ::nn::Linear::new(
                            dt_linear,
                            [di * 2, d],
                            format!("blk.{iblk}.ffn_gate_up.weight"),
                            None,
                        ),
                        act: ::nn::Activation::SwiGLU,
                        down: ::nn::Linear::new(
                            dt_linear,
                            [d, di],
                            format!("blk.{iblk}.ffn_down.weight"),
                            None,
                        ),
                    },
                )
            })
            .collect(),
        output_head: Some(::nn::OutputHead {
            out_norm: ::nn::Normalization {
                d,
                epsilon: epsilon as _,
                items: ::nn::NormType::RmsNorm {
                    dt: dt_norm,
                    scale: "output_norm.weight".into(),
                },
            },
            lm_head: ::nn::Linear::new(
                dt_linear,
                [nvoc, d],
                if gguf.tensors.contains_key("output.weight") {
                    "output.weight"
                } else {
                    "token_embd.weight"
                }
                .into(),
                None,
            ),
        }),
    }
}

#[allow(dead_code)]
pub fn init_mamba(gguf: &mut GGufModel) -> nn::Mamba<String> {
    let nvoc = meta![gguf => tokenizer_ggml_tokens].len();
    let nblk = meta![gguf => llm_block_count];
    let d = meta![gguf => llm_embedding_length];
    let epsilon = meta![gguf => llm_attention_layer_norm_rms_epsilon; 1e-5];
    let dt_embd = gguf.tensors["token_embd.weight"].dt();
    let dt_norm = gguf.tensors["output_norm.weight"].dt();
    let dt_linear = gguf.tensors["blk.0.ssm_in.weight"].dt();

    let d_kernel = 4;
    let d_inner = 5120;
    let d_state = 16;
    let dt_rank = 160; // ggus todo: mamba.ssm.

    ::nn::Mamba {
        embedding: ::nn::Embedding {
            dt: dt_embd,
            d,
            wte: ::nn::Table {
                row: nvoc,
                weight: "token_embd.weight".to_string(),
            },
            wpe: None,
        },
        blks: (0..nblk)
            .map(|iblk| ::nn::MambaBlock {
                mamba_norm: nn::Normalization {
                    d,
                    epsilon: epsilon as _,
                    items: ::nn::NormType::RmsNorm {
                        dt: dt_norm,
                        scale: format!("blk.{iblk}.attn_norm.weight"),
                    },
                },
                mamba_mixer: nn::MambaMixer {
                    d_inner,
                    in_proj: nn::Linear {
                        dt: dt_embd,
                        shape: [d_inner * 2, d],
                        weight: format!("blk.{iblk}.ssm_in.weight"),
                        bias: None,
                        allow_residual: false,
                    },
                    causal_conv1d: nn::CausalConv1d::new(
                        dt_norm,
                        format!("blk.{iblk}.ssm_conv1d.weight"),
                        format!("blk.{iblk}.ssm_conv1d.bias"),
                        d_kernel,
                        d_inner,
                    ),
                    act: nn::Activation::SiLU,
                    selective_ssm: SelectiveSSM {
                        dt: dt_norm,
                        d_state,
                        dt_rank,
                        dt_proj: nn::Linear {
                            dt: dt_linear,
                            shape: [d_inner, dt_rank],
                            weight: format!("blk.{iblk}.ssm_dt.weight"),
                            bias: Some(dt_linear).map(|dt| (dt, format!("blk.{iblk}.ssm_dt.bias"))),
                            allow_residual: false,
                        },
                        x_proj: nn::Linear {
                            dt: dt_linear,
                            shape: [dt_rank + d_state * 2, d_inner],
                            weight: format!("blk.{iblk}.ssm_x.weight"),
                            bias: None,
                            allow_residual: false,
                        },
                        a: format!("blk.{iblk}.ssm_a"),
                        d: format!("blk.{iblk}.ssm_d"),
                    },
                    out_proj: nn::Linear {
                        dt: dt_linear,
                        shape: [d, d_inner],
                        weight: format!("blk.{iblk}.ssm_out.weight"),
                        bias: None,
                        allow_residual: true,
                    },
                },
            })
            .collect(),
        output_head: Some(::nn::OutputHead {
            out_norm: ::nn::Normalization {
                d,
                epsilon: epsilon as _,
                items: ::nn::NormType::RmsNorm {
                    dt: dt_norm,
                    scale: "output_norm.weight".into(),
                },
            },
            lm_head: ::nn::Linear::new(
                dt_linear,
                [nvoc, d],
                if gguf.tensors.contains_key("output.weight") {
                    "output.weight"
                } else {
                    "token_embd.weight"
                }
                .into(),
                None,
            ),
        }),
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
