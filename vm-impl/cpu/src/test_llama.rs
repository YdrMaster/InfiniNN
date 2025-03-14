use crate::CpuVM;
use core::slice;
use digit_layout::types;
use ggus::{GGmlTokenType, GGuf, GGufMetaMapExt, ggml_quants::f16};
use memmap2::Mmap;
use nn::{
    VirtualMachineExt, WeightBiasData,
    lm_output::{self, LmOutput},
    mlp,
    normalization::{Normalization, Type},
    self_attn,
    token_embed::{self, TokenEmbed},
    transformer::{self, Repeat, Transformer},
    transformer_blk::{self, TransformerBlk},
};
use std::{env::var_os, fs::File, io::Write, ops::Deref, slice::from_raw_parts, sync::Arc};
use tokeneer::{Bpe, Tokeneer};
use vm::{VirtualMachine, op::RotaryType};

#[test]
fn test() {
    let Some(path) = var_os("TEST_MODEL") else {
        return;
    };
    let file = File::open(path).unwrap();
    let file = Arc::new(unsafe { Mmap::map(&file) }.unwrap());
    let gguf = GGuf::new(&file).unwrap();

    assert_eq!(gguf.general_architecture().unwrap(), "llama");

    // tokenization

    let vocabs = gguf.tokenizer_ggml_tokens().unwrap();
    let ntok = vocabs.len();
    let vocabs = vocabs.map(|t| t.unwrap());

    let scores = gguf.tokenizer_ggml_scores().unwrap();
    let scores = scores.map(|t| t.unwrap());

    let is_byte = gguf.tokenizer_ggml_token_type().unwrap();
    let is_byte = is_byte.map(|t| t.unwrap() == GGmlTokenType::Byte as i32);

    let unk = gguf.tokenizer_ggml_unknown_token_id().unwrap();

    let tokenizer = Tokeneer::new(Bpe::new(vocabs, scores, is_byte, unk));
    let text = "▁Once▁upon▁a▁time,";
    let tokens = tokenizer.encode(text);

    println!("tokens: {tokens:?}");

    let vm = CpuVM::default();

    // token embedding

    let text_embed = vm.register("test-embed");

    let data = Data::mmap(&file, &gguf, "token_embd.weight");
    vm.init::<TokenEmbed>(text_embed, 0, token_embed::Data { embed_table: data });

    // llama

    let llama = vm.register("llama");

    let nblk = gguf.llm_block_count().unwrap();
    let nctx = gguf.llm_context_length().unwrap();
    let nh = gguf.llm_attention_head_count().unwrap();
    let nkvh = gguf.llm_attention_head_count_kv().unwrap();
    let d = gguf.llm_embedding_length().unwrap();
    let dh = gguf.llm_rope_dimension_count().unwrap();
    let di = gguf.llm_feed_forward_length().unwrap();
    let theta = gguf.llm_rope_freq_base().unwrap();
    let epsilon = gguf.llm_attention_layer_norm_rms_epsilon().unwrap();

    let [sin, cos] = RotaryType::Normal { theta }.generate(nctx, dh);
    let sin = share_raw(&sin.into_iter().map(f16::from_f32).collect::<Vec<_>>());
    let cos = share_raw(&cos.into_iter().map(f16::from_f32).collect::<Vec<_>>());

    let data = (0..nblk)
        .map(|i| transformer_blk::Data {
            pre_norm: Data::mmap(&file, &gguf, &format!("blk.{i}.attn_norm.weight")).as_weight(),
            self_attn: self_attn::Data {
                qkv: Data::mmap(&file, &gguf, &format!("blk.{i}.attn_qkv.weight")).as_weight(),
                rope: Some([Data::share(sin.clone()), Data::share(cos.clone())]),
                output: Data::mmap(&file, &gguf, &format!("blk.{i}.attn_output.weight"))
                    .as_weight(),
            },
            post_norm: Data::mmap(&file, &gguf, &format!("blk.{i}.ffn_norm.weight")).as_weight(),
            mlp: mlp::Data {
                up: Data::mmap(&file, &gguf, &format!("blk.{i}.ffn_gate_up.weight")).as_weight(),
                down: Data::mmap(&file, &gguf, &format!("blk.{i}.ffn_down.weight")).as_weight(),
            },
        })
        .collect();
    vm.init::<Transformer<Repeat<TransformerBlk>>>(llama, 0, data);

    // lm output

    let lm_output = vm.register("lm-output");

    let data = lm_output::Data {
        norm: Data::mmap(&file, &gguf, "output_norm.weight").as_weight(),
        head: Data::mmap(&file, &gguf, "output.weight"),
    };
    vm.init::<LmOutput>(lm_output, 0, data);

    // forward

    let mut tokens = tokens;
    let mut pos = 0;
    let kv_cache = vm.workspace(types::F16, &[nctx, nblk, 2, nkvh, dh]);
    loop {
        let token = vm.workspace(types::U32, &[tokens.len()]);
        unsafe {
            std::ptr::copy_nonoverlapping(
                tokens.as_ptr().cast::<u8>(),
                token.blob().as_ptr().cast_mut(),
                size_of_val(tokens.as_slice()),
            )
        }
        let embed = vm.workspace(types::F16, &[tokens.len(), d]);
        let logits = vm.workspace(types::F16, &[1, ntok]);
        vm.forward(
            text_embed,
            0,
            &TokenEmbed { ntok },
            token_embed::Args {
                embed: embed.clone(),
                token,
            },
        )
        .forward(
            llama,
            0,
            &Transformer::repeat(
                TransformerBlk::llama(types::F32, types::F16, nh, nkvh, dh, di, epsilon),
                nblk,
            ),
            transformer::Args {
                embed: embed.clone(),
                n_sin: nctx,
                n_cos: nctx,
                reqs: vec![transformer::Request {
                    kv_cache: kv_cache.clone(),
                    n_seq: tokens.len(),
                    pos,
                }],
            },
        )
        .forward(
            lm_output,
            0,
            &LmOutput {
                norm: Normalization {
                    ty: Type::RmsNorm { epsilon },
                    dt_w: types::F32,
                },
                dt_w: types::F16,
            },
            lm_output::Args {
                logit: logits.clone(),
                x: embed,
                requests: vec![lm_output::Request {
                    n_seq: tokens.len(),
                    n_out: 1,
                }],
            },
        );
        let logits =
            unsafe { std::slice::from_raw_parts(logits.blob().as_ptr().cast::<f16>(), ntok) };
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        print!("{}", tokenizer.decode(&[next]));
        std::io::stdout().flush().unwrap();
        pos += tokens.len();
        tokens = vec![next];
    }
}

enum Data {
    Generate(Arc<[u8]>),
    Mmap {
        _mmap: Arc<Mmap>,
        slice: &'static [u8],
    },
}

impl Data {
    fn mmap(mmap: &Arc<Mmap>, gguf: &GGuf, name: &str) -> Box<Self> {
        let info = gguf.tensors[name].to_info();
        let data = &gguf.data[info.offset()..][..info.nbytes()];
        let slice = unsafe { slice::from_raw_parts(data.as_ptr(), data.len()) };
        Box::new(Self::Mmap {
            _mmap: mmap.clone(),
            slice,
        })
    }

    fn share(data: Arc<[u8]>) -> Box<Self> {
        Box::new(Self::Generate(data))
    }

    fn as_weight(self: Box<Self>) -> WeightBiasData {
        WeightBiasData {
            weight: self,
            bias: None,
        }
    }
}

impl Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Generate(data) => data,
            Self::Mmap { slice, .. } => slice,
        }
    }
}

fn share_raw<T: Copy>(data: &[T]) -> Arc<[u8]> {
    unsafe { from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
        .to_vec()
        .into()
}
