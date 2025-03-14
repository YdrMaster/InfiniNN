use digit_layout::types;
use vm::{ObjId, Tensor, op::TokenEmbed};

impl TokenEmbed for crate::CpuVM {
    fn token_embed(
        &self,
        _stack: ObjId,
        embd: &mut Tensor<Self>,
        tok: &Tensor<Self>,
        table: &Tensor<Self>,
    ) {
        let dt = Tensor::check_dt_same(&[embd, table]).unwrap();
        let [n, d] = embd.shape() else { panic!() };
        let [n_] = tok.shape() else { panic!() };
        let [_, d_] = table.shape() else { panic!() };
        assert_eq!(n, n_);
        assert_eq!(d, d_);

        match tok.dt() {
            types::U32 => {
                let ([], tok, []) = (unsafe { tok.blob().align_to::<u32>() }) else {
                    panic!()
                };

                let line = d * dt.nbytes();
                let embd = &**embd.blob();
                let table = &**table.blob();
                for (i, &tok) in tok.iter().enumerate() {
                    let dst = &embd[i * line..];
                    let src = &table[tok as usize * line..];
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr().cast_mut(), line)
                    }
                }
            }
            _ => todo!(),
        }
    }
}
