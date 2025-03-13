use crate::{Context, Mapping, NuralNetwork};
use std::ops::Deref;
use vm::{Tensor, VirtualMachine, op};

pub struct TokenEmbed {
    pub ntok: usize,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub embed: Tensor<'vm, VM>,
    pub token: Tensor<'vm, VM>,
}

pub struct Data {
    pub embed_table: Box<dyn Deref<Target = [u8]>>,
}

pub trait Ops: op::TokenEmbed {}
impl<VM> Ops for VM where VM: op::TokenEmbed + ?Sized {}

impl<VM> NuralNetwork<VM> for TokenEmbed
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Data = Data;
    type Obj = ();
    type Sub = ();

    fn init(data: Self::Data, mut mapping: Mapping<VM, Self>) {
        let Self::Data { embed_table } = data;
        mapping.map_host((), embed_table);
    }

    fn forward(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {
        let &Self { ntok } = self;
        let Args {
            embed: mut embd,
            token: tok,
        } = args;
        let &[_, d] = embd.shape() else { panic!() };

        let table = ctx.get_mapped((), embd.dt(), &[ntok, d]);
        ctx.token_embed(&mut embd, &tok, &table)
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Data, TokenEmbed};
    use crate::VirtualMachineExt;
    use digit_layout::{DigitLayout, types as ty};
    use test_vm::{TestVM, test_data};
    use vm::{VirtualMachine, device_id};

    const DT: DigitLayout = ty::F16;
    const DT_TOK: DigitLayout = ty::U32;
    const DEVICE: device_id = 0;
    const N_TOK: usize = 32000;
    const D: usize = 2048;
    const N: usize = 11;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("token_embed");

        vm.init::<TokenEmbed>(
            pid,
            DEVICE,
            Data {
                embed_table: test_data(DT, &[N_TOK, D]),
            },
        )
        .forward(
            pid,
            DEVICE,
            &TokenEmbed { ntok: N_TOK },
            Args {
                embed: vm.workspace(DT, &[N, D]),
                token: vm.workspace(DT_TOK, &[N]),
            },
        );

        vm.unregister(pid)
    }
}
