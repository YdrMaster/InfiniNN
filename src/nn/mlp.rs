use super::{NuralNetwork, def};
use crate::{
    Context, Tensor, VirtualMachine,
    op::{Add, GeLU, MatMul, Rearrange, SwiGLU},
    split,
};
use digit_layout::DigitLayout;

pub struct Mlp {
    ty: Type,
    dt_w: DigitLayout,
    di: usize,
    up_bias: bool,
    down_bias: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Type {
    SwiGLU,
    GeLU,
}

def!(Args: <mut: y, x> scale: f32, residual: bool);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Obj {
    Wup,
    Wdown,
    Bup,
    Bdown,
}

pub trait Ops: Rearrange + MatMul + SwiGLU + GeLU + Add {}
impl<VM> Ops for VM where VM: Rearrange + MatMul + SwiGLU + GeLU + Add {}

impl<VM> NuralNetwork<VM> for Mlp
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'ctx, 'vm: 'ctx>
        = Args<'ctx, 'vm, VM>
    where
        VM: 'vm;
    type Obj = Obj;
    type Sub = ();

    fn launch(&self, args: Self::Args<'_, '_>, mut ctx: Context<VM, Self>) {
        let &Self {
            ty,
            dt_w,
            di,
            up_bias,
            down_bias,
        } = self;
        let Args {
            y,
            x,
            scale,
            residual,
        } = args;

        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());
        let &[n, d] = y.shape() else { panic!() };

        let d_up = match ty {
            Type::SwiGLU => di * 2,
            Type::GeLU => di,
        };
        let mut mid = ctx.workspace(y.dt(), &[n, d_up]);

        let beta = if up_bias {
            let b = ctx.get_mapped(Obj::Bup, dt_w, &[1, d_up]).broadcast(0, n);
            ctx.rearrange(&mut mid, &b);
            1.
        } else {
            0.
        };
        {
            let w = ctx
                .get_mapped(Obj::Wup, dt_w, &[d_up, d])
                .transpose(&[1, 0]);
            ctx.mat_mul(&mut mid, beta, x, &w, 1.)
        }

        match ty {
            Type::SwiGLU => {
                split!(mid => gate, up; [di, di] @ 1);
                mid = gate;
                ctx.swiglu(&mut mid, &up)
            }
            Type::GeLU => ctx.gelu(&mut mid),
        }

        let w = ctx
            .get_mapped(Obj::Wdown, dt_w, &[d, di])
            .transpose(&[1, 0]);
        if down_bias {
            {
                let x1 = if residual { &mut *x } else { &mut *y };
                {
                    let down_bias = ctx.get_mapped(Obj::Bdown, dt_w, &[1, d]).broadcast(0, n);
                    ctx.rearrange(x1, &down_bias)
                }
                ctx.mat_mul(x1, scale, &mid, &w, scale)
            }
            if residual {
                ctx.add(y, x)
            }
        } else {
            let beta = if residual { 1. } else { 0. };
            ctx.mat_mul(y, beta, &mid, &w, scale)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Mlp, Obj, Type};
    use crate::{Exec, Map, VirtualMachine, test::TestVM};
    use digit_layout::types as ty;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("norm");
        let device = 0;

        {
            let wup = vec![0u8; 1024 * 1536 * 2 * 2];
            let wdown = vec![0u8; 1024 * 1536 * 2];
            let mlp = vm.map::<Mlp>(pid, device);
            mlp.map_host(Obj::Wup, Box::new(wup));
            mlp.map_host(Obj::Wdown, Box::new(wdown));

            let mut y = vm.workspace(Some(device), ty::F16, &[7, 1024]);
            let mut x = vm.workspace(Some(device), ty::F16, &[7, 1024]);

            vm.exec(
                pid,
                0,
                &Mlp {
                    ty: Type::SwiGLU,
                    dt_w: ty::F16,
                    di: 1536,
                    up_bias: false,
                    down_bias: false,
                },
                Args {
                    y: &mut y,
                    x: &mut x,
                    scale: 1.,
                    residual: true,
                },
            )
        }

        vm.unregister(pid)
    }
}
