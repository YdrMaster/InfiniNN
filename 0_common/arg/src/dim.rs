//! 简单的符号运算系统，用于将形状符号化。
//!
//! 考虑到形状运算的实际情况，只支持多项式的运算。

use std::collections::{BTreeSet, HashMap};
use symbolic_expr::Expr;

/// 形状的一个维度，或参与维度运算的值。
///
/// ```rust
/// # use std::collections::HashMap;
/// # use arg::Dim;
/// let a = Dim::from("a");
/// let b = Dim::from("b");
/// let _1 = Dim::from(1);
/// let expr = (a + _1 - 2) * 3 / (b + 1);
/// assert_eq!(expr.substitute(&HashMap::from([("a", 8), ("b", 6)])), Some(3));
/// ```
#[derive(Clone, Debug)]
pub struct Dim {
    expr: Expr,
    eq_constraints: Vec<Expr>,
}

impl Dim {
    /// 统计表达式中出现的变量名。
    pub fn append_variables<'s>(&'s self, set: &mut BTreeSet<&'s str>) {
        self.expr.append_variables(set);
    }

    pub fn substitute(&self, value: &HashMap<&str, usize>) -> Option<usize> {
        if self
            .eq_constraints
            .iter()
            .any(|constraint| constraint.substitute(value) != 0)
        {
            None
        } else {
            Some(self.expr.substitute(value))
        }
    }

    pub fn to_usize(&self) -> usize {
        match self.expr {
            Expr::Constant(c) => c,
            _ => panic!("Dim is not a constant"),
        }
    }
}

/// 从多个 `Dim` 引用创建一个带有相等约束的新 `Dim`。
///
/// 此函数接收多个应该相等的 `Dim` 表达式，并生成一个新的带有相等约束的 `Dim`。
/// 返回的 `Dim` 将继承第一个输入的表达式，并添加确保所有输入相等的约束。
///
/// # 参数
///
/// * `dims` - 一个包含多个应该相等的 `Dim` 引用的切片
///
/// # 返回值
///
/// * `Some(Dim)` - 如果表达式可以相等；如果表达式恒等，返回一个不带约束的`Dim`，否则，返回一个带有相等约束的新 `Dim`，相等约束会在substitute时被计算
/// * `None` - 如果表达式被判定为永远不相等
///
/// # Panic
///
/// 当 `dims` 长度小于 2 时会发生 panic
pub fn make_eq(dims: &[&Dim]) -> Option<Dim> {
    assert!(dims.len() > 1);
    let mut dim = dims[0].clone();
    for other in dims[1..].iter() {
        let eq = dim.expr.equivalent(&other.expr);
        match eq {
            Some(true) => continue,
            Some(false) => return None,
            None => {
                dim.eq_constraints
                    .push(dim.expr.clone() - other.expr.clone());
            }
        }
    }
    Some(dim)
}

impl PartialEq for Dim {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }

    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, other: &Self) -> bool {
        self.expr != other.expr
    }
}

macro_rules! impl_ {
    (from: $ty:ty) => {
        impl From<$ty> for Dim {
            fn from(value: $ty) -> Self {
                Self {
                    expr: value.into(),
                    eq_constraints: Vec::new(),
                }
            }
        }
    };

    (op: $trait:ident, $fn:ident) => {
        impl std::ops::$trait for Dim {
            type Output = Self;
            fn $fn(self, rhs: Self) -> Self::Output {
                Self {
                    expr: self.expr.$fn(rhs.expr),
                    eq_constraints: Vec::new(),
                }
            }
        }
    };

    (num-op: $trait:ident, $fn:ident) => {
        impl std::ops::$trait<usize> for Dim {
            type Output = Self;
            fn $fn(self, rhs: usize) -> Self::Output {
                self.$fn(Self::from(rhs))
            }
        }
    };
}

impl_!(from: usize);
impl_!(from: &str);
impl_!(from: String);
impl_!(op: Add, add);
impl_!(op: Sub, sub);
impl_!(op: Mul, mul);
impl_!(op: Div, div);
impl_!(num-op: Add, add);
impl_!(num-op: Sub, sub);
impl_!(num-op: Mul, mul);
impl_!(num-op: Div, div);
