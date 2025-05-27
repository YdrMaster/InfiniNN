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

    pub fn check_eq(&mut self, other: &Self) -> bool {
        if self.expr == other.expr {
            true
        } else if self.expr != other.expr {
            false
        } else {
            self.eq_constraints
                .push(self.expr.clone() - other.expr.clone());
            true
        }
    }
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
