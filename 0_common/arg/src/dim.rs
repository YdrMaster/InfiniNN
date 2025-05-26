//! 简单的符号运算系统，用于将形状符号化。
//!
//! 考虑到形状运算的实际情况，只支持多项式的运算。

use std::{
    collections::{BTreeSet, HashMap},
    ops::{Add, Div, Mul, Sub},
};
use symbolic_expr::Expr;

/// 形状的一个维度，或参与维度运算的值。
///
/// ```rust
/// # use std::collections::HashMap;
/// # use arg::Dim;
/// let a = Dim::var("a");
/// let b = Dim::var("b");
/// let _1 = Dim::from(1);
/// let expr = (a + _1 - 2) * 3 / (b + 1);
/// assert_eq!(expr.substitute(&HashMap::from([("a", 8), ("b", 6)])), 3);
/// ```
#[derive(Clone, Debug)]
pub struct Dim{
    expr: Expr,
    eq_constraints: Vec<Expr>,
}

impl Default for Dim {
    fn default() -> Self {
        Self {
            expr: Expr::Constant(0),
            eq_constraints: vec![],
        }
    }
}

impl Dim {

    /// 统计表达式中出现的变量名。
    pub fn variables(&self) -> BTreeSet<&str> {
        self.expr.variables()
    }

    pub fn append_variables<'s>(&'s self, set: &mut BTreeSet<&'s str>) {
        self.expr.append_variables(set);
    }

    pub fn substitute(&self, value: &HashMap<&str, usize>) -> Option<usize> {
        if self.eq_constraints.iter().any(|constraint| {
            constraint.substitute(value) != 0
        }) {
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
            self.eq_constraints.push(self.expr.clone() - other.expr.clone());
            true
        }
    }
}

impl From<usize> for Dim {
    fn from(value: usize) -> Self {
        Self {
            expr: Expr::from(value),
            eq_constraints: vec![],
        }
    }
}

impl From<&str> for Dim {
    fn from(value: &str) -> Self {
        Self {
            expr: Expr::from(value),
            eq_constraints: vec![],
        }
    }
}

impl From<String> for Dim {
    fn from(value: String) -> Self {
        Self {
            expr: Expr::from(value),
            eq_constraints: vec![],
        }
    }
}

impl Add for Dim {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            expr: self.expr + rhs.expr,
            eq_constraints: vec![],
        }
    }
}

impl Sub for Dim {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            expr: self.expr - rhs.expr,
            eq_constraints: vec![],
        }
    }
}

impl Mul for Dim {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            expr: self.expr * rhs.expr,
            eq_constraints: vec![],
        }
    }
}

impl Div for Dim {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            expr: self.expr / rhs.expr,
            eq_constraints: vec![],
        }
    }
}

impl Add<usize> for Dim {
    type Output = Self;
    fn add(self, rhs: usize) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl Sub<usize> for Dim {
    type Output = Self;
    fn sub(self, rhs: usize) -> Self::Output {
        self - Self::from(rhs)
    }
}

impl Mul<usize> for Dim {
    type Output = Self;
    fn mul(self, rhs: usize) -> Self::Output {
        self * Self::from(rhs)
    }
}

impl Div<usize> for Dim {
    type Output = Self;
    fn div(self, rhs: usize) -> Self::Output {
        self / Self::from(rhs)
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


