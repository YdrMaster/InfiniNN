//! 简单的符号运算系统，用于将形状符号化。
//!
//! 考虑到形状运算的实际情况，只支持多项式的运算。

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};
use num_rational::Ratio;

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
pub enum Dim {
    /// 常量
    Constant(usize),
    /// 变量
    Variable(String),
    /// 和式
    Sum(VecDeque<Operand>),
    /// 积式
    Product(VecDeque<Operand>),
}

impl Default for Dim {
    fn default() -> Self {
        Self::Constant(0)
    }
}

impl Dim {
    /// 变量。
    pub fn var(symbol: impl Display) -> Self {
        Self::Variable(symbol.to_string())
    }

    /// 维度作为正操作数。
    pub fn positive(self) -> Operand {
        Operand {
            ty: Type::Positive,
            dim: self,
        }
    }

    /// 维度作为负操作数。
    pub fn negative(self) -> Operand {
        Operand {
            ty: Type::Negative,
            dim: self,
        }
    }

    /// 统计表达式中出现的变量名。
    pub fn variables(&self) -> BTreeSet<&str> {
        let mut ans = BTreeSet::new();
        self.append_variables(&mut ans);
        ans
    }

    /// 遍历表达式，递归地将变量名添加到集合。
    pub fn append_variables<'s>(&'s self, set: &mut BTreeSet<&'s str>) {
        match self {
            Self::Constant(_) => {}
            Self::Variable(name) => {
                set.insert(name);
            }
            Self::Sum(operands) | Self::Product(operands) => {
                operands.iter().for_each(|op| op.dim.append_variables(set))
            }
        }
    }

    pub fn substitute(&self, value: &HashMap<&str, usize>) -> usize {
        match self {
            &Self::Constant(value) => value,
            Self::Variable(name) => *value
                .get(&**name)
                .unwrap_or_else(|| panic!("unknown variable \"{name}\"")),
            Self::Sum(operands) => operands.iter().fold(0, |acc, Operand { ty, dim }| {
                let value = dim.substitute(value);
                match ty {
                    Type::Positive => acc + value,
                    Type::Negative => acc.checked_sub(value).unwrap(),
                }
            }),
            Self::Product(operands) => operands.iter().fold(1, |acc, Operand { ty, dim }| {
                let value = dim.substitute(value);
                match ty {
                    Type::Positive => acc * value,
                    Type::Negative => {
                        assert_eq!(acc % value, 0);
                        acc / value
                    }
                }
            }),
        }
    }

    /// Checks if two Dim expressions are mathematically equivalent.
    /// Returns:
    /// - Some(true) if expressions are definitely equivalent
    /// - Some(false) if expressions are definitely not equivalent
    /// - None if equivalence cannot be determined without substitution
    pub fn equivalent(&self, other: &Self) -> Option<bool> {
        // Convert both expressions to canonical form and compare
        let self_rational = RationalExpression::from_dim(self)?;
        let other_rational = RationalExpression::from_dim(other)?;

        // If both have denominator 1 and are not equal, they are definitely not equivalent
        if self_rational.denom == vec![CanonicalTerm::new(1)] && 
           other_rational.denom == vec![CanonicalTerm::new(1)] {
            if self_rational == other_rational {
            Some(true)
        } else {
            Some(false)
        }
        } else {
            // For other cases, we can only determine equivalence if they match exactly
            if self_rational == other_rational {
                Some(true)
            } else {
                None
            }
        }
    }
}

impl PartialEq for Dim {
    fn eq(&self, other: &Self) -> bool {
        self.equivalent(other) == Some(true)
    }

    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, other: &Self) -> bool {
        self.equivalent(other) == Some(false)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Type {
    Positive,
    Negative,
}

impl Type {
    pub fn rev(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Operand {
    ty: Type,
    dim: Dim,
}

impl Operand {
    pub fn rev_assign(&mut self) {
        self.ty = self.ty.rev()
    }
}

impl Neg for Operand {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let Self { ty, dim } = self;
        Self { ty: ty.rev(), dim }
    }
}

impl From<usize> for Dim {
    fn from(value: usize) -> Self {
        Dim::Constant(value)
    }
}

impl From<String> for Dim {
    fn from(value: String) -> Self {
        Dim::Variable(value)
    }
}

macro_rules! impl_op {
    ($op:ty; $fn:ident; positive: $variant: ident) => {
        impl $op for Dim {
            type Output = Self;
            fn $fn(self, rhs: Self) -> Self::Output {
                match self {
                    Dim::$variant(mut l) => match rhs {
                        Self::$variant(r) => {
                            l.extend(r);
                            Self::$variant(l)
                        }
                        r => {
                            l.push_back(r.positive());
                            Self::$variant(l)
                        }
                    },
                    l => match rhs {
                        Self::$variant(mut r) => {
                            r.push_front(l.positive());
                            Self::$variant(r)
                        }
                        r => Self::$variant([l.positive(), r.positive()].into()),
                    },
                }
            }
        }
    };

    ($op:ty; $fn:ident; negative: $variant: ident) => {
        impl $op for Dim {
            type Output = Self;
            fn $fn(self, rhs: Self) -> Self::Output {
                match self {
                    Dim::$variant(mut l) => match rhs {
                        Self::$variant(r) => {
                            l.extend(r.into_iter().map(Neg::neg));
                            Self::$variant(l)
                        }
                        r => {
                            l.push_back(r.negative());
                            Self::$variant(l)
                        }
                    },
                    l => match rhs {
                        Self::$variant(mut r) => {
                            r.iter_mut().for_each(Operand::rev_assign);
                            r.push_front(l.positive());
                            Self::$variant(r)
                        }
                        r => Self::$variant([l.positive(), r.negative()].into()),
                    },
                }
            }
        }
    };

    ($op:ident; $fn:ident; usize) => {
        impl $op<usize> for Dim {
            type Output = Self;
            fn $fn(self, rhs: usize) -> Self::Output {
                self.$fn(Self::Constant(rhs))
            }
        }
    };
}

impl_op!(Add; add; positive: Sum    );
impl_op!(Sub; sub; negative: Sum    );
impl_op!(Mul; mul; positive: Product);
impl_op!(Div; div; negative: Product);

impl_op!(Add; add; usize);
impl_op!(Sub; sub; usize);
impl_op!(Mul; mul; usize);
impl_op!(Div; div; usize);

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct Factor {
    base: String,
    exponent: isize,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct CanonicalTerm {
    coef: Ratio<isize>,
    factors: Vec<Factor>,  // sorted factors representing the term
}

impl CanonicalTerm {
    fn new(coef: isize) -> Self {
        Self {
            coef: Ratio::new(coef, 1),
            factors: Vec::new(),
        }
    }

    fn neg(&mut self) {
        self.coef = -self.coef;
    }

    fn with_var(coef: isize, var: String) -> Self {
        Self {
            coef: Ratio::new(coef, 1),
            factors: vec![Factor { base: var, exponent: 1 }],
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        let mut result = Self::new(1);
        result.coef = self.coef * other.coef;
        
        // Combine factors
        let mut factors = self.factors.clone();
        factors.extend(other.factors.iter().cloned());
        
        // Sort and combine like factors
        factors.sort_by(|a, b| a.base.cmp(&b.base));
        let mut combined = Vec::new();
        let mut current: Option<Factor> = None;
        
        for factor in factors {
            match &mut current {
                Some(prev) if prev.base == factor.base => {
                    prev.exponent += factor.exponent;
                }
                _ => {
                    if let Some(prev) = current.take() {
                        if prev.exponent != 0 {
                            combined.push(prev);
                        }
                    }
                    current = Some(factor);
                }
            }
        }
        
        if let Some(prev) = current {
            if prev.exponent != 0 {
                combined.push(prev);
            }
        }
        
        result.factors = combined;
        result
    }

    fn divide(&self, other: &Self) -> Self {
        let mut result = Self::new(1);
        result.coef = self.coef / other.coef;
        
        // Combine factors with negative exponents for division
        let mut factors = self.factors.clone();
        factors.extend(other.factors.iter().map(|f| Factor {
            base: f.base.clone(),
            exponent: -f.exponent,
        }));
        
        // Sort and combine like factors
        factors.sort_by(|a, b| a.base.cmp(&b.base));
        let mut combined = Vec::new();
        let mut current: Option<Factor> = None;
        
        for factor in factors {
            match &mut current {
                Some(prev) if prev.base == factor.base => {
                    prev.exponent += factor.exponent;
                }
                _ => {
                    if let Some(prev) = current.take() {
                        if prev.exponent != 0 {
                            combined.push(prev);
                        }
                    }
                    current = Some(factor);
                }
            }
        }
        
        if let Some(prev) = current {
            if prev.exponent != 0 {
                combined.push(prev);
            }
        }
        
        result.factors = combined;
        result
    }

    /// Combine like terms by adding coefficients of terms with the same variables.
    pub fn combine_like_terms(mut terms: Vec<CanonicalTerm>) -> Vec<CanonicalTerm> {
        terms.sort_by(|a, b| {
            let mut a_vars: Vec<_> = a.factors.iter().collect();
            let mut b_vars: Vec<_> = b.factors.iter().collect();
            a_vars.sort();
            b_vars.sort();
            a_vars.cmp(&b_vars)
        });
        
        let mut result = Vec::new();
        let mut current: Option<CanonicalTerm> = None;
        
        for term in terms {
            match &mut current {
                Some(prev) if prev.factors == term.factors => {
                    prev.coef += term.coef;
                }
                _ => {
                    if let Some(prev) = current.take() {
                        if prev.coef != Ratio::new(0, 1) {
                            result.push(prev);
                        }
                    }
                    current = Some(term);
                }
            }
        }
        
        if let Some(prev) = current {
            if prev.coef != Ratio::new(0, 1) {
                result.push(prev);
            }
        }
        
        result
    }

    fn sum_terms(terms: &[Self], other: &[Self]) -> Vec<Self> {
        let mut result = terms.to_owned();
        result.extend(other.to_owned());
        CanonicalTerm::combine_like_terms(result)
    }

    fn multiply_terms(terms: &Vec<Self>, other: &Vec<Self>) -> Vec<Self> {
        let mut result_terms = Vec::new();
        for term1 in terms {
            for term2 in other {
                result_terms.push(term1.multiply(term2));
            }
        }
        CanonicalTerm::combine_like_terms(result_terms)
    }

    fn terms_divide_by_term(terms: &Vec<Self>, dividend: &Self) -> Vec<Self> {
        let mut result_terms = Vec::new();
        for term in terms {
            result_terms.push(term.divide(dividend));
        }
        CanonicalTerm::combine_like_terms(result_terms)
    }

}

impl PartialOrd for CanonicalTerm {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CanonicalTerm {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.factors.cmp(&other.factors) {
            std::cmp::Ordering::Equal => self.coef.cmp(&other.coef),
            other => other,
        }
    }
}

impl PartialOrd for Factor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Factor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.base.cmp(&other.base) {
            std::cmp::Ordering::Equal => self.exponent.cmp(&other.exponent),
            other => other,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct RationalExpression {
    numer: Vec<CanonicalTerm>,
    denom: Vec<CanonicalTerm>,
}

impl RationalExpression {
    fn new_zero() -> Self {
        Self { numer: vec![CanonicalTerm::new(0)], denom: vec![CanonicalTerm::new(1)] }
    }

    fn new_one() -> Self {
        Self { numer: vec![CanonicalTerm::new(1)], denom: vec![CanonicalTerm::new(1)] }
    }

    fn new(numer: Vec<CanonicalTerm>, denom: Vec<CanonicalTerm>) -> Self {
        Self { numer, denom }
    }

    fn neg(&mut self) {
        self.numer.iter_mut().for_each(CanonicalTerm::neg);
    }

    fn invert(&mut self) {
        std::mem::swap(&mut self.numer, &mut self.denom);
    }

    fn from_dim(dim: &Dim) -> Option<Self> {
        match dim {
            Dim::Constant(value) => Some(Self::new(vec![CanonicalTerm::new(*value as isize)], vec![CanonicalTerm::new(1)])),
            Dim::Variable(name) => Some(Self::new(vec![CanonicalTerm::with_var(1, name.clone())], vec![CanonicalTerm::new(1)])),
            Dim::Sum(operands) => {
                let mut result = RationalExpression::new_zero();
                for operand in operands {
                    let sign = match operand.ty {
                        Type::Positive => 1,
                        Type::Negative => -1,
                    };
                    let mut rational = RationalExpression::from_dim(&operand.dim)?;
                    if sign == -1 {
                        rational.neg();
                    }
                    result = RationalExpression::new(
                        CanonicalTerm::sum_terms(&CanonicalTerm::multiply_terms(&result.numer, &rational.denom), &CanonicalTerm::multiply_terms(&rational.numer, &result.denom)),
                        CanonicalTerm::multiply_terms(&result.denom, &rational.denom),
                    );
                }
                Some(result)
            }
            Dim::Product(operands) => {
                let mut result = RationalExpression::new_one();

                for operand in operands {
                    let sign = match operand.ty {
                        Type::Positive => 1,
                        Type::Negative => -1,
                    };
                    let mut rational = RationalExpression::from_dim(&operand.dim)?;
                    if sign == -1 {
                        rational.invert();
                    }

                    if rational.denom.len() > 1 {
                    
                        result = RationalExpression::new(
                            CanonicalTerm::multiply_terms(&result.numer, &rational.numer),
                            CanonicalTerm::multiply_terms(&result.denom, &rational.denom),
                        );
                    } else {
                        result.numer = CanonicalTerm::multiply_terms(&result.numer, &rational.numer);
                        result.numer = CanonicalTerm::terms_divide_by_term(&result.numer, &rational.denom[0]);
                    }
                }
                Some(result)
            }
        }
    }

}

impl PartialOrd for RationalExpression {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RationalExpression {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.numer.cmp(&other.numer) {
            std::cmp::Ordering::Equal => self.denom.cmp(&other.denom),
            other => other,
        }
    }
}

impl From<RationalExpression> for Dim {
    fn from(rational: RationalExpression) -> Self {
        // Assert denominator is not empty
        assert!(!rational.denom.is_empty(), "Denominator cannot be empty in RationalExpression");

        // Convert numerator terms to Dim
        let numer_dim = if rational.numer.is_empty() {
            Dim::Constant(0)
        } else {
            let mut terms = VecDeque::new();
            for term in rational.numer {
                let coef = term.coef.reduced();
                let mut dim = 
                    Dim::Product(VecDeque::from([Operand { ty: Type::Positive, dim: Dim::Constant(coef.numer().unsigned_abs()) }, Operand { ty: Type::Negative, dim: Dim::Constant(coef.denom().unsigned_abs()) }]));
                for factor in term.factors {
                    let var = Dim::Variable(factor.base);
                    if factor.exponent > 0 {
                        for _ in 0..factor.exponent {
                            dim = dim * var.clone();
                        }
                    } else {
                        for _ in 0..-factor.exponent {
                            dim = dim / var.clone();
                        }
                    }
                }
                terms.push_back(Operand {
                    ty: if *term.coef.numer() < 0 { Type::Negative } else { Type::Positive },
                    dim,
                });
            }
            Dim::Sum(terms)
        };

        // Convert denominator terms to Dim
        let denom_dim = if rational.denom.is_empty() {
            Dim::Constant(1)
        } else {
            let mut terms = VecDeque::new();
            for term in rational.denom {
                let coef = term.coef.reduced();
                let mut dim = 
                    Dim::Product(VecDeque::from([Operand { ty: Type::Positive, dim: Dim::Constant(coef.numer().unsigned_abs()) }, Operand { ty: Type::Negative, dim: Dim::Constant(coef.denom().unsigned_abs()) }]));
                for factor in term.factors {
                    let var = Dim::Variable(factor.base);
                    if factor.exponent > 0 {
                        for _ in 0..factor.exponent {
                            dim = dim * var.clone();
                        }
                    } else {
                        for _ in 0..-factor.exponent {
                            dim = dim / var.clone();
                        }
                    }
                }
                terms.push_back(Operand {
                    ty: if *term.coef.numer() < 0 { Type::Negative } else { Type::Positive },
                    dim,
                });
            }
            Dim::Sum(terms)
        };

        // If denominator is 1, just return numerator
        if denom_dim == Dim::Constant(1) {
            numer_dim
        } else {
            // Otherwise, divide numerator by denominator
            numer_dim / denom_dim
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_example() {
        let a = Dim::var("a");
        let b = Dim::var("b");
        let _1 = Dim::from(1);
        let expr = (a + _1 - 2) * 3 / (b + 1);
        assert_eq!(expr.substitute(&HashMap::from([("a", 8), ("b", 6)])), 3);
    }

    #[test]
    fn test_dim_equivalence() {
        // Test constant equivalence
        assert_eq!(Dim::from(1).equivalent(&Dim::from(1)), Some(true));
        assert!(Dim::from(1) != Dim::from(2));
        assert_eq!(Dim::from(1).equivalent(&Dim::from(2)), Some(false));

        // Test variable equivalence
        let a = Dim::var("a");
        let b = Dim::var("b");
        println!("asserting a != b");
        assert!(a != b);
        assert_eq!(a.equivalent(&b), Some(false));

        // Test sum equivalence
        let expr1 = a.clone() + 1;
        let expr2 = a.clone() + 1;
        let expr3 = a.clone() + 2;
        println!("asserting a + 1 == a + 1");
        assert!(expr1 == expr2);
        assert_eq!(expr1.equivalent(&expr2), Some(true));
        println!("asserting a + 1 != a + 2");
        assert!(expr1 != expr3);
        assert_eq!(expr1.equivalent(&expr3), Some(false));

        // Test product equivalence
        let expr4 = a.clone() * 2;
        let expr5 = a.clone() * 2;
        let expr6 = a.clone() * 3;
        println!("asserting a * 2 == a * 2");
        assert!(expr4 == expr5);
        assert_eq!(expr4.equivalent(&expr5), Some(true));
        println!("asserting a * 2 != a * 3");
        assert!(expr4 != expr6);
        assert_eq!(expr4.equivalent(&expr6), Some(false));

        // Test complex expression equivalence
        let complex1 = (a.clone() + 1) * 2;
        let complex2 = a.clone() * 2 + 2;
        let complex3 = a.clone() * 2 + 3;
        println!("asserting (a + 1) * 2 == a * 2 + 2");
        assert!(complex1 == complex2);
        assert_eq!(complex1.equivalent(&complex2), Some(true)); // (a + 1) * 2 = a * 2 + 2
        println!("asserting (a + 1) * 2 != a * 2 + 3");
        assert!(complex1 != complex3);
        assert_eq!(complex1.equivalent(&complex3), Some(false));

        // Test commutative operations
        let expr7 = a.clone() + b.clone();
        let expr8 = b.clone() + a.clone();
        println!("asserting a + b == b + a");
        assert!(expr7 == expr8);
        assert_eq!(expr7.equivalent(&expr8), Some(true)); // a + b = b + a

        let expr9 = a.clone() * b.clone();
        let expr10 = b.clone() * a.clone();
        println!("asserting a * b == b * a");
        assert!(expr9 == expr10);
        assert_eq!(expr9.equivalent(&expr10), Some(true)); // a * b = b * a

        // Test distributive property
        let expr11 = a.clone() * (b.clone() + 1);
        let expr12 = a.clone() * b.clone() + a.clone();
        println!("asserting a * (b + 1) == a * b + a");
        assert!(expr11 == expr12);
        assert_eq!(expr11.equivalent(&expr12), Some(true)); // a * (b + 1) = a * b + a

        // Test division and complex expressions
        let c = Dim::var("c");

        // Test division equivalence
        let expr13 = (a.clone() * b.clone()) / c.clone();
        let expr14 = a.clone() * (b.clone() / c.clone());
        println!("asserting (a * b) / c == a * (b / c)");
        assert!(expr13 == expr14);
        assert_eq!(expr13.equivalent(&expr14), Some(true)); // (a * b) / c = a * (b / c)

        // Test mixed operations with division
        let expr15 = (a.clone() + b.clone()) / c.clone();
        let expr16 = a.clone() / c.clone() + b.clone() / c.clone();
        println!("asserting (a + b) / c == a/c + b/c");
        assert!(expr15 == expr16);
        assert_eq!(expr15.equivalent(&expr16), Some(true)); // (a + b) / c = a/c + b/c

        // Test complex nested expressions
        let expr17 = (a.clone() * b.clone() + c.clone()) / (a.clone() + Dim::from(1));
        let expr18 = (b.clone() * a.clone() + c.clone()) / (Dim::from(1) + a.clone());
        println!("asserting (a*b + c)/(a + 1) == (b*a + c)/(1 + a)");
        assert!(expr17 == expr18);
        assert_eq!(expr17.equivalent(&expr18), Some(true)); // (a*b + c)/(a + 1) = (b*a + c)/(1 + a)

        // Test expressions with multiple divisions
        let expr19 = (a.clone() / b.clone()) / c.clone();
        let expr20 = a.clone() / (b.clone() * c.clone());
        println!("asserting (a/b)/c == a/(b*c)");
        assert!(expr19 == expr20);
        assert_eq!(expr19.equivalent(&expr20), Some(true)); // (a/b)/c = a/(b*c)

        // Test expressions with constants and variables
        let expr21 = (a.clone() * Dim::from(2) + b.clone() * Dim::from(3)) / Dim::from(6);
        let expr22 = a.clone() / Dim::from(3) + b.clone() / Dim::from(2);
        println!("asserting (2a + 3b)/6 == a/3 + b/2");
        assert!(expr21 == expr22);
        assert_eq!(expr21.equivalent(&expr22), Some(true)); // (2a + 3b)/6 = a/3 + b/2

        // Test expressions with nested divisions and multiplications
        let expr23 = a.clone() * (b.clone() / (c.clone() * Dim::from(2)));
        let expr24 = (a.clone() * b.clone()) / (Dim::from(2) * c.clone());
        println!("asserting a * (b/(c*2)) == (a*b)/(2*c)");
        assert!(expr23 == expr24);
        assert_eq!(expr23.equivalent(&expr24), Some(true)); // a * (b/(c*2)) = (a*b)/(2*c)

        // Test expressions with subtraction and division
        let expr25 = (a.clone() - b.clone()) / c.clone();
        let expr26 = a.clone() / c.clone() - b.clone() / c.clone();
        println!("asserting (a - b)/c == a/c - b/c");
        assert!(expr25 == expr26);
        assert_eq!(expr25.equivalent(&expr26), Some(true)); // (a - b)/c = a/c - b/c

        // Test expressions with multiple operations
        let expr27 = (a.clone() * b.clone() + c.clone() * Dim::from(2)) / (b.clone() + Dim::from(2));
        let expr28 = (b.clone() * a.clone() + Dim::from(2) * c.clone()) / (Dim::from(2) + b.clone());
        println!("asserting (a*b + 2c)/(b + 2) == (b*a + 2c)/(2 + b)");
        assert!(expr27 == expr28);
        assert_eq!(expr27.equivalent(&expr28), Some(true)); // (a*b + 2c)/(b + 2) = (b*a + 2c)/(2 + b)
    }

    #[test]
    fn test_power_expressions() {
        let a = Dim::var("a");
        let b = Dim::var("b");
        let c = Dim::var("c");

        // Test simple power expressions
        let pow1 = a.clone() * a.clone();
        let pow2 = a.clone() * a.clone();
        println!("asserting a * a == a * a");
        assert!(pow1 == pow2);
        assert_eq!(pow1.equivalent(&pow2), Some(true));

        // Test power with division
        let pow3 = (a.clone() * a.clone()) / b.clone();
        let pow4 = a.clone() * (a.clone() / b.clone());
        println!("asserting (a * a) / b == a * (a / b)");
        assert!(pow3 == pow4);
        assert_eq!(pow3.equivalent(&pow4), Some(true));

        // Test multiple variables with powers
        let pow5 = (a.clone() * a.clone() * b.clone()) / (c.clone() * c.clone());
        let pow6 = (a.clone() * b.clone()) * (a.clone() / (c.clone() * c.clone()));
        println!("asserting (a² * b) / c² == (a * b) * (a / c²)");
        assert!(pow5 == pow6);
        assert_eq!(pow5.equivalent(&pow6), Some(true));

        // Test complex power expressions with constants
        let pow7 = (a.clone() * a.clone() * Dim::from(2) + b.clone() * b.clone() * Dim::from(3)) / Dim::from(6);
        let pow8 = (a.clone() * a.clone()) / Dim::from(3) + (b.clone() * b.clone()) / Dim::from(2);
        println!("asserting (2a² + 3b²)/6 == a²/3 + b²/2");
        assert!(pow7 == pow8);
        assert_eq!(pow7.equivalent(&pow8), Some(true));

        // Test nested power expressions
        let pow9 = (a.clone() * a.clone() + b.clone()) / (a.clone() * c.clone());
        let pow10 = a.clone() / c.clone() + b.clone() / (a.clone() * c.clone());
        println!("asserting (a² + b)/(a * c) == a/c + b/(a * c)");
        assert!(pow9 == pow10);
        assert_eq!(pow9.equivalent(&pow10), Some(true));

        // Test power expressions with multiple operations
        let pow11 = (a.clone() * a.clone() * b.clone() + c.clone() * c.clone()) / (b.clone() + Dim::from(2));
        let pow12 = (b.clone() * a.clone() * a.clone() + c.clone() * c.clone()) / (Dim::from(2) + b.clone());
        println!("asserting (a² * b + c²)/(b + 2) == (b * a² + c²)/(2 + b)");
        assert!(pow11 == pow12);
        assert_eq!(pow11.equivalent(&pow12), Some(true));

        // Test power expressions with division and multiplication
        let pow13 = (a.clone() * a.clone()) / (b.clone() * b.clone()) * c.clone();
        let pow14 = (a.clone() * a.clone() * c.clone()) / (b.clone() * b.clone());
        println!("asserting (a²/b²) * c == (a² * c)/b²");
        assert!(pow13 == pow14);
        assert_eq!(pow13.equivalent(&pow14), Some(true));

        // Test sum of terms with same variable but different exponents
        let sum1 = a.clone() + a.clone() * a.clone() + a.clone() * a.clone() * a.clone();
        let sum2 = a.clone() * a.clone() * a.clone() + a.clone() * a.clone() + a.clone();
        println!("asserting a + a² + a³ == a³ + a² + a");
        assert!(sum1 == sum2);
        assert_eq!(sum1.equivalent(&sum2), Some(true));

        // Test sum of terms with same variable and coefficients
        let sum3 = a.clone() * Dim::from(2) + a.clone() * a.clone() * Dim::from(3) + a.clone() * a.clone() * a.clone() * Dim::from(4);
        let sum4 = a.clone() * a.clone() * a.clone() * Dim::from(4) + a.clone() * Dim::from(2) + a.clone() * a.clone() * Dim::from(3);
        println!("asserting 2a + 3a² + 4a³ == 4a³ + 2a + 3a²");
        assert!(sum3 == sum4);
        assert_eq!(sum3.equivalent(&sum4), Some(true));

        // Test sum of terms with same variable and negative coefficients
        let sum5 = a.clone() * Dim::from(2) - a.clone() * a.clone() * Dim::from(3) + a.clone() * a.clone() * a.clone() * Dim::from(4);
        let sum6 = a.clone() * a.clone() * a.clone() * Dim::from(4) + a.clone() * Dim::from(2) - a.clone() * a.clone() * Dim::from(3);
        println!("asserting 2a - 3a² + 4a³ == 4a³ + 2a - 3a²");
        assert!(sum5 == sum6);
        assert_eq!(sum5.equivalent(&sum6), Some(true));

        // Test sum of terms with same variable and mixed operations
        let sum7 = (a.clone() * a.clone() + a.clone()) / Dim::from(2) + a.clone() * a.clone() * a.clone();
        let sum8 = a.clone() * a.clone() * a.clone() + (a.clone() + a.clone() * a.clone()) / Dim::from(2);
        println!("asserting (a² + a)/2 + a³ == a³ + (a + a²)/2");
        assert!(sum7 == sum8);
        assert_eq!(sum7.equivalent(&sum8), Some(true));
    }

    #[test]
    fn test_complex_expressions() {
        let a = Dim::var("a");
        let b = Dim::var("b");
        let c = Dim::var("c");
        
        // Test complex denominator expressions
        let complex1 = (a.clone() * a.clone() * b.clone()) / ((c.clone() + Dim::from(1)) * (c.clone() + Dim::from(2)));
        let complex2 = (a.clone() * a.clone() * b.clone()) / (c.clone() * c.clone() + c.clone() * 3 + Dim::from(2));
        println!("asserting (a² * b)/((c+1)(c+2)) == (a² * b)/(c² + 3c + 2)");
        assert!(complex1 == complex2);
        assert_eq!(complex1.equivalent(&complex2), Some(true));
    }

    #[test]
    fn test_factor_ordering() {
        let a = Dim::var("a");
        let b = Dim::var("b");
        
        // Test ordering of terms with same base but different exponents
        let expr1 = a.clone() + a.clone() * a.clone() + a.clone() * a.clone() * a.clone();
        let expr2 = a.clone() * a.clone() * a.clone() + a.clone() * a.clone() + a.clone();
        println!("asserting a + a² + a³ == a³ + a² + a");
        assert!(expr1 == expr2);
        assert_eq!(expr1.equivalent(&expr2), Some(true));

        // Test ordering of terms with different bases and exponents
        let expr3 = a.clone() + b.clone() + a.clone() * a.clone() + b.clone() * b.clone();
        let expr4 = b.clone() * b.clone() + a.clone() * a.clone() + b.clone() + a.clone();
        println!("asserting a + b + a² + b² == b² + a² + b + a");
        assert!(expr3 == expr4);
        assert_eq!(expr3.equivalent(&expr4), Some(true));

        // Test ordering with negative exponents
        let expr5 = a.clone() / b.clone() + a.clone() * a.clone() / b.clone();
        let expr6 = a.clone() * a.clone() / b.clone() + a.clone() / b.clone();
        println!("asserting a/b + a²/b == a²/b + a/b");
        assert!(expr5 == expr6);
        assert_eq!(expr5.equivalent(&expr6), Some(true));
    }

    #[test]
    fn test_rational_expression_into_dim() {
        let a = Dim::var("a");
        let b = Dim::var("b");
        let c = Dim::var("c");

        // Test simple constant expressions
        let rational1 = RationalExpression::new(
            vec![CanonicalTerm::new(5)],
            vec![CanonicalTerm::new(1)]
        );
        let dim1: Dim = rational1.into();
        assert_eq!(dim1, Dim::Constant(5));

        // Test simple variable expressions
        let rational2 = RationalExpression::new(
            vec![CanonicalTerm::with_var(1, "a".to_string())],
            vec![CanonicalTerm::new(1)]
        );
        let dim2: Dim = rational2.into();
        assert_eq!(dim2, a.clone());

        // Test expressions with negative coefficients
        let rational3 = RationalExpression::new(
            vec![CanonicalTerm::new(-3)],
            vec![CanonicalTerm::new(1)]
        );
        let dim3: Dim = rational3.into();
        let expected3 = Dim::Sum(VecDeque::from([Operand { ty: Type::Negative, dim: Dim::Constant(3) }]));
        assert_eq!(dim3, expected3);

        // Test simple division
        let rational4 = RationalExpression::new(
            vec![CanonicalTerm::with_var(1, "a".to_string())],
            vec![CanonicalTerm::with_var(1, "b".to_string())]
        );
        let dim4: Dim = rational4.into();
        assert_eq!(dim4, a.clone() / b.clone());

        // Test complex rational expressions
        let rational5 = RationalExpression::new(
            vec![
                CanonicalTerm::with_var(2, "a".to_string()),
                CanonicalTerm::with_var(3, "b".to_string())
            ],
            vec![CanonicalTerm::new(6)]
        );
        let dim5: Dim = rational5.into();
        assert_eq!(dim5, (a.clone() * 2 + b.clone() * 3) / 6);

        // Test expressions with multiple variables and exponents
        let rational6 = RationalExpression::new(
            vec![
                CanonicalTerm {
                    coef: Ratio::new(1, 1),
                    factors: vec![
                        Factor { base: "a".to_string(), exponent: 2 },
                        Factor { base: "b".to_string(), exponent: 1 }
                    ]
                }
            ],
            vec![
                CanonicalTerm {
                    coef: Ratio::new(1, 1),
                    factors: vec![
                        Factor { base: "c".to_string(), exponent: 2 }
                    ]
                }
            ]
        );
        let dim6: Dim = rational6.into();
        assert_eq!(dim6, (a.clone() * a.clone() * b.clone()) / (c.clone() * c.clone()));

        // Test expressions with negative exponents
        let rational7 = RationalExpression::new(
            vec![
                CanonicalTerm {
                    coef: Ratio::new(1, 1),
                    factors: vec![
                        Factor { base: "a".to_string(), exponent: 1 },
                        Factor { base: "b".to_string(), exponent: -1 }
                    ]
                }
            ],
            vec![CanonicalTerm::new(1)]
        );
        let dim7: Dim = rational7.into();
        assert_eq!(dim7, a.clone() / b.clone());

        // Test expressions with multiple terms in numerator and denominator
        let rational8 = RationalExpression::new(
            vec![
                CanonicalTerm::with_var(2, "a".to_string()),
                CanonicalTerm::with_var(3, "b".to_string())
            ],
            vec![
                CanonicalTerm::with_var(1, "c".to_string()),
                CanonicalTerm::new(2)
            ]
        );
        let dim8: Dim = rational8.into();
        assert_eq!(dim8, (a.clone() * 2 + b.clone() * 3) / (c.clone() + 2));

        // Test empty numerator (should become 0)
        let rational9 = RationalExpression::new(
            vec![],
            vec![CanonicalTerm::new(1)]
        );
        let dim9: Dim = rational9.into();
        assert_eq!(dim9, Dim::Constant(0));
    }

    #[test]
    #[should_panic(expected = "Denominator cannot be empty in RationalExpression")]
    fn test_rational_expression_empty_denominator() {
        let rational = RationalExpression::new(
            vec![CanonicalTerm::new(5)],
            vec![]
        );
        let _: Dim = rational.into();
    }

    #[test]
    fn test_rational_function_equivalence() {
        let a = Dim::var("a");
        let b = Dim::var("b");
        let c = Dim::var("c");
        let d = Dim::var("d");

        // Create a rational function: (a + b) / (c + d)
        let rational1 = (a.clone() + b.clone()) / (c.clone() + d.clone());

        // Create an equivalent rational function by multiplying both numerator and denominator
        // by the same expression (a + b + c)
        let common_factor = a.clone() + b.clone() + c.clone();
        let rational2 = ((a.clone() + b.clone()) * common_factor.clone()) / ((c.clone() + d.clone()) * common_factor);

        // The two expressions should be equivalent
        println!("asserting (a + b)/(c + d) == ((a + b)(a + b + c))/((c + d)(a + b + c))");
        assert!(!(rational1 == rational2));
        assert!(!(rational1 != rational2));
        assert_eq!(rational1.equivalent(&rational2), None);

        // Test with more complex expressions
        let complex1 = (a.clone() * b.clone() + c.clone()) / (a.clone() + d.clone());
        let common_factor2 = a.clone() * b.clone() + c.clone() + d.clone();
        let complex2 = ((a.clone() * b.clone() + c.clone()) * common_factor2.clone()) / ((a.clone() + d.clone()) * common_factor2);

        println!("asserting (ab + c)/(a + d) == ((ab + c)(ab + c + d))/((a + d)(ab + c + d))");
        assert!(!(complex1 == complex2));
        assert!(!(complex1 != complex2));
        assert_eq!(complex1.equivalent(&complex2), None);

        // Test with expressions containing constants
        let const1 = (a.clone() * 2 + b.clone() * 3) / (c.clone() + 4);
        let common_factor3 = a.clone() + b.clone() + c.clone();
        let const2 = ((a.clone() * 2 + b.clone() * 3) * common_factor3.clone()) / ((c.clone() + 4) * common_factor3);

        println!("asserting (2a + 3b)/(c + 4) == ((2a + 3b)(a + b + c))/((c + 4)(a + b + c))");
        assert!(!(const1 == const2));
        assert!(!(const1 != const2));
        assert_eq!(const1.equivalent(&const2), None);

        // Test with nested expressions
        let nested1 = ((a.clone() + b.clone()) * c.clone()) / ((a.clone() - b.clone()) * d.clone());
        let common_factor4 = a.clone() * b.clone() + c.clone() * d.clone();
        let nested2 = (((a.clone() + b.clone()) * c.clone()) * common_factor4.clone()) / (((a.clone() - b.clone()) * d.clone()) * common_factor4);

        println!("asserting ((a + b)c)/((a - b)d) == ((a + b)c(ab + cd))/((a - b)d(ab + cd))");
        assert!(!(nested1 == nested2));
        assert!(!(nested1 != nested2));
        assert_eq!(nested1.equivalent(&nested2), None);
    }
}