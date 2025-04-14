macro_rules! destruct {
    ([$( $name:ident ),+] = $iter:expr) => {
        let mut iter = $iter.into_iter();
        $( let $name = iter.next().unwrap(); )+
        assert!(iter.next().is_none());
    };
}

macro_rules! dims {
    ($pat:pat = $tensor:expr) => {
        let &$pat = $tensor.shape() else {
            panic!("Ndim mismatch ( = {})", $tensor.shape().len())
        };
    };
}

pub(crate) use {destruct, dims};
