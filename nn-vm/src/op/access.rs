use std::ops::BitOr;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    R,
    W,
    RW,
}

impl Access {
    pub fn may_read(self) -> bool {
        matches!(self, Self::R | Self::RW)
    }

    pub fn may_write(self) -> bool {
        matches!(self, Self::W | Self::RW)
    }
}

impl BitOr for Access {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let r = self.may_read() || rhs.may_read();
        let w = self.may_write() || rhs.may_write();
        match (r, w) {
            (true, false) => Access::R,
            (false, true) => Access::W,
            (true, true) => Access::RW,
            (false, false) => unreachable!(),
        }
    }
}
