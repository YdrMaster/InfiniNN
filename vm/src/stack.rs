use crate::{Id, device_id, pid};
use std::{fmt, sync::Arc};

#[derive(Clone)]
pub struct StackTracer {
    domain: Domain,
    seq: Vec<u8>,
}

impl StackTracer {
    pub fn new(pid: pid, dev: device_id) -> Self {
        Self {
            domain: Domain { pid, dev },
            seq: Vec::new(),
        }
    }

    pub fn push(&mut self, id: impl Id) {
        let name = id.name();

        assert!(name.len() < 128);
        self.seq.extend_from_slice(name.as_bytes());

        let meta = (name.len() as u8) << 1;
        match id.idx() {
            Some(idx) => {
                assert!(idx <= u32::MAX as usize);
                self.seq.extend_from_slice(&(idx as u32).to_ne_bytes());
                self.seq.push(meta | 1);
            }
            None => {
                self.seq.push(meta);
            }
        }
    }

    pub fn pop(&mut self) {
        let meta = self.seq.pop().unwrap();
        let len = (meta >> 1) as usize + if meta & 1 == 1 { 4 } else { 0 };
        self.seq.truncate(self.seq.len() - len);
    }

    pub fn obj(&self, which: impl Id) -> ObjId {
        let mut temp = self.clone();
        temp.push(which);
        temp.build_obj_id(true)
    }

    pub fn path(&self) -> ObjId {
        self.clone().build_obj_id(false)
    }

    fn build_obj_id(self, is_obj: bool) -> ObjId {
        let Self { domain, mut seq } = self;
        seq.extend_from_slice(&domain.pid.to_ne_bytes());
        seq.extend_from_slice(&domain.dev.to_ne_bytes());
        seq.push(if is_obj { 1 } else { 0 });
        ObjId(seq.into())
    }
}

#[derive(Clone)]
pub struct ObjId(Arc<[u8]>);

impl AsRef<[u8]> for ObjId {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl ObjId {
    pub fn global() -> Self {
        StackTracer::new(pid::MAX, device_id::MAX).path()
    }

    pub fn is_obj(&self) -> bool {
        matches!(self.0.last(), Some(1))
    }

    pub fn domain(&self) -> String {
        let head = self.0.len() - size_of::<pid>() - size_of::<device_id>() - 1;
        let tail = &self.0[head..];
        let pid = unsafe { tail.as_ptr().cast::<pid>().read_unaligned() };
        let dev = unsafe {
            tail[size_of::<pid>()..]
                .as_ptr()
                .cast::<device_id>()
                .read_unaligned()
        };
        Domain { pid, dev }.to_string()
    }

    pub fn body(&self) -> String {
        let body = self.0.len() - size_of::<pid>() - size_of::<device_id>() - 1;
        let mut body = &self.0[..body];

        let mut elements = Vec::new();
        while let &[ref body_ @ .., meta] = body {
            body = body_;
            let len = (meta >> 1) as usize;
            let idx = (meta & 1) == 1;
            if idx {
                let (head, tail) = body.split_at(body.len() - 4 - len);
                body = head;

                let idx = unsafe { tail[len..].as_ptr().cast::<u32>().read_unaligned() };
                elements.push((&tail[..len], Some(idx)))
            } else {
                let (head, tail) = body.split_at(body.len() - len);
                body = head;

                elements.push((tail, None))
            }
        }

        let mut ans = "Ω".to_string();
        for (name, idx) in elements.into_iter().rev() {
            ans.push('.');
            ans.push_str(unsafe { std::str::from_utf8_unchecked(name) });
            if let Some(idx) = idx {
                ans.push_str(&format!("#{idx}"))
            }
        }
        ans
    }
}

#[derive(Clone, Copy)]
struct Domain {
    pid: pid,
    dev: device_id,
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self { pid, dev } = self;
        write!(f, "[")?;
        match pid {
            pid::MAX => write!(f, "vm")?,
            n => write!(f, "#{n:x}")?,
        }
        write!(f, ":")?;
        match dev {
            device_id::MAX => write!(f, "H")?,
            n => write!(f, "{n}")?,
        }
        write!(f, "]")
    }
}
