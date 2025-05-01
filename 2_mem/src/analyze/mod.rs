mod action;
mod key_weak;
mod life_time;
mod mem_range;

pub use action::Action;
pub use life_time::BlobLifeTime;
pub use mem_range::MemRangeMap;

pub fn print_lifetime<T>(lt: &[BlobLifeTime<T>]) {
    for (i, BlobLifeTime { blob, life_time }) in lt.iter().enumerate() {
        let Some(&crate::Info::Internal(size)) = blob.upgrade().as_deref() else {
            panic!()
        };
        print!("{i:>3} {size:6} ");
        for _ in 0..life_time.start {
            print!(" ")
        }
        for _ in life_time.start..=life_time.end {
            print!("#")
        }
        println!()
    }
}

#[allow(clippy::reversed_empty_ranges)]
const EMPTY_RANGE: std::ops::Range<usize> = usize::MAX..0;
