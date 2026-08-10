//! Per-token byte offsets. Monotonic, non-overlapping, exactly covering the
//! input (modulo documented normalization drops — none for byte-level BPE).

pub type OffsetList = smallvec::SmallVec<[Offset; 64]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Offset {
    pub start: usize,
    pub end: usize,
}

impl Offset {
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}
