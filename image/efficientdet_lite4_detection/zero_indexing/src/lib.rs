#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};
use num::One;

#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
pub struct ZeroIndexing {}

impl<T> Transform<Tensor<T>> for ZeroIndexing
where
    T: One + core::ops::Sub<Output = T> + Copy,
{
    type Output = Tensor<T>;

    fn transform(&mut self, input: Tensor<T>) -> Self::Output {
        let indices = input
            .elements()
            .iter()
            .map(|&x| (x - T::one()) as u32)
            .collect::<Vec<u32>>();

        Tensor::new_vector(indices)
    }
}

impl HasOutputs for ZeroIndexing {}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_zero_indexing() {
        let input = vec![2, 5, 7, 9, 11];
        let input = Tensor::new_vector(input);
        let mut zero_indexing = ZeroIndexing::default();
        let output = zero_indexing.transform(input);

        assert_eq!(output, Tensor::new_vector(vec![1, 4, 6, 8, 10]));
    }
}
