#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};

#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
pub struct ZeroIndexing {}

impl Transform<Tensor<f32>> for ZeroIndexing {
    type Output = Tensor<u32>;

    fn transform(&mut self, input: Tensor<f32>) -> Self::Output {
        let indices = input
            .elements()
            .iter()
            .map(|x| (x - 1.0) as u32)
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
        let input = vec![2.0, 5.0, 7.0, 9.0, 11.0];
        let input = Tensor::new_vector(input);
        let mut zero_indexing = ZeroIndexing::default();
        let output = zero_indexing.transform(input);

        assert_eq!(output, Tensor::new_vector(vec![1, 4, 6, 8, 10]));
    }
}
