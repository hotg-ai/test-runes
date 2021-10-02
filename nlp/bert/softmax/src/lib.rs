#![no_std]

extern crate alloc;
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};
use libm::expf;

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Softmax {}

impl Softmax {
    pub const fn new() -> Self {
        Softmax {}
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Softmax::new()
    }
}

impl Transform<Tensor<f32>> for Softmax {
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<f32> {
        let b = input.map(|_, &x| expf(x as f32));
        let sum: f32 = b.elements().iter().sum();

        b.map(|_, &x| x / sum)
    }
}

impl HasOutputs for Softmax {}
