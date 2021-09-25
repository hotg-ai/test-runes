#![no_std]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Argmax {}

impl Argmax {
    pub const fn new() -> Self {
        Argmax {}
    }
}

impl Default for Argmax{
    fn default() -> Self{
        Argmax::new()
    }
}

impl Transform<Tensor<f32>> for Argmax {
    type Output = Tensor<u32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<u32> {
        
        let (index, _) = input.elements().iter().enumerate().fold((0, 0.0), |max, (ind, &val)| if val > max.1 {(ind, val)} else {max}); 

        let v: Vec<u32> = vec![index as u32];

        Tensor::new_vector(v)

    }
}

impl HasOutputs for Argmax {}


