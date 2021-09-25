#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};


#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct TextExtractor {}

impl TextExtractor {
    pub const fn new() -> Self {
        TextExtractor {}
    }
}

impl Default for TextExtractor {
    fn default() -> Self{
        TextExtractor::new()
    }
}



impl Transform<(Tensor<i32>, Tensor<u32>, Tensor<u32>)> for TextExtractor {
    type Output = Tensor<i32>;

    fn transform(&mut self, inputs : (Tensor<i32>, Tensor<u32>, Tensor<u32>)) -> Tensor<i32> {
        
        let (token_ids, start_logits, end_logits) = inputs;
        let ids: Vec<i32> = token_ids.elements().iter().map(|&x| x).collect::<Vec<i32>>();
        let start_index: u32 = start_logits.elements().iter().map(|&x| x).collect::<Vec<u32>>()[0];
        let end_index: u32 = end_logits.elements().iter().map(|&x| x).collect::<Vec<u32>>()[0];
        if end_index <= start_index {
            panic!("Start index: {} is greater than end index: {}", start_index, end_index);
        }

        let mut v: Vec<i32> = ids[start_index as usize..end_index as usize].to_vec();
        v.resize(384, 0);

        Tensor::new_vector(v)

    }
}

impl HasOutputs for TextExtractor {}