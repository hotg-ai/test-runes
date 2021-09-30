#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::vec;
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};
use crate::alloc::borrow::ToOwned;

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

impl Transform<(Tensor<u8>, Tensor<u32>, Tensor<u32>)> for TextExtractor {
    type Output = Tensor<&'static str>;

    fn transform(&mut self, inputs : (Tensor<u8>, Tensor<u32>, Tensor<u32>)) -> Tensor<&'static str> {
        
        let (text, start_logits, end_logits) = inputs;

        let underlying_bytes: &[u8] = text.elements();
        let input_text = core::str::from_utf8(underlying_bytes).expect("Input tensor should be valid UTF8");

        let input_text: Vec<&'static str> = input_text.lines().collect::<Vec<&'static str>>();

        let start_index: u32 = start_logits.elements().iter().map(|&x| x).collect::<Vec<u32>>()[0];
        let end_index: u32 = end_logits.elements().iter().map(|&x| x).collect::<Vec<u32>>()[0];
        if end_index <= start_index {
            panic!("Start index: {} is greater than or equal to end index: {}", start_index, end_index);
        }
        
        let v: Vec<&'static str> = input_text[start_index as usize..end_index as usize+1].to_vec();
        
        let output_text: &'static str = v.join(" ").replace("##", "").trim().to_owned().as_str();
        let output_text: Vec<&'static str> = vec![output_text];

        Tensor::new_vector(output_text)

    }
}

impl HasOutputs for TextExtractor {}