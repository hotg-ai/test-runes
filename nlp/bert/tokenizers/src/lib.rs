#![no_std]
pub mod tokenizer;
pub mod vocab;

pub use crate::tokenizer::base_tokenizer::{
    ConsolidatableTokens, ConsolidatedTokenIterator, Mask, Offset, OffsetSize, Token,
    TokenIdsWithOffsets, TokenIdsWithSpecialTokens, TokenRef, TokenTrait, TokenizedInput,
    TokensWithOffsets,
};
use alloc::vec::Vec;
use alloc::vec;
use core::default::Default;
use core::str::FromStr;

#[macro_use]
extern crate alloc;

pub use crate::{
    tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy},
    vocab::{BertVocab, Vocab},
};
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};

#[derive(ProcBlock)]
pub struct Tokenizers {
    bert_tokenizer: BertTokenizer,
}

impl Default for Tokenizers {
    fn default() -> Tokenizers {
        let vocabulary_text = include_str!("bert-base-uncased-vocab.txt");

        let vocab = BertVocab::from_str(vocabulary_text).unwrap();

        let bert_tokenizer = BertTokenizer::from_existing_vocab(vocab, true, true);

        Tokenizers { bert_tokenizer }
    }
}

impl Transform<(Tensor<u8>, Tensor<u8>)> for Tokenizers {
    type Output = (Tensor<i32>, Tensor<i32>, Tensor<i32>);

    fn transform(&mut self, s: (Tensor<u8>, Tensor<u8>)) -> (Tensor<i32>, Tensor<i32>, Tensor<i32>) {
        let (s1, s2) = s;
        let underlying_bytes_1: &[u8] = s1.elements();
        let input_text_1: &str =
            core::str::from_utf8(underlying_bytes_1).expect("Input tensor should be valid UTF8");

        let underlying_bytes_2: &[u8] = s2.elements();
        let input_text_2: &str =
            core::str::from_utf8(underlying_bytes_2).expect("Input tensor should be valid UTF8");

        let token: Tokenizers = Default::default();



        let TokenizedInput {
            mut token_ids,
            special_tokens_mask: _,
            mut segment_ids,
            ..
        } = token.bert_tokenizer.encode(input_text_1,
            Some(input_text_2),
            384,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let mut mask_ids: Vec<i32> = vec![1; token_ids.len()];
        token_ids.resize(384, 0);
        mask_ids.resize(384, 0);
        segment_ids.resize(384, 0);

        let input_ids: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect::<Vec<i32>>();
        // let mask_ids: Vec<i32> = token_mask
        //     .iter()
        //     .map(|&x| x as i32)
        //     .collect::<Vec<i32>>();
        let seg_ids: Vec<i32> = segment_ids.iter().map(|&x| x as i32).collect::<Vec<i32>>();

        (
            Tensor::new_row_major(input_ids.into(), vec![1, 384]),
            Tensor::new_row_major(mask_ids.into(), vec![1, 384]),
            Tensor::new_row_major(seg_ids.into(), vec![1, 384]),
        )
    }
}

impl HasOutputs for Tokenizers {}
