// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use anyhow::Result;
use crate::vocab::base_vocab::{swap_key_values,Vocab};
use alloc::{collections::BTreeMap, string::String};
use core::str::FromStr;
use crate::alloc::string::ToString;


#[derive(Debug, Clone)]
pub enum TokenError {
    TokenNotFound{
        word: String,
    },
}
/// # BERT Vocab
/// Vocabulary for BERT tokenizer. Contains the following special values:
/// - CLS token
/// - SEP token
/// - PAD token
/// - MASK token
///
/// Expects a flat text vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct BertVocab {
    pub values: BTreeMap<String, i64>,
    pub indices: BTreeMap<i64, String>,
    pub special_value_indices: BTreeMap<String, i64>,
    pub special_indices: BTreeMap<i64, String>,
}

impl BertVocab {
    /// Returns the PAD token for BERT (`[PAD]`)
    pub const PAD: &'static str = "[PAD]";

    /// Returns the SEP token for BERT (`[SEP]`)
    pub const SEPARATOR: &'static str = "[SEP]";

    /// Returns the CLS token for BERT (`[CLS]`)
    pub const CLS: &'static str = "[CLS]";

    /// Returns the MASK token for BERT (`[MASK]`)
    pub const MASK: &'static str = "[MASK]";

    /// Returns the MASK token for BERT (`[UNK]`)
    pub const UNKNOWN: &'static str = "[UNK]";

    pub const SPECIAL_VALUES: &'static [&'static str] = &[
        BertVocab::UNKNOWN,
        BertVocab::MASK,
        BertVocab::SEPARATOR,
        BertVocab::CLS,
        BertVocab::PAD,
    ];
}

#[derive(Debug, Clone)]
pub enum ParseError {
    DuplicateWord {
        word: String,
        original_index: i64,
        duplicate_index: i64,
    },
}

impl FromStr for BertVocab{
    type Err = ParseError;

    fn from_str(dictionary: &str) -> Result<Self, ParseError> {
        
        let mut values = BTreeMap::new();
        let mut next_index = 0;

        for line in dictionary.lines() {
            let word = line.trim();


            if let Some(original_index) = values.insert(word.to_string(), next_index) {
                return Err(ParseError::DuplicateWord {
                    word: word.to_string(),
                    original_index,
                    duplicate_index: next_index,
                });
            }

            next_index += 1;
        }

        let mut special_value_indices = BTreeMap::new();

        let unknown_value = BertVocab::UNKNOWN;
        BertVocab::_register_as_special_value(unknown_value, &values, &mut special_value_indices).expect("Token index not found in vocabulary");

        let pad_value = BertVocab::PAD;
        BertVocab::_register_as_special_value(pad_value, &values, &mut special_value_indices).expect("Token index not found in vocabulary");

        let sep_value = BertVocab::SEPARATOR;
        BertVocab::_register_as_special_value(sep_value, &values, &mut special_value_indices).expect("Token index not found in vocabulary");

        let cls_value = BertVocab::CLS;
        BertVocab::_register_as_special_value(cls_value, &values, &mut special_value_indices).expect("Token index not found in vocabulary");

        let mask_value = BertVocab::MASK;
        BertVocab::_register_as_special_value(mask_value, &values, &mut special_value_indices).expect("Token index not found in vocabulary");

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_value_indices);

        Ok(BertVocab {
            values,
            indices,
            special_value_indices,
            special_indices
        })
    }
}

impl Vocab for BertVocab {
    fn values(&self) -> &BTreeMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &BTreeMap<i64, String> {
        &self.indices
    }

    fn special_values(&self) -> &BTreeMap<String, i64> {
        &self.special_value_indices
    }

    fn special_indices(&self) -> &BTreeMap<i64, String> {
        &self.special_indices
    }

    fn token_to_id(&self, token: &str) -> i64 {
        self._token_to_id(
            token,
            &self.values,
            &self.special_value_indices,
            "[UNK]"
        )
    }
    

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(id, &self.indices, &self.special_indices, "[UNK]")
    }
}

//==============================
// Unit tests
//==============================
// #[cfg(test)]
// mod tests {
//     use super::*;
//     extern crate anyhow;
//     use std::io::Write;

//     #[test]
//     fn test_create_object() {
//         //        Given
//         let values: BTreeMap<String, i64> = BTreeMap::new();
//         let special_values: BTreeMap<String, i64> = BTreeMap::new();
//         let indices: BTreeMap<i64, String> = BTreeMap::new();
//         let special_indices: BTreeMap<i64, String> = BTreeMap::new();
//         let unknown_value = BertVocab::unknown_value();

//         //        When
//         let base_vocab = BertVocab {
//             values,
//             indices,
//             unknown_value,
//             special_values,
//             special_indices,
//         };

//         //        Then
//         assert_eq!(base_vocab.unknown_value, "[UNK]");
//         assert_eq!(base_vocab.unknown_value, BertVocab::unknown_value());
//         assert_eq!(BertVocab::pad_value(), "[PAD]");
//         assert_eq!(BertVocab::sep_value(), "[SEP]");
//         assert_eq!(BertVocab::cls_value(), "[CLS]");
//         assert_eq!(BertVocab::mask_value(), "[MASK]");
//         assert_eq!(base_vocab.values, *base_vocab.values());
//         assert_eq!(base_vocab.special_values, *base_vocab.special_values());
//     }

//     #[test]
//     fn test_create_object_from_file() -> anyhow::Result<()> {
//         //        Given
//         let mut vocab_file = tempfile::NamedTempFile::new()?;
//         write!(
//             vocab_file,
//             "hello \n world \n [UNK] \n ! \n [CLS] \n [SEP] \n [MASK] \n [PAD]"
//         )?;
//         let path = vocab_file.into_temp_path();
//         let target_values: BTreeMap<String, i64> = [
//             ("hello".to_owned(), 0),
//             ("world".to_owned(), 1),
//             ("[UNK]".to_owned(), 2),
//             ("!".to_owned(), 3),
//             ("[CLS]".to_owned(), 4),
//             ("[SEP]".to_owned(), 5),
//             ("[MASK]".to_owned(), 6),
//             ("[PAD]".to_owned(), 7),
//         ]
//         .iter()
//         .cloned()
//         .collect();

//         let special_values: BTreeMap<String, i64> = [
//             ("[UNK]".to_owned(), 2),
//             ("[CLS]".to_owned(), 4),
//             ("[SEP]".to_owned(), 5),
//             ("[MASK]".to_owned(), 6),
//             ("[PAD]".to_owned(), 7),
//         ]
//         .iter()
//         .cloned()
//         .collect();

//         //        When
//         let base_vocab = BertVocab::from_file(path.to_path_buf().to_str().unwrap())?;

//         //        Then
//         assert_eq!(base_vocab.unknown_value, "[UNK]");
//         assert_eq!(base_vocab.values, target_values);
//         assert_eq!(base_vocab.special_values, special_values);
//         drop(path);
//         Ok(())
//     }

//     #[test]
//     #[should_panic]
//     fn test_create_object_from_file_without_unknown_token() {
//         //        Given
//         let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
//         write!(vocab_file, "hello \n world \n [UNK] \n ! \n [CLS]").unwrap();
//         let path = vocab_file.into_temp_path();

//         //        When & Then
//         let _base_vocab = BertVocab::from_file(path.to_path_buf().to_str().unwrap()).unwrap();
//     }

//     #[test]
//     fn test_encode_tokens() -> anyhow::Result<()> {
//         //        Given
//         let mut vocab_file = tempfile::NamedTempFile::new()?;
//         write!(
//             vocab_file,
//             "hello \n world \n [UNK] \n ! \n [CLS] \n [SEP] \n [MASK] \n [PAD]"
//         )?;
//         let path = vocab_file.into_temp_path();
//         let base_vocab = BertVocab::from_file(path.to_path_buf().to_str().unwrap())?;

//         //        When & Then
//         assert_eq!(base_vocab.token_to_id("hello"), 0);
//         assert_eq!(base_vocab.token_to_id("world"), 1);
//         assert_eq!(base_vocab.token_to_id("!"), 3);
//         assert_eq!(base_vocab.token_to_id("[UNK]"), 2);
//         assert_eq!(base_vocab.token_to_id("oov_value"), 2);
//         assert_eq!(base_vocab.token_to_id("[PAD]"), 7);
//         assert_eq!(base_vocab.token_to_id("[MASK]"), 6);
//         assert_eq!(base_vocab.token_to_id("[CLS]"), 4);
//         assert_eq!(base_vocab.token_to_id("[SEP]"), 5);

//         drop(path);
//         Ok(())
//     }

//     #[test]
//     fn test_decode_tokens() -> anyhow::Result<()> {
//         //        Given
//         let mut vocab_file = tempfile::NamedTempFile::new()?;
//         write!(
//             vocab_file,
//             "hello \n world \n [UNK] \n ! \n [CLS] \n [SEP] \n [MASK] \n [PAD]"
//         )?;
//         let path = vocab_file.into_temp_path();
//         let bert_vocab = BertVocab::from_file(path.to_path_buf().to_str().unwrap())?;

//         //        When & Then
//         assert_eq!(bert_vocab.id_to_token(&(0_i64)), "hello");
//         assert_eq!(bert_vocab.id_to_token(&(1_i64)), "world");
//         assert_eq!(bert_vocab.id_to_token(&(3_i64)), "!");
//         assert_eq!(bert_vocab.id_to_token(&(2_i64)), "[UNK]");
//         assert_eq!(bert_vocab.id_to_token(&(7_i64)), "[PAD]");
//         assert_eq!(bert_vocab.id_to_token(&(6_i64)), "[MASK]");
//         assert_eq!(bert_vocab.id_to_token(&(4_i64)), "[CLS]");
//         assert_eq!(bert_vocab.id_to_token(&(5_i64)), "[SEP]");

//         drop(path);
//         Ok(())
//     }
// }
