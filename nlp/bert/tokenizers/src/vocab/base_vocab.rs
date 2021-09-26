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
use alloc::{collections::BTreeMap, string::String};
use crate::alloc::string::ToString;


#[derive(Debug, Clone)]
pub enum TokenError {
    TokenNotFound{
        word: String,
    },
}

/// # Base Vocab trait
/// Defines a common interface to the vocabularies for use in the tokenizers.
pub trait Vocab {
    // Associative function returning the unknown value for the vocabulary
    // fn unknown_value() -> &'static str;

    // / Returns the unknown value on an instance
    // fn get_unknown_value(&self) -> &'static str;

    /// Return the map of token strings to IDs
    fn values(&self) -> &BTreeMap<String, i64>;

    // Return the map of token IDs to strings
    // fn indices(&self) -> &BTreeMap<i64, String>;

    // Return the map of token strings to IDs
    fn special_values(&self) -> &BTreeMap<String, i64>;


    /// Converts a token to an id, provided a `BTreeMap` of values, a `BTreeMap` of special values and
    /// the unknown value token string representation. This is not meant to be directly used, the method
    /// `token_to_id` offers a more convenient interface for most vocabularies, but needs to be implemented
    /// by the specific vocabulary.
    ///
    /// # Parameters
    /// - token (`&str`): token to convert
    /// - values (`&BTreeMap<String, i64>`): mapping from tokens to ids
    /// - special_values (`&BTreeMap<String, i64>`): mapping from special tokens to ids
    /// - unknown_value (`&str`): unknown token value
    ///
    /// # Returns
    /// - `i64`: index value for the provided token
    fn _token_to_id(
        &self,
        token: &str,
        values: &BTreeMap<String, i64>,
        special_values: &BTreeMap<String, i64>,
        unknown_value: &str,
    ) -> i64 {
        match special_values.get(token) {
            Some(index) => *index,
            None => match values.get(token) {
                Some(index) => *index,
                None => *values.get(unknown_value).unwrap(),
            },
        }
    }

    /// Converts a token to an id.
    ///
    /// # Parameters
    /// - token (`&str`): token to convert
    ///
    /// # Returns
    /// - `i64`: token index for the value provided. If not found in the indices, returns the unknown token index
    fn token_to_id(&self, token: &str) -> i64;

    /// Register a token as a special value
    ///
    /// # Parameters
    /// - token (`&str`): token to register as a special value
    /// - values (`&BTreeMap<String, i64>`): mapping from tokens to ids. This should contain the token to add and will be used to read the id for registration in `special_values`
    /// - special_values (`&BTreeMap<String, i64>`): mapping from special tokens to ids
    fn _register_as_special_value(
        token: &str,
        values: &BTreeMap<String, i64>,
        special_values: &mut BTreeMap<String, i64>,
    ) -> Result<(), TokenError> {
        let token_id = match values.get(token) {
            Some(index) => *index,
            None => {
                return Err(
                    TokenError::TokenNotFound{word:token.to_string()}
                );
            }
        };
        special_values.insert(String::from(token), token_id);
        Ok(())
    }

}

//==============================
// Unit tests
//==============================
// #[cfg(test)]
// mod tests {
//     extern crate anyhow;

//     use super::*;
//     use std::io::Write;

//     #[test]
//     fn test_create_object() {
//         //        Given
//         let values: BTreeMap<String, i64> = BTreeMap::new();
//         let special_values: BTreeMap<String, i64> = BTreeMap::new();
//         let indices: BTreeMap<i64, String> = BTreeMap::new();
//         let special_indices: BTreeMap<i64, String> = BTreeMap::new();
//         let unknown_value = BaseVocab::unknown_value();

//         //        When
//         let base_vocab = BaseVocab {
//             values,
//             indices,
//             unknown_value,
//             special_values,
//             special_indices,
//         };

//         //        Then
//         assert_eq!(base_vocab.unknown_value, "[UNK]");
//         assert_eq!(base_vocab.unknown_value, BaseVocab::unknown_value());
//         assert_eq!(base_vocab.values, *base_vocab.values());
//         assert_eq!(base_vocab.special_values, *base_vocab.special_values());
//     }

//     #[test]
//     fn test_create_object_from_file() -> anyhow::Result<()> {
//         //        Given
//         let mut vocab_file = tempfile::NamedTempFile::new()?;
//         write!(vocab_file, "hello \n world \n [UNK] \n !")?;
//         let path = vocab_file.into_temp_path();
//         let target_values: BTreeMap<String, i64> = [
//             ("hello".to_owned(), 0),
//             ("world".to_owned(), 1),
//             ("[UNK]".to_owned(), 2),
//             ("!".to_owned(), 3),
//         ]
//         .iter()
//         .cloned()
//         .collect();

//         let special_values: BTreeMap<String, i64> =
//             [("[UNK]".to_owned(), 2)].iter().cloned().collect();

//         //        When
//         let base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap())?;

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
//         write!(vocab_file, "hello \n world \n !").unwrap();
//         let path = vocab_file.into_temp_path();

//         //        When & Then
//         let _base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap()).unwrap();
//     }

//     #[test]
//     fn test_encode_tokens() -> anyhow::Result<()> {
//         //        Given
//         let mut vocab_file = tempfile::NamedTempFile::new()?;
//         write!(vocab_file, "hello \n world \n [UNK] \n !")?;
//         let path = vocab_file.into_temp_path();
//         let base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap())?;

//         //        When & Then
//         assert_eq!(base_vocab.token_to_id("hello"), 0);
//         assert_eq!(base_vocab.token_to_id("world"), 1);
//         assert_eq!(base_vocab.token_to_id("!"), 3);
//         assert_eq!(base_vocab.token_to_id("[UNK]"), 2);
//         assert_eq!(base_vocab.token_to_id("oov_value"), 2);

//         drop(path);
//         Ok(())
//     }

//     #[test]
//     fn test_decode_tokens() -> anyhow::Result<()> {
//         //        Given
//         let mut vocab_file = tempfile::NamedTempFile::new()?;
//         write!(vocab_file, "hello \n world \n [UNK] \n !")?;
//         let path = vocab_file.into_temp_path();
//         let base_vocab = BaseVocab::from_file(path.to_path_buf().to_str().unwrap())?;

//         //        When & Then
//         assert_eq!(base_vocab.id_to_token(&(0_i64)), "hello");
//         assert_eq!(base_vocab.id_to_token(&(1_i64)), "world");
//         assert_eq!(base_vocab.id_to_token(&(3_i64)), "!");
//         assert_eq!(base_vocab.id_to_token(&(2_i64)), "[UNK]");

//         drop(path);
//         Ok(())
//     }
// }
