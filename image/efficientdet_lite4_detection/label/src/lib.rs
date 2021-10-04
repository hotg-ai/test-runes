#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::{convert::TryInto, fmt::Debug};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use itertools::Either;
use num::One;

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Label {
    labels: Vec<&'static str>,
    class_index_numbering: &'static str,
}

impl Label {
    pub fn new(labels: Vec<&'static str>, class_index_numbering: &'static str) -> Self {
        Label {
            labels: labels,
            class_index_numbering: class_index_numbering,
        }
    }
}

impl Default for Label {
    fn default() -> Self {
        Label::new(vec![" "], "zero-based indexing")
    }
}

impl<T> Transform<Tensor<T>> for Label
where
    T: Copy + TryInto<f64> + One + core::ops::Sub<Output = T>,
    <T as TryInto<f64>>::Error: Debug,
{
    type Output = Tensor<&'static str>;

    fn transform(&mut self, input: Tensor<T>) -> Self::Output {
        let indices = if self.class_index_numbering == "one-based indexing" {
            Either::Left(
                input
                    .elements()
                    .iter()
                    .map(|&x| (x - T::one()))
                    .map(|ix| ix.try_into().expect("Unable to convert the index to a f64")),
            )
        } else {
            Either::Right(
                input
                    .elements()
                    .iter()
                    .map(|&x| (x))
                    .map(|ix| ix.try_into().expect("Unable to convert the index to a f64")),
            )
        };

        // Note: We use a more cumbersome match statement instead of unwrap()
        // to provide the user with more useful error messages

        indices
            .map(|ix| match self.labels.get(ix as usize) {
                Some(&label) => label,
                None => panic!("Index out of bounds: there are  labels but label was requested"),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_the_correct_labels() {
        let mut proc_block = Label::new(vec!["zero", "one", "two", "three"], "one-based indexing");
        // proc_block.set_labels(["zero", "one", "two", "three"]);
        let input = Tensor::new_vector(alloc::vec![3, 1, 2]);
        let should_be = Tensor::new_vector(alloc::vec!["two", "zero", "one"]);

        let got = proc_block.transform(input);

        assert_eq!(got, should_be);
    }
}
