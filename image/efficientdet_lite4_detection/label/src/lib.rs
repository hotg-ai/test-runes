#![no_std]

extern crate alloc;

use alloc::prelude::v1::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::{convert::TryInto, fmt::Debug};
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};
use itertools::Either;
use num::One;
use num::Zero;

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
    T: Copy + TryInto<f64> + One + Zero + core::ops::Sub<Output = T>,
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

impl HasOutputs for Label {
    fn set_output_dimensions(&mut self, dimensions: &[usize]) {
        match dimensions {
            [rest @ .., _] if rest.iter().all(|d| *d == 1) => {}
            _ => {
                panic!(
                    "This proc block only supports 1D outputs (requested output: {:?})",
                    dimensions
                );
            }
        }
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
