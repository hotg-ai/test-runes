#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::{convert::TryInto, fmt::Debug};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

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
    T: Copy + num_traits::ToPrimitive + TryInto<f64>,
    <T as TryInto<f64>>::Error: Debug,
{
    type Output = Tensor<&'static str>;

    fn transform(&mut self, input: Tensor<T>) -> Self::Output {
        // let view = input
        //     .view::<1>()
        //     .expect("This proc block only supports 1D inputs");

        let indices = input
            .elements()
            .iter()
            .copied()
            .map(|ix| {
                ix.to_usize()
                    .expect("Unable to convert the index to a usize")
            })
            .map(|ix| offset(ix, self.class_index_numbering));

        // Note: We use a more cumbersome match statement instead of unwrap()
        // to provide the user with more useful error messages
        indices
            .map(|ix| match self.labels.get(ix) {
                Some(&label) => label,
                None => panic!(
                    "Index out of bounds: there are {} labels but label {} was requested",
                    self.labels.len(),
                    ix
                ),
            })
            .collect()
    }
}

fn offset(original_index: usize, class_index_numbering: &str) -> usize {
    match class_index_numbering {
        "one-based indexing" => original_index - 1,
        "zero-based indexing" => original_index,
        _ => unreachable!("The class_index_numbering should be either \"one-based indexing\" or \"zero-based indexing\"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_the_correct_labels() {
        let mut proc_block = Label::new(vec!["zero", "one", "two", "three"], "one-based indexing");
        // proc_block.set_labels(["zero", "one", "two", "three"]);
        let input = Tensor::new_vector(alloc::vec![3.0, 1.0, 2.0]);
        let should_be = Tensor::new_vector(alloc::vec!["two", "zero", "one"]);

        let got = proc_block.transform(input);

        assert_eq!(got, should_be);
    }
}
