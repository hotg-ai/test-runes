#![no_std]

extern crate alloc;

use core::{convert::TryInto, fmt::Debug};

use alloc::vec::Vec;
use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};

/// A proc block which, when given a set of indices, will return their
/// associated labels.
///
/// # Examples
/// ```rust
/// # use label::Label;
/// # use hotg_rune_core::Tensor;
/// # use hotg_rune_proc_blocks::Transform;
/// let mut proc_block = Label::default();
/// proc_block.set_labels(["zero", "one", "two", "three"]);
/// let input = Tensor::new_vector(vec![3, 1, 2]);
///
/// let got = proc_block.transform(input);
///
/// assert_eq!(got.elements(), &["three", "one", "two"]);
/// ```
#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
pub struct Label {
    labels: Vec<&'static str>,
    label_numbering_strategy: Vec<&`static str>,

}
impl Default for Label {
    fn default()-> Self {
        Label::new()
    }
}

impl<T> Transform<Tensor<T>> for Label
where
    T: Copy + TryInto<f64>,
    <T as TryInto<f64>>::Error: Debug,
{
    type Output = Tensor<&'static str>;

    fn transform(&mut self, input: Tensor<T>) -> Self::Output {
        // let view = input
        //     .view::<1>()
        //     .expect("This proc block only supports 1D inputs");

        // let indices = view.elements().iter().copied().map(|ix| {
        //     ix.try_into()
        //         .expect("Unable to convert the index to a usize")
        // });
        if self.label_numbering_strategy =="one-based indexing"{
            let input = input
            .elements()
            .iter()
            .map(|x| (x - 1.0) as u32)
            .collect::<Vec<u32>>();
        }
        else {
            let input = input
            .elements()
            .iter()
            .map(|x| x)
            .collect::<Vec<u32>>();
        }

        let indices = input
            .iter()
            .map(|ix| ix.try_into().expect("Unable to convert the index to a f64"));

        // Note: We use a more cumbersome match statement instead of unwrap()
        // to provide the user with more useful error messages
        indices
            .map(|ix| match self.labels.get(ix as usize) {
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
    #[should_panic]
    fn only_works_with_1d_inputs() {
        let mut proc_block = Label::default();

        proc_block.set_output_dimensions(&[1, 2, 3]);
    }

    #[test]
    #[should_panic = "Index out of bounds: there are 2 labels but label 42 was requested"]
    fn label_index_out_of_bounds() {
        let mut proc_block = Label::default();
        proc_block.set_labels(["first", "second"]);
        let input = Tensor::new_vector(alloc::vec![0_usize, 42]);

        let _ = proc_block.transform(input);
    }

    #[test]
    fn get_the_correct_labels() {
        let mut proc_block = Label::default();
        proc_block.set_labels(["zero", "one", "two", "three"]);
        let input = Tensor::new_vector(alloc::vec![3, 1, 2]);
        let should_be = Tensor::new_vector(alloc::vec!["three", "one", "two"]);

        let got = proc_block.transform(input);

        assert_eq!(got, should_be);
    }
}
