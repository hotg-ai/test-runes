#![no_std]

extern crate alloc;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use libm::fabsf;

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct ObjectFilter {}

impl ObjectFilter {
    pub const fn new() -> Self {
        ObjectFilter {}
    }
}

impl Default for ObjectFilter {
    fn default() -> Self {
        ObjectFilter::new()
    }
}

impl Transform<Tensor<f32>> for ObjectFilter {
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<f32> {
        let input: Vec<f32> = input.elements().iter().map(|&x| x).collect();

        let mut vec_2d: Vec<Vec<f32>> = Vec::new();

        let mut j = 0;
        let dimensions = 85;
        let confidence_index = 4;
        let label_start_index = 5;

        for i in 0..6300 {
            let index = i * dimensions;

            if j != 0 {
                let rows = vec_2d.len();
                let mut x = 0;
                for i in 0..rows {
                    if fabsf(input[index] - vec_2d[i][0]) <= 0.01
                        && fabsf(input[index + 1] - vec_2d[i][1]) <= 0.01
                    {
                        x = 1;
                        continue;
                    }
                }
                if x == 1 {
                    continue;
                }
            }
            if input[index + confidence_index] <= 0.7f32 {
                continue;
            }

            let (ind, value) = &input[index + label_start_index..index + dimensions]
                .iter()
                .enumerate()
                .fold(
                    (0, 0.0),
                    |max, (ind, &val)| if val > max.1 { (ind, val) } else { max },
                );

            if value <= &0.7f32 {
                continue;
            }
            vec_2d.push(vec![]);
            vec_2d[j].push(input[index]); // x-coordinate
            vec_2d[j].push(input[index + 1]); // y-coordinate
            vec_2d[j].push(input[index + 2]); // h
            vec_2d[j].push(input[index + 3]); // w
            vec_2d[j].push(*value as f32); // label confidence values
            vec_2d[j].push(*ind as f32); // label index

            j = j + 1;
        }

        let rows = vec_2d.len();
        let columns = vec_2d[0].len();

        let elements: Arc<[f32]> = vec_2d
            .into_iter()
            .flat_map(|v: Vec<f32>| v.into_iter())
            .collect();

        Tensor::new_row_major(elements, alloc::vec![rows, columns])
    }
}
