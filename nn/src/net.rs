use std::f32;

use linalg::Matrix;

pub struct Network {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub activation: Activation,
}

impl Network {
    pub fn new(layers: Vec<usize>, input_dim: usize, activation: Activation) -> Self {
        let weights = (0..layers.len())
            .map(|layer| {
                let size = layers[layer];
                let prev = if layer == 0 {
                    input_dim
                } else {
                    layers[layer - 1]
                };
                Matrix::random(size, prev)
            })
            .collect();

        let biases = layers
            .iter()
            .map(|&num_n| Matrix::random(num_n, 1))
            .collect();

        Self {
            layers,
            weights,
            biases,
            activation,
        }
    }

    fn activate_fn(&self, x: Matrix) -> Matrix {
        let mut result = Matrix::new(x.rows, x.cols);

        for i in 0..x.rows {
            for j in 0..x.cols {
                let index = i * x.cols + j;
                result.data[index] = calc_activate_fn(&self.activation, x.data[index]);
            }
        }
        result
    }

    pub fn forward(&self, x: Matrix) -> Matrix {
        let mut prev_layer = x;
        for l in 0..self.layers.len() {
            assert!(
                self.weights[l].cols == prev_layer.rows,
                "Weights: [{} x {}] | Input shape: [{} x {}]",
                self.weights[l].rows,
                self.weights[l].cols,
                prev_layer.rows,
                prev_layer.cols,
            );

            println!("Layer: [{}]", l);
            let a = self.weights[l].dot(&prev_layer);
            let added = a.add(&self.biases[l]);
            let act = self.activate_fn(added);
            println!("out: {}", act.pretty_shape());
            prev_layer = act;
        }

        prev_layer
    }
}

pub enum Activation {
    Sigmoid,
    ReLu,
    Tanh,
}

fn calc_activate_fn(func: &Activation, x: f32) -> f32 {
    match func {
        Activation::Sigmoid => sigmoid(x),
        _ => {
            panic!("Not implemented")
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + f32::consts::E.powf(-x))
}
