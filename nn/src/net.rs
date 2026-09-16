use std::f32;

use linalg::Matrix;

pub struct Network {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub activation: Activation,

    a: Vec<Matrix>, // Activation functions. a = σ(Wx + b)
    z: Vec<Matrix>, // Activation functions. z = Wx + b
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
            a: Vec::new(),
            z: Vec::new(),
        }
    }

    fn activate_fn(&self, x: &Matrix) -> Matrix {
        let mut result = Matrix::zeros(x.rows, x.cols);

        for i in 0..x.rows {
            for j in 0..x.cols {
                let index = i * x.cols + j;
                result.data[index] = calc_activate_fn(&self.activation, x.data[index]);
            }
        }
        result
    }

    pub fn backward(&self, x_input: &Matrix, y: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        assert!(y.cols == 1);

        let num_layers = self.layers.len();

        let mut weight_grads = Vec::with_capacity(num_layers);
        let mut bias_grads = Vec::with_capacity(num_layers);

        //println!(
        //    "y: {} | a: {}",
        //    y.pretty_shape(),
        //    self.a[num_layers - 1].pretty_shape()
        //);
        let mut grad_a = self.a[num_layers - 1].subtract(y);

        for l in (0..num_layers).rev() {
            // ∂L/∂a = (a - y)
            let a = &self.a[l];
            let one_minus_a = Matrix::ones(a.rows, a.cols).subtract(a);

            // ∂L/∂z = a(1 - a)
            // = grad_a * a(1 - a)
            let grad_z = grad_a.mul(&a.mul(&one_minus_a));

            // ∂L/∂w
            // = x (prev a)
            let input = if l == 0 { &x_input } else { &self.a[l - 1] };
            let grad_w = grad_z.dot(&input.t()); // outer product.

            // ∂L/∂b
            // = 1
            let grad_b = grad_z.clone();

            weight_grads.push(grad_w);
            bias_grads.push(grad_b);
            if l > 0 {
                grad_a = self.weights[l].t().dot(&grad_z);
            };
        }
        weight_grads.reverse();
        bias_grads.reverse();
        (weight_grads, bias_grads)
    }

    pub fn loss(&self, y: &Matrix) -> f32 {
        let diff = y.t().subtract(&self.a[self.layers.len() - 1]).pow(2.0);
        let total_loss = diff.data.iter().sum::<f32>();
        total_loss / y.rows as f32
    }

    pub fn forward(&mut self, x: Matrix) -> Matrix {
        self.a.clear();
        self.z.clear();
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

            // = Wx
            let wx = self.weights[l].dot(&prev_layer);

            // z = Wx + b
            // ∂z/∂w = x
            // ∂z/∂x = w
            // ∂z/∂b = 1
            let z = wx.add(&self.biases[l]);

            // a = σ(z)
            // ∂a/∂z = a(1 - a)
            let a = self.activate_fn(&z);

            self.a.push(a.clone());
            self.z.push(z.clone());

            prev_layer = a;
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
