use core::num;

use linalg::Matrix;
use rand_distr::Normal;

pub struct Network {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub activation: Activation,

    pub grad_w: Vec<Vec<Matrix>>,
    pub a: Vec<Matrix>, // Activation functions. a = σ(Wx + b)
    pub z: Vec<Matrix>, // Activation functions. z = Wx + b
}

impl Network {
    pub fn new(layers: Vec<usize>, input_dim: usize, activation: Activation) -> Self {
        let normal = Normal::new(0., (2.0 / input_dim as f32).sqrt()).unwrap();
        let weights: Vec<Matrix> = (0..layers.len())
            .map(|layer| {
                let size = layers[layer];
                let prev = if layer == 0 {
                    input_dim
                } else {
                    layers[layer - 1]
                };
                Matrix::sample(size, prev, normal)
            })
            .collect();

        let biases = layers
            .iter()
            .map(|&num_n| Matrix::zeros(num_n, 1))
            .collect();

        Self {
            grad_w: Vec::new(),
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

    pub fn backward(&mut self, x_input: &Matrix, y: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        assert!(y.cols == 1);

        let num_layers = self.layers.len();

        let mut weight_grads = Vec::with_capacity(num_layers);
        let mut bias_grads = Vec::with_capacity(num_layers);

        //println!(
        //    "y: {} | a: {}",
        //    y.pretty_shape(),
        //    self.a[num_layers - 1].pretty_shape()
        //);

        // meas sqaure loss
        //let mut grad_a = self.a[num_layers - 1].subtract(y);
        let mut grad_a = self.a[num_layers - 1].subtract(y);

        for v in grad_a.data.iter_mut() {
            if *v > 0.0 {
                *v = 1.0;
            } else if *v < 0.0 {
                *v = -1.0;
            } else {
                *v = 0.0;
            }
        }

        for l in (0..num_layers).rev() {
            // ∂L/∂a = (a - y)
            let a = &self.a[l];
            //let one_minus_a = Matrix::ones(a.rows, a.cols).subtract(a);

            // ∂L/∂z = a(1 - a)
            // = grad_a * a(1 - a)
            //let grad_z = grad_a.mul(&a.mul(&one_minus_a)); // sigmoid

            //let grad_z = grad_a.clone(); // linear

            //tanh
            let a_sq = a.mul(a); // a²
            let one_minus_a_sq = Matrix::ones(a.rows, a.cols).subtract(&a_sq);
            let grad_z = grad_a.mul(&one_minus_a_sq);

            //// relu
            //let mut relu_mask = a.clone();
            //for v in relu_mask.data.iter_mut() {
            //    *v = if *v > 0.0 { 1.0 } else { 0.01 };
            //}

            //// ∂L/∂z = grad_a ⊙ relu_mask
            //let grad_z = grad_a.mul(&relu_mask);

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
        //self.grad_w.push(weight_grads.clone());
        bias_grads.reverse();
        (weight_grads, bias_grads)
    }

    pub fn loss(&self, out: &Matrix, y: &Matrix) -> f32 {
        let mut diff = y.t().subtract(out);

        diff.pow_assign(2.0);

        let total_loss = diff.data.iter().sum::<f32>();
        total_loss / y.data.len() as f32
    }

    pub fn l1_loss(&self, out: &Matrix, target: &Matrix) -> f32 {
        let mut sum = 0.0;
        for (o, t) in out.data.iter().zip(target.data.iter()) {
            sum += (o - t).abs();
        }
        sum / out.data.len() as f32
    }

    pub fn forward(&mut self, x: &Matrix, pr: bool) -> Matrix {
        self.a.clear();
        self.z.clear();
        let mut prev_layer = x.clone();
        for l in 0..self.layers.len() {
            if pr {
                println!("layer: {}", l);
            }
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
    Linear,
}

pub fn calc_activate_fn(func: &Activation, x: f32) -> f32 {
    match func {
        Activation::Sigmoid => sigmoid(x),
        Activation::Linear => linear(x),
        Activation::ReLu => relu(x),
        Activation::Tanh => tanh(x),
        _ => {
            panic!("Not implemented")
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-x))
}
fn linear(x: f32) -> f32 {
    x.clamp(-10., 10.)
}
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.01 }
}
fn tanh(x: f32) -> f32 {
    x.tanh()
    //let exp_x = x.exp();
    //let exp_nx = (-x).exp();
    //(exp_x - exp_nx) / (exp_x + exp_nx)
}

pub fn softmax(val: &Matrix) -> Vec<f32> {
    assert!(val.cols == 1);

    let max_val = val.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = val.data.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    exps.iter().map(|e| e / sum).collect()
}
