use image::{ImageBuffer, Rgba};
use linalg::Matrix;

use crate::net::{Activation, Network, calc_activate_fn};

pub struct ConvNet {
    pub layers: Vec<ConvLayer>,
    pub fully_connected: Network,
}

impl ConvNet {
    pub fn train(&mut self, training_data: &[(Vec<Matrix>, Matrix)], epochs: usize) {
        let num_layers = self.layers.len();
        println!("training...");

        let lr = 0.001;
        for epoch in 0..epochs {
            for sample in 0..training_data.len() {
                let (x_train, y_train) = &training_data[sample];

                let mut input = x_train.clone();

                for l in 0..num_layers {
                    println!("Layer: {}", l);
                    let layer = &self.layers[l];
                    //println!("{} {}:", input.len(), input[0].pretty_shape());

                    let out = layer.run(&input);

                    generate_img(&out, l);
                    input = out;

                    let loss = self.loss(&out, &y_train);

                    if epoch % 10 == 0 {
                        println!("Epoch {} | Loss: {}", epoch, loss);
                    }

                    for f in 0..layer.kernel.len() {
                        let (weight_grads, bias_grads) = layer.backward(&x_train, &y_train.t());
                        layer.kernel[f] = layer.kernel[f].subtract(&weight_grads[f].scale(lr));
                        layer.bias[f] = layer.bias[f] - (&bias_grads[f].scale(lr));
                    }
                }
            }
        }
    }
}

pub struct ConvLayer {
    pub kernel: Vec<Matrix>, // 4d
    // filters x channels x width x height
    // filters x [channels * width, height]
    pub stride: usize,
    pub kernel_size: usize,
    pub padding: usize,
    pub bias: Vec<f32>,
    pub input_layer: bool,
    pub activation: Option<Activation>,

    pub a: Option<Matrix>, // Activation functions. a = σ(Wx + b)
    pub z: Option<Matrix>, // Activation functions. z = Wx + b
}

pub struct ConvBuilder {
    pub layers: Vec<ConvLayer>,
    pub fully_connected: Option<Network>,
}

impl ConvBuilder {
    pub fn layer(mut self, layer: ConvLayer) -> Self {
        self.layers.push(layer);
        self
    }
    pub fn fully_connected(mut self, net: Network) -> Self {
        self.fully_connected = Some(net);
        self
    }

    pub fn build(self) -> ConvNet {
        ConvNet {
            layers: self.layers,
            fully_connected: self
                .fully_connected
                .expect("No fully connected layer attached"),
        }
    }
}

impl ConvLayer {
    pub fn builder() -> ConvBuilder {
        ConvBuilder {
            layers: vec![],
            fully_connected: None,
        }
    }
    pub fn input(
        num_filters: usize,
        stride: usize,
        kernel_size: usize,
        channels: usize,
        padding: usize,
        activation: Option<Activation>,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            bias: vec![0.; num_filters],
            kernel: vec![Matrix::random(kernel_size * channels, kernel_size); num_filters],
            input_layer: true,
            activation,
            a: None,
            z: None,
        }
    }

    pub fn new(
        num_filters: usize,
        stride: usize,
        kernel_size: usize,
        channels: usize,
        padding: usize,
        activation: Option<Activation>,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            bias: vec![0.; num_filters],
            kernel: vec![Matrix::random(kernel_size * channels, kernel_size); num_filters],
            input_layer: false,
            activation,
            a: None,
            z: None,
        }
    }

    pub fn cross_entropy_loss(&self, target_y: Matrix, pred_y: Matrix) -> f32 {
        assert!(target_y.rows == pred_y.rows);
        assert!(target_y.cols == pred_y.cols);

        let loss: f32 = -(0..target_y.rows * target_y.cols)
            .map(|i| target_y.data[i] * pred_y.data[i].ln())
            .sum::<f32>();

        loss
    }

    pub fn backward(&mut self, x_input: &Matrix, y: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        assert!(y.cols == 1);

        let mut weight_grads = Vec::with_capacity(y.rows);
        let mut bias_grads = Vec::with_capacity(y.rows);

        let mut grad_a = x_input.subtract(&y);

        let a = self.a.as_ref().unwrap();
        let num_features = self.kernel.len();

        for f in 0..num_features {
            // relu

            let mut relu_mask = a.clone();
            for v in relu_mask.data.iter_mut() {
                *v = if *v > 0.0 { 1.0 } else { 0.01 };
            }

            // ∂L/∂z = grad_a ⊙ relu_mask
            let grad_z = grad_a.mul(&relu_mask);

            // ∂L/∂w
            // = x (prev a)
            let input = if self.input_layer { &x_input } else { a };
            let grad_w = grad_z.dot(&input.t()); // outer product.

            // ∂L/∂b
            // = 1
            let grad_b = grad_z.clone();

            weight_grads.push(grad_w);
            bias_grads.push(grad_b);

            if !self.input_layer {
                grad_a = self.kernel[f].t().dot(&grad_z);
            };
        }

        weight_grads.reverse();
        bias_grads.reverse();
        (weight_grads, bias_grads)
    }

    pub fn activation(&mut self, z: &Vec<Matrix>) -> Vec<Matrix> {
        let h_out = z[0].rows;
        let w_out = z[0].cols;

        let num_features = z.len();
        let mut a = vec![Matrix::zeros(h_out, w_out); num_features];

        for f in 0..num_features {
            for i in 0..h_out {
                for j in 0..w_out {
                    let z_i = z[f].data[i * w_out + j];
                    if let Some(func) = &self.activation {
                        let x = calc_activate_fn(&func, z_i);
                        a[f].data[i * w_out + j] = x;
                    } else {
                        a[f].data[i * w_out + j] = z_i;
                    }
                }
            }
        }
        a
    }
    pub fn forward(&mut self, input: &Vec<Matrix>) -> Vec<Matrix> {
        let z = self.run(input);
        let a = self.activation(&z);

        a
    }
    pub fn run(&self, input: &Vec<Matrix>) -> Vec<Matrix> {
        let rows = input[0].rows;
        let cols = input[0].cols;

        let num_filters = self.kernel.len();

        let c_in = input.len(); // Num channels
        let h_in = input[0].rows; // Height
        let w_in = input[0].cols; // Width

        let padding = self.padding;
        let stride = self.stride;
        let k = self.kernel_size;

        let h_out = (h_in + 2 * padding - k) / stride + 1;
        let w_out = (w_in + 2 * padding - k) / stride + 1;
        assert!(h_out < 100_000);
        assert!(w_out < 100_000);

        let mut output = vec![Matrix::zeros(h_out, w_out); num_filters];

        // each filter/kernel
        for f in 0..num_filters {
            // each pixel in input
            for i in 0..h_out {
                for j in 0..w_out {
                    let mut sum = self.bias[f];

                    // each channel (r,g,b,a)
                    for c in 0..c_in {
                        // each cell in kernel
                        for ki in 0..k {
                            for kj in 0..k {
                                let hi = i * stride + ki - padding;
                                let wi = j * stride + kj - padding;

                                if hi < h_in && wi < w_in {
                                    let kernel = self.kernel[f].data[c * ki + kj];
                                    sum += input[c].data[hi * rows + wi] * kernel;
                                }
                            }
                        }
                        //if let Some(func) = &self.activation {
                        //    let x = calc_activate_fn(&func, sum);
                        //    output[f].data[i * w_out + j] = x;
                        //} else {
                        output[f].data[i * w_out + j] = sum;
                        //}
                    }
                }
            }
        }
        output
    }
}

fn generate_img(channels: &Vec<Matrix>, id: usize) {
    let h_in = channels[0].rows;
    let w_in = channels[0].cols;

    let mut img = ImageBuffer::new(h_in as u32, w_in as u32);

    // Iterate over the mutable pixels and assign colors
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let mut px: [u8; 4] = [0, 0, 0, 0];

        for channel in 0..4 {
            let i = y as usize * w_in + x as usize;

            px[channel] = (channels[channel].data[i] * 255.).round() as u8
        }
        // Assign the RGB values to the pixel
        *pixel = Rgba(px);
    }
    img.save(&format!("./features/feature-{}.png", id)).unwrap();
}
