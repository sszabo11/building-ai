use std::env::home_dir;

use image::{ImageBuffer, Rgba};
use linalg::Matrix;

use crate::net::Network;

pub struct ConvNet {
    pub layers: Vec<ConvLayer>,
    pub fully_connected: Network,
}

impl ConvNet {
    pub fn train(&mut self, training_data: Vec<(Vec<Matrix>, Matrix)>, epochs: usize) {
        let num_layers = self.layers.len();
        println!("training...");

        //for epoch in 0..epochs {
        let (x_train, y_train) = &training_data[0];
        println!("x tr: {:?}", x_train);

        let mut input = x_train;

        for l in 0..num_layers {
            println!("Layer: {}", l);
            let layer = &self.layers[l];

            let out = layer.run(input);
            generate_img(&out, l);
        }
        //}
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
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            bias: vec![0.; num_filters],
            kernel: vec![Matrix::random(kernel_size * channels, kernel_size); num_filters],
            input_layer: true,
        }
    }

    pub fn new(num_filters: usize, stride: usize, kernel_size: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            bias: vec![0.; num_filters],
            kernel: vec![Matrix::random(kernel_size, kernel_size); num_filters],
            input_layer: true,
        }
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

        println!("h in: {}", h_in);
        let h_out = (h_in + 2 * padding - k) / stride + 1;
        let w_out = (w_in + 2 * padding - k) / stride + 1;
        println!("h out: {} | w out: {}", h_out, w_out);
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
                        output[f].data[i * w_out + j] = sum;
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
