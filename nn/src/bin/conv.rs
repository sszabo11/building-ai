use image::{ImageBuffer, Rgb, Rgba};
use linalg::Matrix;
use nn::{
    cnn::ConvLayer,
    data::{ImageRecord, IrisRecord, read_img_labels, read_iris_data},
    net::{Activation, Network, softmax},
    utils::one_hot,
};

const IMG_SIZE: usize = 16;
const INPUT_DIM: usize = IMG_SIZE * IMG_SIZE;
fn main() {
    let layers = vec![5, 50, INPUT_DIM];

    let net = Network::new(layers.clone(), 5, Activation::Sigmoid);

    let conv = ConvLayer::builder();
    let conv1 = ConvLayer::input(10, 2, 4, 4, 0);
    let conv2 = ConvLayer::new(10, 1, 3, 0);

    let mut cnn = conv.layer(conv1).layer(conv2).fully_connected(net).build();

    let training_data = read_img_labels().unwrap();
    let training_data = parse_data(training_data);

    cnn.train(training_data, 100);

    //train(&mut net, 50000, &training_data);

    //generate_img(&mut net, &training_data[0].1);
}

fn parse_data(data: Vec<ImageRecord>) -> Vec<(Vec<Matrix>, Matrix)> {
    let mut parsed: Vec<(Vec<Matrix>, Matrix)> = Vec::new();
    for record in data.iter() {
        let channels: [Vec<f32>; 4] = record.pxs.iter().fold(
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            |mut acc, (r, g, b, a)| {
                acc[0].push(*r);
                acc[1].push(*g);
                acc[2].push(*b);
                acc[3].push(*a);
                acc
            },
        );

        let mut x_in = vec![Matrix::zeros(16, 16); 4];
        for channel in 0..4 {
            for y in 0..IMG_SIZE {
                for x in 0..IMG_SIZE {
                    let px = y * IMG_SIZE + x;

                    x_in[channel].data[px] = channels[channel][px];
                }
            }
        }

        let mut y = Matrix::zeros(1, 5);
        y.data = record.label.clone();
        parsed.push((x_in, y));
    }

    parsed
}
