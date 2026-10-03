use linalg::Matrix;
use nn::{
    data::{ImageRecord, IrisRecord, read_img_labels, read_iris_data},
    net::{Activation, Network, softmax},
    utils::one_hot,
};

const IMG_SIZE: usize = 16;
fn main() {
    let layers = vec![IMG_SIZE * IMG_SIZE, 30, 4];

    let mut net = Network::new(layers, 4, Activation::Sigmoid);
    //println!("{:?}", net.weights[0].data);

    train(&mut net, 5000);
}

fn parse_data(data: Vec<ImageRecord>) -> Vec<(Matrix, Matrix)> {
    let mut parsed: Vec<(Matrix, Matrix)> = Vec::new();
    for record in data.iter() {
        let channels: [Vec<f32>; 4] = record.pxs.iter().fold(
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            |mut acc, (r, g, b, a)| {
                acc[0].push(*r as f32);
                acc[1].push(*g as f32);
                acc[2].push(*b as f32);
                acc[3].push(*a as f32);
                acc
            },
        );

        let mut x_in = Matrix::zeros(4, IMG_SIZE * IMG_SIZE);
        for channel in 0..4 {
            for y in 0..IMG_SIZE {
                for x in 0..IMG_SIZE {
                    let i = (channel * IMG_SIZE * IMG_SIZE) + (y * IMG_SIZE) + x;
                    let px = y * IMG_SIZE + x;

                    x_in.data[i] = channels[channel][px];
                }
            }
        }

        let mut y = Matrix::zeros(1, 5);
        y.data = record.label.clone();
        parsed.push((x_in, y));
    }

    parsed
}
fn train(net: &mut Network, epochs: usize) {
    let lr = 0.1;

    let training_data = read_img_labels().unwrap();
    let training_data = parse_data(training_data);
    let training_samples = training_data.len();

    let (x_train, y_train): (Vec<Matrix>, Vec<Matrix>) = training_data.clone().into_iter().unzip();
    println!("{}", x_train[0].pretty_shape());

    for epoch in 0..epochs {
        for sample in 0..x_train.len() {
            let x = &x_train[sample];
            let y = &y_train[sample];

            let out = net.forward(x);

            let loss = net.loss(&y);
            if epoch % 10 == 0 && sample == 1 {
                println!("Epoch {} | Loss: {}", epoch, loss);
            }

            let (weight_grads, bias_grads) = net.backward(&x.t(), &y.t());
            //{
            //    let l = net.layers.len() - 1;
            //    net.weights[l] = net.weights[l].subtract(&weight_grads[l].scale(lr));
            //    net.biases[l] = net.biases[l].subtract(&bias_grads[l].scale(lr));
            //}
            //for l in (net.layers.len() - 1)..net.layers.len() {
            for l in 0..net.layers.len() {
                net.weights[l] = net.weights[l].subtract(&weight_grads[l].scale(lr));
                net.biases[l] = net.biases[l].subtract(&bias_grads[l].scale(lr));
            }
        }
    }

    let mut correct = 0;
    for sample in 0..training_samples {
        let (x, y) = &training_data[sample];

        let logits = net.forward(x);
        let max_index = logits
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();

        let probs = softmax(&logits);
        println!("{:?}", probs);
        //println!(
        //    "Raw: [{}, {}, {}]\nTrue: [{:?}]\nOutput: {}\n",
        //    out.data[0], out.data[1], out.data[2], y.data, SPECIES_LIST[max_index]
        //);
        let y_pos = y.data.iter().position(|&y| y == 1.).unwrap();
        if max_index == y_pos {
            correct += 1;
        }
    }
    let acc = correct as f32 / training_samples as f32 * 100.;
    println!("Accuracy: {:.2}%", acc);
    //plot_grad_mag(&net);
}
