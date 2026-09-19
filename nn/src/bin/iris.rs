use std::str::LinesAny;

use linalg::Matrix;
use nn::{
    data::{IrisRecord, read_iris_data},
    net::{Activation, Network, softmax},
    utils::{one_hot, un_hot},
};

fn main() {
    let layers = vec![4, 30, 3];

    let mut net = Network::new(layers, 4, Activation::Sigmoid);
    println!("{:?}", net.weights[0].data);

    train(&mut net, 5000);
}

const SPECIES_LIST: &[&str] = &["iris-setosa", "iris-versicolor", "iris-virginica"];

fn parse_data(data: Vec<IrisRecord>) -> Vec<(Matrix, Matrix)> {
    let mut parsed: Vec<(Matrix, Matrix)> = Vec::new();

    for record in data.iter() {
        let mut m = Matrix::zeros(1, 4);
        // ORDER MATTErs.
        m.data = vec![
            record.petal_width,
            record.sepal_width,
            record.sepal_length,
            record.petal_length,
        ];

        let y_v = one_hot(SPECIES_LIST, &record.species.to_lowercase());

        let mut y = Matrix::zeros(1, 3);
        y.data = y_v;
        parsed.push((m, y));
    }

    parsed
}
fn train(net: &mut Network, epochs: usize) {
    let lr = 0.1;

    let (training_data, test_data) = read_iris_data().unwrap();
    let training_data = parse_data(training_data);
    let test_data = parse_data(test_data);

    let (x_train, y_train): (Vec<Matrix>, Vec<Matrix>) = training_data.into_iter().unzip();

    for epoch in 0..epochs {
        for sample in 0..x_train.len() {
            let x = &x_train[sample];
            let y = &y_train[sample];

            let out = net.forward(x.t());

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
    for sample in 0..test_data.len() {
        let (x, y) = &test_data[sample];

        let logits = net.forward(x.t());
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
    let acc = correct as f32 / test_data.len() as f32 * 100.;
    println!("Accuracy: {:.2}%", acc);
    //plot_grad_mag(&net);
}

fn plot_grad_mag(net: &Network) {
    let mut vals = vec![vec![0.0; net.grad_w[0].len()]; net.grad_w.len()];

    for i in 0..net.grad_w.len() {
        for j in 0..net.grad_w[i].len() {
            let f = net.grad_w[i][j].frobenius_norm();
            vals[i][j] = f;
        }
    }

    let d1: Vec<f32> = vals.iter().map(|x| x[0]).collect();
    let d2: Vec<f32> = vals.iter().map(|x| x[1]).collect();
    let d3: Vec<f32> = vals.iter().map(|x| x[2]).collect();
    simple_plot::plot!("title", d1, d2, d3);
}
