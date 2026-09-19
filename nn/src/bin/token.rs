use std::{
    collections::{HashMap, HashSet},
    fs,
};

use linalg::Matrix;
use nn::{
    dataset::Dataset,
    net::{Activation, Network, softmax},
    utils::{one_hot, un_hot},
};

fn main() {
    let corpus = fs::read_to_string("data/seuss.txt").unwrap();

    let dataset = Dataset::new(corpus);
    let layers = vec![dataset.vocab_size, 10, 10, dataset.vocab_size];

    let mut net = Network::new(layers, dataset.vocab_size, Activation::Sigmoid);
    //println!("{:?}", net.weights[0].data);

    train(&mut net, dataset, 10);
}

fn train(net: &mut Network, dataset: Dataset, epochs: usize) {
    let lr = 0.01;

    let training_data = dataset.parse_data();

    let (x_train, y_train): (Vec<Matrix>, Vec<Matrix>) = training_data.into_iter().unzip();

    for epoch in 0..epochs {
        for sample in 0..x_train.len() {
            let x = &x_train[sample];
            let y = &y_train[sample];

            let out = net.forward(&x);

            let loss = net.loss(&y.t());
            if epoch % 2 == 0 && sample == 1 {
                println!("Epoch {} | Loss: {}", epoch, loss);
            }

            let (weight_grads, bias_grads) = net.backward(&x, &y);
            //{
            //    let l = net.layers.len() - 1;
            //    net.weights[l] = net.weights[l].subtract(&weight_grads[l].scale(lr));
            //    net.biases[l] = net.biases[l].subtract(&bias_grads[l].scale(lr));
            //}
            //for l in (net.layers.len() - 1)..net.layers.len() {
            for l in 0..net.layers.len() {
                //net.weights[l] = net.weights[l].subtract(&weight_grads[l].scale(lr));
                //net.biases[l] = net.biases[l].subtract(&bias_grads[l].scale(lr));
                net.weights[l].sub_assign_scaled(&weight_grads[l], lr);
                net.biases[l].sub_assign_scaled(&bias_grads[l], lr);
            }
        }
    }

    let mut output = String::from("The cat sat");

    for sample in 0..100 {
        let x = dataset.one_hot(
            &output.split_whitespace().collect::<Vec<&str>>()
                [output.split_whitespace().count() - 1]
                .to_string(),
        );

        let logits = net.forward(&x);
        let max_index = logits
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();

        let probs = softmax(&logits);
        println!("{:?}", probs);

        let token = dataset.idx_to_token.get(&max_index).unwrap();
        output.push_str(token);

        //println!(
        //    "Raw: [{}, {}, {}]\nTrue: [{:?}]\nOutput: {}\n",
        //    out.data[0], out.data[1], out.data[2], y.data, SPECIES_LIST[max_index]
        //);
    }
    println!("Output:\n'{}'", output);
}
