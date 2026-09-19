use std::{
    collections::{HashMap, HashSet},
    fs,
};

use linalg::Matrix;
use nn::{
    dataset::{Dataset, TokenType},
    net::{Activation, Network, softmax},
    res2::Reservoir,
    utils::{one_hot, un_hot},
};
use rand::{Rng, RngExt};
use rand_distr::Normal;

fn main() {
    let corpus = fs::read_to_string("../data/cat.txt").unwrap();

    let dataset = Dataset::new(corpus, TokenType::Word);

    //println!("Vocab size: {}", dataset.vocab_size);
    let lr = 0.99;
    let sr = 1.7;
    let input_dim = dataset.vocab_size;
    let conns = 0.8;
    let size = 100;
    let mut reservoir = Reservoir::new(size, input_dim, sr, conns, lr, 42);
    //let s1 = reservoir.step(&dataset.one_hot("c"));
    //reservoir.reset();
    //let s2 = reservoir.step(&dataset.one_hot("s"));
    //println!(
    //    "diff: {}",
    //    s1.subtract(&s2)
    //        .data
    //        .iter()
    //        .map(|x| x * x)
    //        .sum::<f32>()
    //        .sqrt()
    //);
    reservoir.reset();

    let hidden = 100;
    let layers = vec![size, hidden, dataset.vocab_size];
    let mut net = Network::new(layers, size, Activation::Sigmoid);
    //println!("{:?}", net.weights[0].data);

    let epochs = 50;
    println!("[TEST]");
    println!("Leak rate: {}", lr);
    println!("Spectral: {}", sr);
    println!("Conns: {}", conns);
    println!("Reservoir size: {}", size);
    println!("Hidden size: {}", hidden);
    println!("Epochs: {}", epochs);
    println!();
    train(&mut net, &mut reservoir, dataset, epochs);
}

fn train(net: &mut Network, reservoir: &mut Reservoir, dataset: Dataset, epochs: usize) {
    let lr = 0.1;

    let training_data = dataset.parse_data();

    let (x_train, y_train): (Vec<Matrix>, Vec<Matrix>) = training_data.into_iter().unzip();

    for epoch in 0..epochs {
        reservoir.reset();
        for sample in 0..x_train.len() {
            let x = &x_train[sample];
            let y = &y_train[sample];
            let state = reservoir.step(x);

            let out = net.forward(&state);

            let loss = net.loss(&y.t());
            if epoch % 5 == 0 && sample == 1 {
                println!("Epoch {} | Loss: {}", epoch, loss);
            }

            let (weight_grads, bias_grads) = net.backward(&state, &y);
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

    //let mut output = String::from("The cat sat");
    let seed = "The Cat in the Hat

By Dr. Seuss

The sun did not shine.";
    let mut tokens: Vec<String> = seed.split_whitespace().map(|c| c.to_string()).collect();

    reservoir.reset();
    for token in tokens.iter() {
        let x = dataset.one_hot(&token);
        reservoir.step(&x);
    }

    for sample in 0..100 {
        //println!("W: {:?}", tokens);
        let last_w = tokens[tokens.len() - 1].to_string();

        let x = dataset.one_hot(&last_w);

        let state = reservoir.step(&x);
        let logits = net.forward(&state);
        let max_index = logits
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();

        let probs = softmax(&logits);
        let mut rng = rand::rng();
        let chosen_index = sample_from(&probs, &mut rng);
        //println!("{:?}", probs);

        //println!("top idx: {}", max_index);
        //println!("chosen idx: {}", chosen_index);
        //println!("{:?}", dataset.idx_to_token.keys());
        let token = dataset.idx_to_token.get(&max_index).unwrap();
        tokens.push(token.to_string());

        //println!(
        //    "Raw: [{}, {}, {}]\nTrue: [{:?}]\nOutput: {}\n",
        //    out.data[0], out.data[1], out.data[2], y.data, SPECIES_LIST[max_index]
        //);
    }
    println!("Output:\n'{}'", tokens.join(""));
}

fn sample_from(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}
