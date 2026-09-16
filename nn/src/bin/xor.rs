use linalg::Matrix;
use nn::net::{Activation, Network};

fn main() {
    let layers = vec![3, 5, 2];

    let mut net = Network::new(layers, 2, Activation::Sigmoid);
    println!("{:?}", net.weights[0].data);

    //let x = Matrix::from(&[&[1.0, 0.0]]);
    //let true_y = Matrix::from(&[&[0.0, 0.0]]);

    train(&mut net, 10000);
}

fn train(net: &mut Network, epochs: usize) {
    let lr = 0.1;
    let x_train = vec![
        Matrix::from(&[&[1.0, 0.0]]),
        Matrix::from(&[&[0.0, 1.0]]),
        Matrix::from(&[&[1.0, 1.0]]),
        //Matrix::from(&[&[0.0, 0.0]]),
    ];
    let y_train = vec![
        Matrix::from(&[&[0.0, 1.0]]),
        Matrix::from(&[&[1.0, 0.0]]),
        Matrix::from(&[&[0.0, 0.0]]),
        //Matrix::from(&[&[1.0, 1.0]]),
    ];

    for epoch in 0..epochs {
        for sample in 0..x_train.len() {
            let x = &x_train[sample];
            let y = &y_train[sample];

            let out = net.forward(x.t());

            let loss = net.loss(&y);
            println!("Epoch {} | Loss: {}", epoch, loss);

            let (weight_grads, bias_grads) = net.backward(&x.t(), &y.t());
            for l in 0..net.layers.len() {
                net.weights[l] = net.weights[l].subtract(&weight_grads[l].scale(lr));
                net.biases[l] = net.biases[l].subtract(&bias_grads[l].scale(lr));
            }
        }
    }

    let x = Matrix::from(&[&[0.0, 0.0]]);
    let out = net.forward(x.t());
    println!("Output: {:.4} {:.4}", out.data[0], out.data[1]);
}
