//use linalg::Matrix;
//use rand::{RngExt, SeedableRng, rngs::StdRng};
//use rand_distr::{Distribution, Normal};
//
//pub struct Reservoir {
//    pub weights: Matrix,        // (size, size) recurrent weights
//    pub eff_weights: Matrix,    // (size, size) recurrent weights
//    pub conns: Matrix,          // (size, size) topology mask, 0.0 or 1.0
//    pub input_weights: Matrix,  // (size, input_dim)
//    pub bias: Matrix,           // (size,)
//    pub state: Matrix,          // (size,)
//    pub pre_activation: Matrix, // (size,) scratch buffer
//    pub activated: Matrix,      // (size,) scratch buffer
//    pub leak_rate: f32,
//    pub size: usize,
//    pub input_dim: usize,
//    seed: u64,
//}
//
//impl Reservoir {
//    pub fn new(size: usize, input_dim: usize) -> Self {
//        Reservoir {
//            weights: Matrix::zeros(size, size),
//            conns: Matrix::zeros(size, size),
//            input_weights: Matrix::zeros(size, input_dim),
//            bias: Matrix::zeros(size, 1),
//            state: Matrix::zeros(size, 1),
//            pre_activation: Matrix::zeros(size, 1),
//            activated: Matrix::zeros(size, 1),
//            leak_rate: 1.0,
//            size,
//            eff_weights: Matrix::zeros(size, size),
//            seed: 42,
//            input_dim,
//        }
//    }
//
//    pub fn random(
//        size: usize,
//        input_dim: usize,
//        density: f32,
//        weight_std: f32,
//        input_std: f32,
//        bias_std: f32,
//        leak_rate: f32,
//        seed: u64,
//    ) -> Self {
//        let mut r = StdRng::seed_from_u64(seed);
//
//        let w_dist = Normal::new(0.0, weight_std).unwrap();
//        let in_dist = Normal::new(0.0, input_std).unwrap();
//        let b_dist = Normal::new(0.0, bias_std).unwrap();
//
//        let weights = Matrix::from_shape_fn(size, size, |_| w_dist.sample(&mut r));
//        let conns = Matrix::from_shape_fn(size, size, |_| {
//            if r.random::<f32>() < density {
//                1.0
//            } else {
//                0.0
//            }
//        });
//        let input_weights = Matrix::from_shape_fn((size, input_dim), |_| in_dist.sample(&mut r));
//        let bias = Matrix::from_shape_fn(size, 1, |_| b_dist.sample(&mut r));
//        let state = Matrix::zeros(size, 1);
//        let eff_weights = &weights * &conns;
//
//        Reservoir {
//            weights,
//            seed,
//            conns,
//            input_weights,
//            bias,
//            state,
//            pre_activation: Matrix::zeros(size, 1),
//            activated: Matrix::zeros(size, 1),
//            eff_weights,
//            leak_rate,
//            size,
//            input_dim,
//        }
//    }
//
//    /// Effective recurrent weights = weights * conns (elementwise mask applied)
//    pub fn effective_weights(&self) -> Matrix {
//        &self.weights * &self.conns
//    }
//    pub fn refresh_effective_weights(&mut self) {
//        self.eff_weights = &self.weights * &self.conns;
//    }
//
//    /// Rescale effective weights to a target spectral radius.
//    /// Uses power iteration to estimate the largest eigenvalue magnitude.
//    pub fn scale_spectral_radius(&mut self, target: f32) {
//        let sr = self.estimate_spectral_radius(&self.eff_weights, 100);
//        if sr > 1e-12 {
//            let factor = target / sr;
//            self.weights.scale(factor);
//            self.refresh_effective_weights();
//        }
//    }
//
//    fn estimate_spectral_radius(&self, m: &Matrix, iters: usize) -> f32 {
//        let n = m.shape()[0];
//        let mut r = StdRng::seed_from_u64(self.seed);
//        let mut v = Matrix::from_shape_fn(n, |_| r.random::<f32>() - 0.5);
//        let mut norm = v.dot(&v).sqrt();
//        if norm < 1e-12 {
//            return 0.0;
//        }
//        v /= norm;
//
//        let mut eigenval = 0.0;
//        for _ in 0..iters {
//            let mv = m.dot(&v);
//            norm = mv.dot(&mv).pow(2.);
//            if norm < 1e-12 {
//                return 0.0;
//            }
//            v = &mv / norm;
//            eigenval = norm;
//        }
//        eigenval
//    }
//
//    /// One step: state = (1 - leak) * state + leak * tanh(W_eff·state + W_in·input + bias)
//    /// Fused in-place — zero heap allocations.
//    pub fn step(&mut self, input: &Matrix) {
//        self.pre_activation.data = self.bias.data;
//        self.pre_activation
//            .add_assign(&self.eff_weights.dot(&self.state));
//        self.pre_activation
//            .add_assign(&self.input_weights.dot(input));
//        let lr = self.leak_rate;
//        let omr = 1.0 - lr;
//        //Zip::from(&mut self.state)
//        //    .and(&self.pre_activation)
//        //    .for_each(|s, &p| {
//        //        *s = *s * omr + p.tanh() * lr;
//        //    });
//    }
//
//    pub fn reset_state(&mut self) {
//        self.state.fill(0.0);
//    }
//    //pub fn run_with_reset(&mut self, inputs: &Matrix, reset_every: usize) -> Matrix<f32> {
//    //    let n_timesteps = inputs.shape()[0];
//    //    let mut trajectory = Matrix::zeros((n_timesteps, self.size));
//
//    //    for t in 0..n_timesteps {
//    //        if t % reset_every == 0 {
//    //            self.reset_state();
//    //        }
//    //        self.step(&inputs.row(t));
//    //        trajectory.row_mut(t).assign(&self.state);
//    //    }
//
//    //    trajectory
//    //}
//    pub fn run_with_tokens(&mut self, tokens: &[u32], vocab_size: usize) -> Matrix {
//        let training_size = tokens.len();
//        //let n_timesteps = inputs.shape()[0];
//        let mut trajectory = Matrix::zeros((training_size, self.size));
//
//        for t in 0..training_size {
//            //let hot = one_hot_id(tokens[t], vocab_size);
//            let embed = embedding.lookup(tokens[t]);
//            self.step(&embed);
//            trajectory.row_mut(t).assign(&self.state);
//        }
//
//        trajectory
//    }
//
//    pub fn run(&mut self, inputs: &Matrix) -> Matrix<f32> {
//        let n_timesteps = inputs.shape()[0];
//        let mut trajectory = Matrix::zeros((n_timesteps, self.size));
//
//        for t in 0..n_timesteps {
//            self.step(&inputs.row(t));
//            trajectory.row_mut(t).assign(&self.state);
//        }
//
//        trajectory
//    }
//}
//
//pub fn confusion_matrix(predictions: &[u32], true_tokens: &[u32], vocab_size: usize) -> Matrix {
//    let mut cm = Matrix::zeros((vocab_size, vocab_size));
//    for (&pred, &truth) in predictions.iter().zip(true_tokens.iter()) {
//        cm[[truth as usize, pred as usize]] += 1;
//    }
//    cm
//}
//fn weight_magnitude_stats(w: &Matrix) {
//    let abs_vals: Vec<f32> = w.iter().map(|x| x.abs()).collect();
//    let max = abs_vals.iter().cloned().fold(0.0, f32::max);
//    let mean = abs_vals.iter().sum::<f32>() / abs_vals.len() as f32;
//    println!(
//        "weight |max|: {:.3}, |mean|: {:.4}, ratio: {:.1}",
//        max,
//        mean,
//        max / mean
//    );
//}
//pub fn diagnose_saturation(trajectory: &Matrix) {
//    let total = trajectory.len();
//    let saturated = trajectory.iter().filter(|&&v| v.abs() > 0.95).count();
//    let dead = trajectory.iter().filter(|&&v| v.abs() < 0.05).count();
//    println!(
//        "saturated: {:.1}%, dead: {:.1}%, healthy: {:.1}%",
//        100.0 * saturated as f32 / total as f32,
//        100.0 * dead as f32 / total as f32,
//        100.0 * (total - saturated - dead) as f32 / total as f32,
//    );
//}
//pub fn effective_rank(trajectory: &Matrix) -> f32 {
//    // center the data
//    let mean = trajectory.mean_axis(Axis(0)).unwrap();
//    let centered = trajectory - &mean;
//    let cov = centered.t().dot(&centered) / trajectory.nrows() as f32;
//
//    // approximate eigenvalues via the trace and squared-trace ratio
//    // (participation ratio - a cheap proxy for effective rank without full eigendecomposition)
//    let trace: f32 = cov.diag().sum();
//    let trace_sq: f32 = cov.iter().map(|x| x * x).sum();
//    (trace * trace) / trace_sq
//}
