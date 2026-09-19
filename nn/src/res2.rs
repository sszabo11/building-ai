use linalg::Matrix;

pub struct Reservoir {
    pub weights: Matrix,
    pub eff_weights: Matrix,
    pub conns: Matrix,
    pub input_weights: Matrix,
    pub bias: Matrix,
    pub state: Matrix,
    pub pre_activation: Matrix,
    pub activated: Matrix,
    pub leak_rate: f32,
    pub size: usize,
    pub input_dim: usize,
    seed: u64,
}

impl Reservoir {
    pub fn new(
        size: usize,
        input_dim: usize,
        spectral_radius: f32,
        connectivity: f32, // fraction of entries in `conns` that are 1.0, e.g. 0.1 = 10% sparse
        leak_rate: f32,
        seed: u64,
    ) -> Self {
        let mut rng_state = seed;

        let weights = Self::random_matrix(size, size, &mut rng_state);
        let conns = Self::random_mask(size, size, connectivity, &mut rng_state);
        let input_weights = Self::random_matrix(size, input_dim, &mut rng_state);
        let bias = Self::random_matrix(size, 1, &mut rng_state);

        let masked = weights.mul(&conns);
        let eff_weights = Self::scale_to_spectral_radius(&masked, spectral_radius);

        Reservoir {
            weights,
            eff_weights,
            conns,
            input_weights,
            bias,
            state: Matrix::zeros(size, 1),
            pre_activation: Matrix::zeros(size, 1),
            activated: Matrix::zeros(size, 1),
            leak_rate,
            size,
            input_dim,
            seed,
        }
    }

    /// One time step: reads self.state, writes a new self.state, returns a clone of it.
    pub fn step(&mut self, x: &Matrix) -> Matrix {
        assert!(
            x.rows == self.input_dim && x.cols == 1,
            "input shape mismatch"
        );

        // pre_activation = W_in . x + W_eff . state + bias
        let input_term = self.input_weights.dot(x);
        let recurrent_term = self.eff_weights.dot(&self.state);
        self.pre_activation = input_term.add(&recurrent_term).add(&self.bias);

        // activated = tanh(pre_activation)
        for i in 0..self.pre_activation.data.len() {
            self.activated.data[i] = self.pre_activation.data[i].tanh();
        }

        // leaky update: state = (1 - leak) * state + leak * activated
        for i in 0..self.state.data.len() {
            self.state.data[i] = (1.0 - self.leak_rate) * self.state.data[i]
                + self.leak_rate * self.activated.data[i];
        }

        self.state.clone()
    }

    pub fn reset(&mut self) {
        self.state = Matrix::zeros(self.size, 1);
    }

    // --- construction helpers ---

    fn random_matrix(rows: usize, cols: usize, rng_state: &mut u64) -> Matrix {
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(Self::next_uniform(rng_state) * 2.0 - 1.0); // [-1, 1]
        }
        Matrix { rows, cols, data }
    }

    fn random_mask(rows: usize, cols: usize, density: f32, rng_state: &mut u64) -> Matrix {
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(if Self::next_uniform(rng_state) < density {
                1.0
            } else {
                0.0
            });
        }
        Matrix { rows, cols, data }
    }

    /// xorshift64 — deterministic, seedable, dependency-free PRNG.
    /// Swap for the `rand` crate if you'd rather not hand-roll this.
    fn next_uniform(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state as f64 / u64::MAX as f64) as f32
    }

    /// Power iteration: estimates the largest-magnitude eigenvalue of `m`,
    /// then rescales `m` so that eigenvalue equals `target_radius`.
    fn scale_to_spectral_radius(m: &Matrix, target_radius: f32) -> Matrix {
        assert!(
            m.rows == m.cols,
            "spectral radius only defined for square matrices"
        );
        let n = m.rows;

        let mut v = Matrix::zeros(n, 1);
        for i in 0..n {
            v.data[i] = 1.0 + (i as f32) * 0.01; // arbitrary non-degenerate starting vector
        }

        let mut eigenvalue_estimate = 0.0;
        for _ in 0..100 {
            let mv = m.dot(&v);
            let norm = mv.data.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-12 {
                break; // m is (numerically) the zero matrix
            }
            eigenvalue_estimate = norm;
            v = mv.scale(1.0 / norm);
        }

        if eigenvalue_estimate < 1e-12 {
            return m.clone(); // degenerate case, nothing sensible to scale by
        }

        m.scale(target_radius / eigenvalue_estimate)
    }
}
