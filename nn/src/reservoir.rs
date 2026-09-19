use linalg::Matrix;

pub struct Reservoir {
    w_in: Matrix,
    w_res: Matrix,
    state: Matrix,
}

impl Reservoir {
    pub fn new(size: usize, in_dim: usize) -> Self {
        Self {
            w_in: Matrix::random(in_dim, size),
            w_res: Matrix::random(size, size),
            state: Matrix::zeros(size, size),
        }
    }
}
