use rand::RngExt;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from(data: &[&[f32]]) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        Self {
            rows,
            cols,
            data: data.into_iter().copied().flatten().copied().collect(),
        }
    }
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();

        Self {
            rows,
            cols,
            data: (0..rows * cols).map(|_| rng.random::<f32>()).collect(),
        }
    }

    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.rows;
        let end = start + self.rows;
        let s = &self.data[start..end];
        s
    }
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert!(self.rows == other.rows && self.cols == other.cols);

        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = i * self.cols + j;
                result.data[index] = self.data[index] + other.data[index];
            }
        }
        result
    }
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert!(self.rows == other.rows && self.cols == other.cols);

        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = i * self.cols + j;
                result.data[index] = self.data[index] - other.data[index];
            }
        }
        result
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn pretty_shape(&self) -> String {
        format!("[{} x {}]", self.rows, self.cols)
    }

    pub fn t(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = j * self.rows + i;
                result.data[index] = self.data[i * self.cols + j];
            }
        }
        result
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert!(
            self.cols == other.rows,
            "Invalid dimensions for dot product. [{} x {}] and [{} x {}]",
            self.rows,
            self.cols,
            other.rows,
            other.cols
        );
        let mut result = Matrix::new(self.rows, other.cols);
        // Each col in matrix B
        for j in 0..other.cols {
            // Each row in matrix A
            for i in 0..self.rows {
                let mut sum = 0.0;

                // Each row in matrix B
                for k in 0..other.rows {
                    let q1 = i * self.cols + k;
                    let q2 = k * other.cols + j;
                    sum += self.data[q1] * other.data[q2];
                }
                result.data[i * other.cols + j] = sum;
            }
        }
        result
    }
}

mod tests {
    use super::*;
    #[test]
    fn test_dot1() {
        let mut a = Matrix::new(2, 2);
        a.data = vec![3., 4., 7., 2.];
        let mut b = Matrix::new(2, 1);
        b.data = vec![6., 2.];

        let c = a.dot(&b);

        println!("{:?}", c.data);

        let res = [26., 46.];
        assert_eq!(c.data, res);
    }
    #[test]
    fn test_dot2() {
        let mut a = Matrix::new(2, 2);
        a.data = vec![3., 4., 7., 2.];
        let mut b = Matrix::new(2, 2);
        b.data = vec![6., 2., 1., 2.];

        let c = a.dot(&b);

        println!("{:?}", c.data);

        let res = [22., 14., 44., 18.];
        assert_eq!(c.data, res);
    }

    #[test]
    fn test_add() {
        let mut a = Matrix::new(2, 2);
        a.data = vec![3., 4., 7., 2.];
        let mut b = Matrix::new(2, 2);
        b.data = vec![6., 2., 1., 2.];

        let c = a.add(&b);

        println!("{:?}", c.data);

        let res = [9., 6., 8., 4.];
        assert_eq!(c.data, res);
    }
    #[test]
    fn test_subtract() {
        let mut a = Matrix::new(2, 2);
        a.data = vec![3., 4., 7., 2.];
        let mut b = Matrix::new(2, 2);
        b.data = vec![6., 2., 1., 2.];

        let c = a.subtract(&b);

        println!("{:?}", c.data);

        let res = [-3., 2., 6., 0.];
        assert_eq!(c.data, res);
    }
}

#[macro_export]
macro_rules! matrix {
    () => {};
}
