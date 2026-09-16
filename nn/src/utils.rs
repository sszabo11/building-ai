pub fn one_hot(options: &[&str], val: &str) -> Vec<f32> {
    let mut ops = options.to_owned();

    ops.sort();

    let mut v = vec![0.0; options.len()];
    for i in 0..ops.len() {
        if val == ops[i] {
            v[i] = 1.
        }
    }
    v
}

pub fn un_hot(options: &[&str], val: usize) -> Option<String> {
    let mut ops: Vec<String> = options.iter().map(|&s| s.to_string()).collect();

    ops.sort();

    ops.get(val).cloned()
}
