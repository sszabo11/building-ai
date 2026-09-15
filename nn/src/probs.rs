use std::collections::HashMap;
fn test1() {
    let numbers = vec![1, 2, 3, 4, 5, 6];

    let probs = calc_probbs(&numbers);

    println!("{:?}", probs);
    let multiplied_twice = mul_twice(&numbers);

    println!("{:?}", multiplied_twice)
}
fn test2() {
    let numbers = vec![2, 2, 1, 1, 2, 1];
    let probs = calc_probbs(&numbers);

    println!("{:?}", probs);
    let (multiplied_twice, m_len) = mul_twice(&numbers);
    println!("m: {:?}", multiplied_twice);
    let l = &hash_to_list(&multiplied_twice);
    println!("l: {:?}", l);
    let (added, _) = add_twice(l);
    println!("{:?}", added)
}

fn hash_to_list(map: &HashMap<u32, f32>) -> Vec<u32> {
    let len = map.len();
    let mut res = Vec::new();

    let mut freqs: HashMap<u32, f32> = HashMap::new();
    for (&n, &p) in map.iter() {
        let freq = p * len as f32;
        freqs.insert(n, freq);
    }
    let min_value = freqs.values().max_by(|a, b| b.total_cmp(a)).unwrap();

    for (&n, freq_f) in freqs.iter() {
        let freq = freq_f / min_value;
        println!("f: {}", freq);
        for _ in 0..(freq as u32) {
            res.push(n);
        }
    }
    res
}

fn add_twice(n: &[u32]) -> (HashMap<u32, f32>, usize) {
    let probs = calc_probbs(n);
    let mut new_probs: HashMap<u32, f32> = HashMap::new();

    let mut l = 0;
    for i in n.iter() {
        for j in n.iter() {
            let new_p = *probs.get(i).unwrap() + *probs.get(j).unwrap();
            let res = i * j;
            l += 1;
            if let Some(v) = new_probs.get(&res) {
                new_probs.insert(res, v + new_p);
            } else {
                new_probs.insert(res, new_p);
            };
        }
    }
    (new_probs, l)
}
fn mul_twice(n: &[u32]) -> (HashMap<u32, f32>, usize) {
    let probs = calc_probbs(n);
    let mut new_probs: HashMap<u32, f32> = HashMap::new();

    let mut l = 0;
    for i in n.iter() {
        for j in n.iter() {
            let new_p = *probs.get(i).unwrap() * *probs.get(j).unwrap();
            let res = i * j;
            l += 1;
            if let Some(v) = new_probs.get(&res) {
                new_probs.insert(res, v + new_p);
            } else {
                new_probs.insert(res, new_p);
            };
        }
    }
    (new_probs, l)
}

fn calc_probbs(n: &[u32]) -> HashMap<u32, f32> {
    let sample = n.len();
    let dups = get_dups(&n);
    let mut probs: HashMap<u32, f32> = HashMap::new();

    for x in n.iter() {
        let prob = *dups.get(x).unwrap() as f32 / sample as f32;
        probs.insert(*x, prob);
    }
    probs
}

fn get_dups(dis: &[u32]) -> HashMap<u32, u32> {
    let mut map: HashMap<u32, u32> = HashMap::new();

    for &x in dis.iter() {
        if let Some(v) = map.get(&x) {
            map.insert(x, v + 1);
        } else {
            map.insert(x, 1);
        };
    }
    map
}
