use rand::seq::SliceRandom;
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize, Clone)]
pub struct IrisRecord {
    pub sepal_length: f32,
    pub sepal_width: f32,
    pub petal_length: f32,
    pub species: String,
    pub petal_width: f32,
}

pub fn read_iris_data() -> Result<(Vec<IrisRecord>, Vec<IrisRecord>), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path("data/iris/IRIS.csv")?;

    let mut data: Vec<IrisRecord> = Vec::new();

    for result in rdr.deserialize() {
        let record: IrisRecord = result?;
        println!("{:?}", record);
        data.push(record);
    }

    data.shuffle(&mut rand::rng());
    let split = (0.3 * data.len() as f32) as usize;

    let train_data = &data[..split];
    println!("Training samples: {}", train_data.len());
    let test_data = &data[split..];

    Ok((train_data.to_vec(), test_data.to_vec()))
}
