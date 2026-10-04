use image::{GenericImageView, ImageReader};
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

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct LabelRecord {
    #[serde(rename = "Image Index")]
    pub image_index: usize,
    #[serde(rename = "Image Path")]
    pub image_path: String,
    #[serde(rename = "Label")]
    pub label: String,
}

type Px = (f32, f32, f32, f32);
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct ImageRecord {
    pub pxs: Vec<Px>,
    pub image_index: usize,
    pub image_path: String,
    pub label: Vec<f32>,
}
fn remove_first_and_last(value: &str) -> &str {
    let mut chars = value.chars();
    chars.next(); // Removes the first character
    chars.next_back(); // Removes the last character
    chars.as_str() // Returns the remaining slice
}

fn parse_label(label: &str) -> Vec<f32> {
    let lab = remove_first_and_last(label);

    let l: Vec<&str> = lab.split(".").collect();

    let v = (0..5)
        .into_iter()
        .map(|i| {
            l[i].trim()
                .parse::<f32>()
                .expect("Failed to parse label int")
        })
        .collect();

    v
}
pub fn read_img_labels() -> Result<Vec<ImageRecord>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path("/home/rabbit/Downloads/archive(11)/labels.csv")?;

    let images_dir = "/home/rabbit/Downloads/archive(11)/images/images";

    let mut data: Vec<ImageRecord> = Vec::new();

    for result in rdr.deserialize() {
        let label: LabelRecord = result?;

        let path = format!("{}/image_{}.JPEG", images_dir, label.image_index);
        let img = ImageReader::open(path)?.decode()?;

        let lab = parse_label(&label.label);
        let pxs: Vec<(f32, f32, f32, f32)> = img
            .pixels()
            .map(|(_x, _y, color)| {
                (
                    color[0] as f32 / 255.,
                    color[1] as f32 / 255.,
                    color[2] as f32 / 255.,
                    color[3] as f32 / 255.,
                )
            })
            .collect();

        let record = ImageRecord {
            pxs,
            image_index: label.image_index,
            image_path: label.image_path,
            label: lab,
        };

        //println!("{:?}", record);
        data.push(record);
    }

    data.shuffle(&mut rand::rng());
    let data = &data[..20];
    //let split = (0.3 * data.len() as f32) as usize;

    println!("Training samples: {}", data.len());

    Ok(data.to_vec())
}
