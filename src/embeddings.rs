use rust_bert::{RustBertError, pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType}};

use crate::file_management::File;


pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}

pub fn create_bert_embeddings(texts: &[String]) -> Result<Vec<Vec<f32>>, RustBertError> {
    let model = SentenceEmbeddingsBuilder::remote(
        SentenceEmbeddingsModelType::AllMiniLmL12V2
    )
    .create_model()?;

    let embeddings = model.encode(texts)?;

    let ebd = embeddings
        .into_iter()
        .collect();

    Ok(ebd)
}

pub fn append_embeddings_to_files(files: Vec<File>) -> Vec<File> {
    let embeddings = create_bert_embeddings(
        files.iter().map(|f| f.content.clone()).collect::<Vec<String>>().as_slice()
    ).unwrap();
    

    let mut files_with_embeddings: Vec<(File, Vec<f32>)> =
        files.into_iter().zip(embeddings).collect::<Vec<(File, Vec<f32>)>>();
    
    for (file, embedding) in &mut files_with_embeddings {
        file.embedding = embedding.clone();
    }
    
    //map to vector of files
    let files_with_embeddings = files_with_embeddings
        .into_iter()
        .map(|(file, _)| file)
        .collect::<Vec<File>>();

    files_with_embeddings
}
