use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

use clap::{command, Parser};
use rust_bert::RustBertError;
use std::fs;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone)]
struct File {
    path: String,
    content: String,
    embedding: Vec<f32>
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// What you want to match against
    #[arg(short, long)]
    query: Option<String>,

    /// Perform search in folder
    #[arg(short, long)]
    path: Option<String>,

    #[arg(short, long)]
    threshold: Option<f32>,

    #[arg(short, long)]
    show_score: bool
}

fn read_files_in_dir(directory: &Path) -> Vec<File> {
    let mut texts = Vec::new();
    if directory.is_dir() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            //ignore dotfiles
            if path.to_str().unwrap().starts_with(".") {
                continue;
            }

            if path.is_dir() {
                let mut sub_texts = read_files_in_dir(&path);
                texts.append(&mut sub_texts);
                continue;
            }

            match fs::File::open(&path) {
                Ok(mut file) => {
                    let mut text = String::new();
                    match file.read_to_string(&mut text) {
                        Ok(_) => {
                            texts.push(File {
                                path: path.to_str().unwrap().to_string(),
                                content: text,
                                embedding: Vec::new()
                            });
                        },
                        Err(e) => {
                            println!("Error reading file {:?} {:?}", path, e);
                            continue;
                        }
                    }
                },
                Err(e) => {
                    println!("Error processing files {:?}", e)
                }
            };
        }
    }
    texts
}


fn create_bert_embeddings(texts: &[String]) -> Result<Vec<Vec<f32>>, RustBertError> {
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}

fn append_embeddings_to_files(files: Vec<File>) -> Vec<File> {
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


fn get_files_with_embeddings_in_dir(path: &Path) -> Vec<File> {
    let files: Vec<File> = read_files_in_dir(path);
    append_embeddings_to_files(files)
}

fn get_files_matching_query_with_precision(query: String, files: Vec<File>, precision: f32) -> Vec<(File, f32)> {
    let query_embeddings = create_bert_embeddings(&[query.to_owned()]).unwrap();
    let query_embedding = &query_embeddings[0];

    let mut scores = Vec::new();
    for file in files {
        let score = cosine_similarity(query_embedding, &file.embedding);
        scores.push((file, score));
    }

    // scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut results = Vec::new();
    for (file, score) in scores {
        if score > precision {
            results.push((file, score));
        }
    }

    results
}

fn search_directory(path: &Path, query: String, precision: f32, show: bool) {
    let files = get_files_with_embeddings_in_dir(path);
    let scored = get_files_matching_query_with_precision(query, files, precision);
    for (file, score) in scored {
        if show {
            println!("{}\t{}", file.path, score);
        } else {
            println!("{}", file.path);
        }
    }
}

fn main() {
    let args = Args::parse();
    let mut query = String::new();
    let mut threshold = 0.5;

    if let Some(t) = args.threshold {
        threshold = t;
    }

    if let Some(q) = args.query {
        query = q;
    }

    if let Some(p) = args.path {
        let scan_path = Path::new(&p);
        search_directory(scan_path, query, threshold, args.show_score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedding() {
        //create some mock files
        let files = vec![
            File {
                path: "test1".to_string(),
                content: "console.log()".to_string(),
                embedding: Vec::new()
            },
            File {
                path: "test2".to_string(),
                content: "System.out.println()".to_string(),
                embedding: Vec::new()
            },
            File {
                path: "test3".to_string(),
                content: "println!(\"Hello{}\")".to_string(),
                embedding: Vec::new()
            },
        ];

        let query = "Java code".to_string();
        let files = append_embeddings_to_files(files);
        let files_with_scores = get_files_matching_query_with_precision(query, files, 0.5);


        let expected_results_count = 1;
        let actual: i32 = files_with_scores.len().try_into().unwrap();

        assert_eq!(expected_results_count, actual, "expected {} got {}", expected_results_count, actual);
        assert!(files_with_scores[0].0.content.contains("System.out.println()"), "Expected java code");
    }
}
