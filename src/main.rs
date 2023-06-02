use embeddings::append_embeddings_to_files;
use embeddings::create_bert_embeddings;

use clap::{command, Parser};
use std::path::Path;

mod file_management;
mod embeddings;

use file_management::File;
use file_management::read_files_in_dir;
use embeddings::cosine_similarity;

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

fn get_files_matching_query_with_precision(query: String, files: Vec<File>, precision: f32) -> Vec<(File, f32)> {
    let query_embeddings = create_bert_embeddings(&[query.to_owned()]).unwrap();
    let query_embedding = &query_embeddings[0];

    let mut scores = Vec::new();
    for file in files {
        let score = cosine_similarity(query_embedding, &file.embedding);
        scores.push((file, score));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut results = Vec::new();
    for (file, score) in scores {
        if score > precision {
            results.push((file, score));
        }
    }

    results
}

fn search_directory(path: &Path, query: String, precision: f32, show: bool) {
    let files: Vec<File> = read_files_in_dir(path);
    let files = append_embeddings_to_files(files);
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
    fn test_embedding_match_with_mocked_files() {
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
