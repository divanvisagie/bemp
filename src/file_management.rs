use std::{path::Path, fs, io::Read};


#[derive(Debug, Clone)]
pub struct File {
    pub path: String,
    pub content: String,
    pub embedding: Vec<f32>
}

pub fn read_files_in_dir(directory: &Path) -> Vec<File> {
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
                    match Read::read_to_string(&mut file, &mut text) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_files_in_dir() {
        let files = read_files_in_dir(Path::new("test_files"));
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "test_files/example.rs");
        assert_eq!(files[1].path, "test_files/test1.js");
    }
}
