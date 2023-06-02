use std::process::Command;
use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // Run the linker command to enable static linking
    Command::new("gcc")
        .args(&["-c", "src/main.rs", "-o"])
        .arg(&format!("{}/main.o", out_dir))
        .status()
        .unwrap();

    Command::new("gcc")
        .args(&[&format!("{}/main.o", out_dir), "-o"])
        .arg(&format!("{}/app", out_dir))
        .args(&["-static", "-l", "stdc++", "-l", "m"])
        .status()
        .unwrap();
}
