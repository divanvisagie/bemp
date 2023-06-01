# bemp
Bert Embedded Meaning Print

Search through the files on your drive based on their meaning, basically a semantic grep

```sh
Usage: bemp [OPTIONS]

Options:
  -q, --query <QUERY>              What you want to match against
  -p, --path <PATH>                Perform search in folder
  -s, --sensitivity <SENSITIVITY>
  -h, --help                       Print help
  -V, --version                    Print version
```

## Installation

### MacOS

Get libtorch
```
cd ~/.config/bemp
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.13.1.zip

```

Export the Environment variables
```sh
export LIBTORCH=~/.config/bemp/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH
```

Install
```sh
cargo install --path .
```
