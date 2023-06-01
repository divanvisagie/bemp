# bemp
Bert Embedded Meaning Print

## Installation

```
cd ~/.config/bemp
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.13.1.zip
```

```sh
export LIBTORCH=~/.config/bemp/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH
```


### MacOS
```sh
brew install libtorch
cargo install --path .
```
