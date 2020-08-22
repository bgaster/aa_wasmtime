# About

### [Project homepage](https://muses-dmi.github.io/projects/)

## Introduction

Rust implementation of the Audio Anywhere Module API, providing support for
loading and linking of Audio Anywhere modules. An interface is provided to load 
WASM modules from binaries, and exposing an interface to the Audio Anywhere WASM 
API.

For an example of the interface in practice see the [Audio Anywhere standalone application](https://github.com/bgaster/aa_standalone).

### Limitations

Currently we only support Wasmtime, as Wasmer had some issues with 128-SIMD. At
somepoint this needs to be generalized and a trait created that enables multiple implementations, or a feature based selection. 

## Including

Add the following line to your Cargo.toml:

```toml
aa_wasmtime = { git = "https://github.com/bgaster/aa_wasmtime" }
```

## Audio Anywhere WASM API

### Parameters

The following functions determine how many of each argument there are:

```rust
/// number of float params
#[no_mangle]
pub fn get_num_params_float() -> u32;

/// number of int params
#[no_mangle]
pub fn get_num_params_int() -> u32;

/// number of bool params
#[no_mangle]
pub fn get_num_params_bool() -> u32;
```

While the following allow setting and getting. Each parameter type, e.g. ```float```, is in its own address space, thus indexed in the range  $\interval[open right]{0}{N}$, where N is the number of parameters in respective address space. 

```rust
/// set float parameter 
/// panics if param not defined
#[no_mangle]
pub fn set_param_float(index: u32, v: f32) -> ()

/// set int parameter 
/// panics if param not defined
#[no_mangle]
pub fn set_param_int(index: u32, v: i32) -> ()

/// set bool parameter 
/// panics if param not defined
#[no_mangle]
pub fn set_param_bool(index: u32, v: bool) -> ()

// float parameter at index
// panics if param not defined
#[no_mangle]
pub fn get_param_float(index: u32) -> f32

// int parameter at index
// panics if param not defined
#[no_mangle]
pub fn get_param_int(index: u32) -> i32

// bool parameter at index
// panics if param not defined
#[no_mangle]
pub fn get_param_bool(index: u32) -> bool

#[no_mangle]
pub fn get_param_index(length: i32) -> i32
```

### Midi

No direct support is provided for Midi in the API, however, there is support for noteOn, noteOff, pitchBend, and so on. It would be good to add MPE at some point, but currently it is not supported.

```rust
#[no_mangle]
pub fn handle_note_on(mn: i32, vel: f32) 

#[no_mangle]
pub fn handle_note_off(mn: i32, vel: f32)
```

### Configuration

```rust
// initialize plugin, call to reset
#[no_mangle]
pub fn init(sample_rate: f64) -> ()

// sample rate
#[no_mangle]
pub fn get_sample_rate() -> f64

// number of input channels
#[no_mangle]
pub fn get_num_input_channels() -> u32

// number of output channels
#[no_mangle]
pub fn get_num_output_channels() -> u32

// number of voices (can be 0)
#[no_mangle]
pub fn get_voices() -> i32
```

### Buffer management

```rust
// get offset of given input within linear memory
#[no_mangle]
pub fn get_input(index: u32) -> u32

// get offset of given output within linear memory
#[no_mangle]
pub fn get_output(index: u32) -> u32

// set offset of given input within linear memory
#[no_mangle]
pub fn set_input(index: u32, offset: u32)

// set offset of given output within linear memory
#[no_mangle]
pub fn set_output(index: u32, offset: u32)
```

### Audio

```rust
#[no_mangle]
pub fn compute(frames: u32) -> ()
```

# License
© 2020 [Benedict R. Gaster (cuberoo_)](https://bgaster.github.io/)

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
