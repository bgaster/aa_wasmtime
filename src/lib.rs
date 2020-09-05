//! 
//! Wasmtime implementation of AAUnit
//! Copyright: Benedict R. Gaster
#![allow(dead_code)]

use std::iter;
use wasmtime::*;

use anyhow::{anyhow, Result};

//------------------------------------------------------------------------------

const MODULE_PREFIX: &'static str = "instance";

//------------------------------------------------------------------------------

/// Representation of an Audio Anywhere Module or Unit
pub struct AAUnitIndividual {
    /// module init function
    init: Func,

    /// handle note on message
    handle_note_on: Func,
    /// handle note off message
    handle_note_off: Func,
    /// get number of voices (can be zero)
    get_voices: Func,
    /// get offset for particular input buffer
    get_input: Func,
    /// get offset for particular output buffer
    get_output: Func,
    /// set offset for particular input buffer
    set_input: Func,
    /// set offset for particular output buffer
    set_output: Func,
    /// get parameter index
    get_param_index : Func,
    /// get current sample rate from WASM
    get_sample_rate: Func,
    /// get number of inputs channels for current module
    get_num_inputs: Func,
    /// get number of outputs channels for current module
    get_num_outputs: Func,
    /// set float parameter in module
    set_param_float: Func,
    /// set int parameter in module
    set_param_int: Func,
    /// get float parameter in module
    get_param_float: Func,
    /// get int parameter in module
    get_param_int: Func,
    /// compute function for module module
    compute: Func,
    /// input buffer offsets
    input_offsets: Vec<usize>,
    /// output buffer offsets
    output_offsets: Vec<usize>,

    /// TODO: remove once Wasmtime supports shared memories!
    memory: Memory,

    /// wasmtime instance for single module
    #[allow(dead_code)]
    instance: Instance,
}

/// Representation of an Audio Anywhere Module or Unit
pub struct AAUnit {
    
    // /// wasmtime memory segment
    // memory: Memory,

    aaunits: Vec<AAUnitIndividual>,
}

impl AAUnit {
    /// create an Audio Anywhere module
    pub fn new(wasm_bytes: Vec<Vec<u8>>) -> Result<Self> {
        // fail if there are no wasm sources to link
        if wasm_bytes.len() == 0 {
            return Err(anyhow!("No AA Wasm modules"))
        }

        let engine = Engine::new(Config::new().wasm_simd(true));
        let store = Store::new(&engine);

        // let mut linker = Linker::new(&store);
        // linker.allow_shadowing(true);
        // // add all but module 0 to the linker, ready for linking with module 0
        // for i in 1..wasm_bytes.len() {
        //     let module = Module::new(store.engine(), &wasm_bytes[i][..])?;
        //     let module_name = [MODULE_PREFIX, &i.to_string()].join("_");
        //     linker.module(&module_name, &module)?;
        // }
        
        // // now link with module 0
        // let module = Module::new(store.engine(), &wasm_bytes[0][..])?;
        // let instance = linker.instantiate(&module)?;
        // println!("printing names:");
        // for e in instance.exports() {
        //     println!("name: {}", e.name());
        // }

        let mut aaunits = Vec::new();
        // push start of graph
        //aaunits.push(Self::create_aaunit("", &instance)?);
        // now push sequenced nodes
        for i in 0..wasm_bytes.len() {
            let module = Module::new(store.engine(), &wasm_bytes[0][..])?;
            let instance = Instance::new(&store, &module, &[])?;
            // handle to Wasm linear memory
            let memory = instance
                        .get_memory("memory")
                        .ok_or(anyhow!("WASM memory failed"))?;

            aaunits.push(Self::create_aaunit("", instance, memory)?);
        }

        Ok(Self {
            aaunits,
        })
    }

    /// get offest for symbol in Wasm linear memory, if not defined returns 0
    /// of course, in theory 0 is a valid offset, so it is important to check this elsewhere
    fn get_global_symbol_offset(symbol: &str, instance: &Instance) -> usize {
        instance.get_global(symbol).map_or(0, |offset| {
            if let Val::I32(o) = offset.get() {
                o as usize
            }
            else {
                0
            }
        })
    }

    fn create_aaunit(prefix: &str, instance: Instance, memory: Memory) -> Result<AAUnitIndividual> {
        let init = instance
            .get_func(&[prefix, "init"].join(""))
            .ok_or(anyhow!("WASM lacked init function"))?;
        let handle_note_on = instance
            .get_func(&[prefix, "handle_note_on"].join(""))
            .ok_or(anyhow!("WASM lacked handle_note_on function"))?;
        let handle_note_off = instance
            .get_func(&[prefix, "handle_note_off"].join(""))
            .ok_or(anyhow!("WASM lacked handle_note_off function"))?;
        let get_input = instance
            .get_func(&[prefix, "get_input"].join(""))
            .ok_or(anyhow!("WASM lacked get_input function"))?;
        let get_output = instance
            .get_func(&[prefix,"get_output"].join(""))
            .ok_or(anyhow!("WASM lacked get_output function"))?;
        let set_input = instance
            .get_func(&[prefix,"set_input"].join(""))
            .ok_or(anyhow!("WASM lacked set_input function"))?;
        let set_output = instance
            .get_func(&[prefix,"set_output"].join(""))
            .ok_or(anyhow!("WASM lacked set_output function"))?;
        let get_voices = instance
            .get_func(&[prefix,"get_voices"].join(""))
            .ok_or(anyhow!("WASM lacked get_voices function"))?;
        let get_param_index = instance
            .get_func(&[prefix,"get_param_index"].join(""))
            .ok_or(anyhow!("WASM lacked get_param_index function"))?;
        let get_sample_rate: Func = instance
            .get_func(&[prefix,"get_sample_rate"].join(""))
            .ok_or(anyhow!("WASM lacked get_sample_rate function"))?;
        let get_num_inputs: Func = instance
            .get_func(&[prefix,"get_num_input_channels"].join(""))
            .ok_or(anyhow!("WASM lacked get_num_input_channels function"))?;
        let get_num_outputs: Func = instance
            .get_func(&[prefix,"get_num_output_channels"].join(""))
            .ok_or(anyhow!("WASM lacked get_num_output_channels function"))?;
        let set_param_float: Func = instance
            .get_func(&[prefix,"set_param_float"].join(""))
            .ok_or(anyhow!("WASM lacked set_param_float function"))?;
        let set_param_int: Func = instance
            .get_func(&[prefix,"set_param_int"].join(""))
            .ok_or(anyhow!("WASM lacked set_param_int function"))?;
        let get_param_float: Func = instance
            .get_func(&[prefix,"get_param_float"].join(""))
            .ok_or(anyhow!("WASM lacked get_param_float function"))?;
        let get_param_int: Func = instance
            .get_func(&[prefix,"get_param_int"].join(""))
            .ok_or(anyhow!("WASM lacked get_param_int function"))?;
        let compute = instance
            .get_func(&[prefix,"compute"].join(""))
            .ok_or(anyhow!("WASM lacked compute function"))?;

        Ok(AAUnitIndividual {
            init,
            handle_note_on,
            handle_note_off,
            get_voices,
            get_input,
            get_output,
            set_input,
            set_output,
            get_param_index,
            get_sample_rate,
            get_num_inputs,
            get_num_outputs,
            set_param_float,
            set_param_int,
            get_param_float,
            get_param_int,
            compute,
            input_offsets: Vec::new(),
            output_offsets: Vec::new(),
            memory,
            instance,
        })
    }

    /// initialize module
    /// must be called for WASM AA Module to be correclty initialized
    #[inline]
    pub fn init(&mut self, sample_rate: f64) -> Result<()> {
        // first initialize WASM module
        let f = self.aaunits[0].init.get1::<f64, ()>()?;
        f(sample_rate)?;

        // now setup buffers

        // determine number of inputs
        let number_inputs = self.aaunits[0].get_num_inputs.get0::<i32>()?()?;
        let number_outputs = self.aaunits[0].get_num_outputs.get0::<i32>()?()?;
        
        // configure inputs
        for i in 0..number_inputs {
            let v = self.aaunits[0].get_input.get1::<i32,i32>()?(i as i32)? as usize;
            self.aaunits[0].input_offsets.push(v);
        }
        
        // configure outputs
        for i in 0..number_outputs {
            let v = self.aaunits[0].get_output.get1::<i32,i32>()?(i as i32)? as usize;
            self.aaunits[0].output_offsets.push(v);
        }

        Ok(())
    }

    /// send note on message
    #[inline]
    pub fn handle_note_on(&self, note: i32, velocity: f32) -> Result<()> {
        let f = self.aaunits[0].handle_note_on.get2::<i32, f32, ()>()?;
        f(note, velocity).map_err(|_| anyhow!("WASM call handle_note_on failed"))
    }

    /// send note off message
    #[inline]
    pub fn handle_note_off(&self, note: i32, velocity: f32) -> Result<()> {
        let f = self.aaunits[0].handle_note_off.get2::<i32, f32, ()>()?;
        f(note, velocity).map_err(|_| anyhow!("WASM call handle_note_off failed"))
    }

    // get number of voices
    #[inline]
    pub fn get_voices(&self) -> Result<i32> {
        let f = self.aaunits[0].get_voices.get0::<i32>()?;
        f().map_err(|_| anyhow!("WASM call get_voices failed"))
    }

    #[inline]
    pub fn get_param_index(&self, _name: &str) -> Result<i32> {
        let f = self.aaunits[0].get_param_index.get1::<i32,i32>()?;
        f(0).map_err(|_| anyhow!("WASM call get_param_index failed"))
    }

    /// set a float parameter
    #[inline]
    pub fn set_param_float(&self, index: u32, param: f32) -> Result<()> {
        let f = self.aaunits[0].set_param_float.get2::<u32, f32, ()>()?;
        f(index, param).map_err(|_| anyhow!("WASM call set_param_float failed"))
    }

    /// set an int parameter
    #[inline]
    pub fn set_param_int(&self, index: u32, param: i32) -> Result<()> {
        let f = self.aaunits[0].set_param_int.get2::<u32, i32, ()>()?;
        f(index, param).map_err(|_| anyhow!("WASM call set_param_int failed"))
    }

    /// get a float parameter
    #[inline]
    pub fn get_param_float(&self, index: u32) -> Result<f32> {
        let f = self.aaunits[0].get_param_float.get1::<u32, f32>()?;
        f(index).map_err(|_| anyhow!("WASM call get_param_float failed"))
    }

    /// get an int parameter
    #[inline]
    pub fn get_param_int(&self, index: u32) -> Result<i32> {
        let f = self.aaunits[0].get_param_int.get1::<u32, i32>()?;
        f(index).map_err(|_| anyhow!("WASM call get_param_int failed"))
    }

    /// get number of audio input buffers
    #[inline]
    pub fn get_number_inputs(&self) -> Result<i32> {
        let f = self.aaunits[0].get_num_inputs.get0::<i32>()?;
        f().map_err(|_| anyhow!("WASM call get_number_inputs failed"))
    }

    /// get number of audio output buffers
    #[inline]
    pub fn get_number_outputs(&self) -> Result<i32> {
        let f = self.aaunits[0].get_num_outputs.get0::<i32>()?;
        f().map_err(|_| anyhow!("WASM call get_number_outputs failed"))
    }

    /// compute audio for 1 input and 1 output channels
    #[inline]
    pub fn compute_one_one(&self, frames: usize, inputs: &[f32], outputs: &mut [f32]) -> Result<()> {
        // setup and copy input audio
        let inputs0 = inputs[0..frames as usize].iter();
        let wasm_inputs0: &mut [f32] = unsafe { 
            let bytes = 
                &mut self.aaunits[0].memory.data_unchecked_mut()[self.aaunits[0].input_offsets[0]..self.aaunits[0].input_offsets[0] 
                                                      + (frames*std::mem::size_of::<f32>())];
            std::mem::transmute(bytes)
        };

        let zipped_inputs = inputs0.zip(wasm_inputs0);
        for (input, wasm_input) in zipped_inputs {
            *wasm_input = *input;
        }

        // now call compute
        let compute = self.aaunits[0].compute.get1::<u32, ()>()?;
        compute(frames as u32)?;

        // setup and copy audio out of WASM
        let outputs0 = outputs[0..frames as usize].iter_mut();
        let wasm_outputs0: &[f32] = unsafe { 
            let bytes = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[0]..self.aaunits[0].output_offsets[0] + 
                                              (frames*std::mem::size_of::<f32>())];
            std::mem::transmute(bytes)
        };

        let zipper_outputs = outputs0.zip(wasm_outputs0);
        for (output, wasm_output) in zipper_outputs {
            *output = *wasm_output;
        }

        Ok(())
    }

    /// compute audio for 1 input and 2 outputs channels
    /// assume that output channels are interlaced
    #[inline]
    pub fn compute_one_two(&self, frames: usize, inputs: &[f32], outputs: &mut [f32]) -> Result<()> {
        // setup and copy input audio
        let inputs0 = inputs[0..frames as usize].iter();
        let wasm_inputs0: &mut [f32] = unsafe { 
            let bytes = 
                &mut self.aaunits[0].memory.data_unchecked_mut()[self.aaunits[0].input_offsets[0]..self.aaunits[0].input_offsets[0] 
                                                      + (frames*std::mem::size_of::<f32>())];
            std::mem::transmute(bytes)
        };

        let zipped_inputs = inputs0.zip(wasm_inputs0);
        for (input, wasm_input) in zipped_inputs {
            *wasm_input = *input;
        }

        // now call compute
        let compute = self.aaunits[0].compute.get1::<u32, ()>()?;
        compute(frames as u32)?;

        // setup and copy audio out of WASM
        // output is assumed to be interlaced
        let outputs0 = outputs[0..2 * frames as usize].iter_mut();
        let (wasm_outputs0, wasm_outputs1): (&[f32],&[f32]) = unsafe { 
            let bytes0 = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[0]..self.aaunits[0].output_offsets[0] + 
                                              (frames*std::mem::size_of::<f32>())];
            let bytes1 = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[1]..self.aaunits[0].output_offsets[1] + 
                                              (frames*std::mem::size_of::<f32>())];
            (std::mem::transmute(bytes0), std::mem::transmute(bytes1))
        };

        // collect outputs from WASM so they are interlaced
        let collected_wasm_outputs = wasm_outputs0
            .iter()
            .zip(wasm_outputs1)
            .flat_map(|(x, y)| iter::once(x).chain(iter::once(y))); 

        let zipper_outputs = outputs0.zip(collected_wasm_outputs);
        for (output, wasm_output) in zipper_outputs {
            *output = *wasm_output;
        }

        Ok(())
    }

    /// compute audio for 1 input and 1 output channels
    #[inline]
    pub fn compute_two_one(&self, frames: usize, inputs: &[f32], outputs: &mut [f32]) -> Result<()> {
        // setup and copy input audio
        let inputs0 = inputs[0..frames as usize].iter();
        let (wasm_inputs0, wasm_inputs1): (&mut [f32],&mut [f32]) = unsafe { 
            let bytes0 = 
                &mut self.aaunits[0].memory.data_unchecked_mut()[self.aaunits[0].input_offsets[0]..self.aaunits[0].input_offsets[0] 
                                                      + (frames*std::mem::size_of::<f32>())];
            let bytes1 = 
                &mut self.aaunits[0].memory.data_unchecked_mut()[self.aaunits[0].input_offsets[1]..self.aaunits[0].input_offsets[1] 
                                                      + (frames*std::mem::size_of::<f32>())];
            (std::mem::transmute(bytes0), std::mem::transmute(bytes1))
        };

        let collected_wasm_outputs = wasm_inputs0
            .iter_mut()
            .zip(wasm_inputs1)
            .flat_map(|(x, y)| iter::once(x).chain(iter::once(y)));

        let zipped_inputs = inputs0.zip(collected_wasm_outputs);
        for (input, wasm_input) in zipped_inputs {
            *wasm_input = *input;
        }

        // now call compute
        let compute = self.aaunits[0].compute.get1::<u32, ()>()?;
        compute(frames as u32)?;

        // setup and copy audio out of WASM
        let outputs0 = outputs[0..frames as usize].iter_mut();
        let wasm_outputs0: &[f32] = unsafe { 
            let bytes = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[0]..self.aaunits[0].output_offsets[0] + 
                                              (frames*std::mem::size_of::<f32>())];
            std::mem::transmute(bytes)
        };

        let zipper_outputs = outputs0.zip(wasm_outputs0);
        for (output, wasm_output) in zipper_outputs {
            *output = *wasm_output;
        }

        Ok(())
    }

    /// compute audio for 2 input and 2 outputs channels
    /// assume that output channels are interlaced
    #[inline]
    pub fn compute_two_two(&self, frames: usize, inputs: &[f32], outputs: &mut [f32]) -> Result<()> {
        // setup and copy input audio
        let inputs0 = inputs[0..frames as usize].iter();
        let (wasm_inputs0, wasm_inputs1): (&mut [f32],&mut [f32]) = unsafe { 
            let bytes0 = 
                &mut self.aaunits[0].memory.data_unchecked_mut()[self.aaunits[0].input_offsets[0]..self.aaunits[0].input_offsets[0] 
                                                      + (frames*std::mem::size_of::<f32>())];
            let bytes1 = 
                &mut self.aaunits[0].memory.data_unchecked_mut()[self.aaunits[0].input_offsets[1]..self.aaunits[0].input_offsets[1] 
                                                      + (frames*std::mem::size_of::<f32>())];
            (std::mem::transmute(bytes0), std::mem::transmute(bytes1))
        };

        let collected_wasm_outputs = wasm_inputs0
            .iter_mut()
            .zip(wasm_inputs1)
            .flat_map(|(x, y)| iter::once(x).chain(iter::once(y)));

        let zipped_inputs = inputs0.zip(collected_wasm_outputs);
        for (input, wasm_input) in zipped_inputs {
            *wasm_input = *input;
        }

        // now call compute
        let compute = self.aaunits[0].compute.get1::<u32, ()>()?;
        compute(frames as u32)?;

        // setup and copy audio out of WASM
        // output is assumed to be interlaced
        let outputs0 = outputs[0..2 * frames as usize].iter_mut();
        let (wasm_outputs0, wasm_outputs1): (&[f32],&[f32]) = unsafe { 
            let bytes0 = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[0]..self.aaunits[0].output_offsets[0] + 
                                              (frames*std::mem::size_of::<f32>())];
            let bytes1 = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[1]..self.aaunits[0].output_offsets[1] + 
                                              (frames*std::mem::size_of::<f32>())];
            (std::mem::transmute(bytes0), std::mem::transmute(bytes1))
        };

        // collect outputs from WASM so they are interlaced
        let collected_wasm_outputs = wasm_outputs0
            .iter()
            .zip(wasm_outputs1)
            .flat_map(|(x, y)| iter::once(x).chain(iter::once(y))); 

        let zipper_outputs = outputs0.zip(collected_wasm_outputs);
        for (output, wasm_output) in zipper_outputs {
            *output = *wasm_output;
        }

        Ok(())
    }

    /// compute audio for 1 input and 2 outputs channels
    /// assume that output channels are interlaced
    #[inline]
    pub fn compute_zero_two(&self, frames: usize, outputs: &mut [f32]) -> Result<()> {
    
        // now call compute
        let compute = self.aaunits[0].compute.get1::<u32, ()>()?;
        compute(frames as u32)?;

        // setup and copy audio out of WASM
        // output is assumed to be interlaced
        let outputs0 = outputs[0..2 * frames as usize].iter_mut();
        let (wasm_outputs0, wasm_outputs1): (&[f32],&[f32]) = unsafe { 
            let bytes0 = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[0]..self.aaunits[0].output_offsets[0] + 
                                              (frames*std::mem::size_of::<f32>())];
            let bytes1 = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[1]..self.aaunits[0].output_offsets[1] + 
                                              (frames*std::mem::size_of::<f32>())];
            (std::mem::transmute(bytes0), std::mem::transmute(bytes1))
        };

        // collect outputs from WASM so they are interlaced
        let collected_wasm_outputs = wasm_outputs0
            .iter()
            .zip(wasm_outputs1)
            .flat_map(|(x, y)| iter::once(x).chain(iter::once(y))); 

        let zipper_outputs = outputs0.zip(collected_wasm_outputs);
        for (output, wasm_output) in zipper_outputs {
            *output = *wasm_output;
        }

        Ok(())
    }

    /// compute audio for 0 input and 1 output channels
    #[inline]
    pub fn compute_zero_one(&self, frames: usize, outputs: &mut [f32]) -> Result<()> {
        // now call compute
        let compute = self.aaunits[0].compute.get1::<u32, ()>()?;
        compute(frames as u32)?;

        // setup and copy audio out of WASM
        let outputs0 = outputs[0..frames as usize].iter_mut();
        let wasm_outputs0: &[f32] = unsafe { 
            let bytes = 
                &self.aaunits[0].memory.data_unchecked()[self.aaunits[0].output_offsets[0]..self.aaunits[0].output_offsets[0] 
                                              + (frames*std::mem::size_of::<f32>())];
            std::mem::transmute(bytes)
        };

        let zipper_outputs = outputs0.zip(wasm_outputs0);
        for (output, wasm_output) in zipper_outputs {
            *output = *wasm_output;
        }

        Ok(())
    }
}