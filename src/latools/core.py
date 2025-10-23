
from typing import Any
from collections.abc import Iterator, Callable
import awkward as ak
from lgdo.lh5 import read_as

def main_loop(inputArraysDef:list[tuple[str, str]], 
              genArrayDef:list[tuple[list[str],str,Callable[[list[ak.Array]],ak.Array]]], 
              outDef:list[tuple[list[str],Any]], # Callable[[list[ak.Array],str],bool]|
              *, tier_filename_dict: Iterator[dict[str,str]], 
              pre_reducer:tuple[list[str],Callable[[list[ak.Array]],ak.Array]]|None = None, 
              crop:bool=False):
    """
    Runs the main analysis loop, doing a single pass through the data, file-by-file.
    tier_filename_dict 

    Parameters
    ----------
    inputArraysDef
        list of mappings of short names to LH5 object names
    genArrayDef
        list of tuples of (1): list of shortnames existing (loaded in inputArraysDef of generated in genArrayDef before) arrays,
        (2): output shortname of new array, (3): function taking the list of arrays and producing the single output array
    outDef
        output functions/objects. A list of tuples of (1): list of shortnames existing (loaded in inputArraysDef of 
        generated in genArrayDef) arrays and (2) a function taking the list of arrays. The function might return a bool:
        True -> task done; no need to loop any more and False -> need to go on (None is "don't care").
        The function (then object) can define initialize / finalize functions, which get called before/after the loop.
    tier_filename_dict
        iterator yielding dicts of the kind {"raw": <raw_filename>, "dsp": <dsp_filename>, "evt": evt_filename}
        of all tiers which might be needed.
    pre_reducer
        a tuple of (1) a list of shortnames of arrays (have to exist in inputArraysDef) and (2) a function, 
        which takes the named array as input and returns a 1-d array of bools -> masking ALL arrays of inputArraysDef
        before further processing
    crop
        make all input array the same number of rows by cropping to the shortest one. USE WITH CARE!
    """
    def read_spec(spec: str, filename_tiers: dict[str, str]):
        spec_split = spec.strip("/").split("/")
        try:
            if (tier := spec_split[0]) in filename_tiers.keys():
                return read_as(spec, filename_tiers[tier], "ak")
            if (tier := spec_split[1]) in filename_tiers.keys():
                return read_as(spec, filename_tiers[tier], "ak")
        except KeyError as e:
            raise KeyError(f"Cannot find {spec} in file.") from e
        raise RuntimeError(f"Cannot identify tier name in spec {spec}")
    def compile_input_arrays(input_labels: list[str], arrays):
        ins = []
        for input in input_labels:
            ins.append(arrays[input])
        return ins
    def do_crop(arrays):
        min_length = min(map(len, arrays.values()))
        for key in arrays.keys():
            if len(arrays[key]) > min_length:
                print(f"Warning: cropping {key} from {len(arrays[key])} to {min_length}")
                arrays[key] = arrays[key][:min_length]
    
    for _, fcn in outDef:
        if hasattr(fcn, "initialize"):
            fcn.initialize()
    for filename_tiers in tier_filename_dict: # evt_files: # raw_and_evt_filenames():
        #filename_tiers = {"raw": gimme_raw_filename_from_evt(evt), "dsp": gimme_dsp_filename_from_evt(evt), "evt": evt}
        arrays = {}
        for short, spec in inputArraysDef:
            arrays[short] = read_spec(spec, filename_tiers) # read_as(spec, evt, "ak")
        if crop:
            do_crop(arrays)
        if pre_reducer is not None:
            mask = pre_reducer[1](compile_input_arrays(pre_reducer[0], arrays))
            for key in arrays.keys():
                arrays[key] = ak.mask(arrays[key], mask)
        for inputs, output, fcn in genArrayDef:
            arrays[output] = fcn(compile_input_arrays(inputs, arrays))
        flags = []  # output flags of functions. True -> I'm done now, False -> I'm not done yet, None -> I don't care
        for inputs, fcn in outDef:
            flags.append(fcn(compile_input_arrays(inputs, arrays), filename_tiers["raw"]))
        if any(flag == True for flag in flags) and all(flag != False for flag in flags):
            break
    for _, fcn in outDef:
        if hasattr(fcn, "finalize"):
            fcn.finalize()