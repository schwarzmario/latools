import numpy as np
from dataclasses import dataclass

from lgdo.types.vectorofvectors import VectorOfVectors

@dataclass
class DetectorSystem:
    name: str
    default_display_wf_name: str

def get_detector_system_for_channelname(channelname: str) -> DetectorSystem:
    if channelname[0] == "S":
        return DetectorSystem("spms", "waveform_bit_drop")
    if channelname[0] == "B" or channelname[0] == "V" or channelname[0] == "P":
        return DetectorSystem("geds", "waveform_presummed")
    return DetectorSystem("useless", "NO_WAVEFORM")

def get_keys_in_detectorsystem(channelmap, detector_system: str) -> list[str]:
    return [key for key in channelmap if get_detector_system_for_channelname(key).name == detector_system]

def get_filtered_keys_in_detectorsystem(channelmap, detector_system: str, rawids_to_use: list[int]) -> list[str]:
    return [key for key in channelmap if (get_detector_system_for_channelname(key).name == detector_system and channelmap[key].daq.rawid in rawids_to_use)]

def get_key_for_rawid(channelmap, rawid: int) -> str:
    for key in channelmap.keys():
        if channelmap[key].daq.rawid == rawid:
            return key
    raise RuntimeError(f"Could not find rawid {rawid}")

# deprecated; use LegendMetadata now!
def map_detector_name_to_rawid(detector_names: VectorOfVectors, rawids: VectorOfVectors, prev_map: dict[bytes, np.uint32] = {}) -> dict[bytes, np.uint32]:
    this_map = prev_map.copy()
    for det_row, raw_row in zip(detector_names, rawids):
        for det, raw in zip(det_row, raw_row):
            this_map[det] = raw
            #print(det, raw)
    return this_map