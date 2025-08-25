
import os
import numpy as np
import awkward as ak
from dspeed.vis.waveform_browser import WaveformBrowser
from .utils import get_detector_system_for_channelname, get_key_for_rawid

class BrowseTask:
    def __init__(self, fcn, detector: str, *, max_entries: int = 7, autodraw = True,
                 title: str|bool = False, verbosity: int = 0):
        """fcn has to take a list of input awkward arrays and should return a 1-d bool mask of events to be drawn
        Title: str -> use this; True -> use detector, False -> no title :(
        """
        self.fcn = fcn
        self.max_entries = max_entries
        self.detector = detector
        self.autodraw = autodraw
        self.max_entries_drawn = self.max_entries
        self.title = title
        self.verbosity = verbosity
    def initialize(self):
        self.files = []
        self.entries = []   # list of entries ak array (one for each file)
        self.nr_entries = 0
    def __call__(self, x, raw):
        bool_mask = self.fcn(x)
        return self._add_events(bool_mask, raw, self.max_entries)
    def _add_events(self, bool_mask, raw: str, max_entries: int):
        if bool_mask.ndim != 1:
            raise RuntimeError(f"Mask for drawing has to be 1-dim. Got {bool_mask.ndim}")
        entries = np.flatnonzero(bool_mask)
        return self._add_events_as_entries(entries, raw, max_entries)
    def _add_events_as_entries(self, entries, raw: str, max_entries: int):
        if len(entries) == 0:
            return False
        #disabled cropping, since that would be difficult to replicate for deriving classes XXX
        #entries = entries[:(min(len(entries), max_entries-self.nr_entries))] # slice off if we hit goal
        self.files.append(raw)
        self.entries.append(entries)
        if self.verbosity >= 1:
            print(f"Wanna draw file {os.path.basename(raw)}, entries {list(entries)}")
        self.nr_entries += len(entries) # disabled cropping XXX
        #self.nr_entries += min(len(entries), max_entries-self.nr_entries)
        return self.nr_entries >= max_entries # break condition for the loop
    def finalize(self):
        if self.autodraw:
            self.draw()
    def draw(self):
        self._draw(self.files, self.entries, self.nr_entries, self.max_entries_drawn, self.detector, self.verbosity, self.title)
    @staticmethod
    def _draw(files, entries, nr_entries, max_entries_drawn, detector, verbosity, title:str|bool=False):
        if len(files) == 0:
            print("No files found!")
            return
        if verbosity >= 0:
            print(f"We have {nr_entries} entries; plot {min(nr_entries, max_entries_drawn)} of them")
        browser = WaveformBrowser(
            files_in=files,
            lh5_group=f"/{detector}/raw",
            entry_list=entries,
            lines=[get_detector_system_for_channelname(detector).default_display_wf_name],
            #lines=["waveform_presummed"],
            n_drawn=min(nr_entries, max_entries_drawn)
        )
        #print(browser.lh5_it.read(0))
        browser.draw_next()#draw_current()
        if title:
            the_title = title if isinstance(title, str) else detector
            browser.ax.set_title(the_title)

class BrowseAnydetTask(BrowseTask):
    def __init__(self, fcn, *, channelmap, max_entries: int = 7, autodraw = True, oversearch: int = 1000, blacklist: list[str] = [], cycle: int = 1):
        """fcn has to take a list of ak arrays and return a 2-dim ak array of detector rawids to be drawn"""
        super().__init__(fcn, "", max_entries=(max_entries if oversearch == 0 else oversearch), autodraw=autodraw)
        self.max_entries_drawn = max_entries
        self.channelmap = channelmap # from LegendMetadata.channelmap
        self.detector_rawids = [] # list (per-file) of 2-dim ak arrays of drawable rawids per event in file
        self.blacklist = blacklist
        self.cycle = cycle # how many channels we want to have a look at
    def __call__(self, x, raw):
        rawid_ak = self.fcn(x) # has to be a 2-dim array (events, rawids)
        if rawid_ak.ndim != 2:
            raise RuntimeError(f"rawid array has to be 2-dim. Got {rawid_ak.ndim}")
        bool_mask = ak.any(rawid_ak, axis = -1)
        if len(rawid_ak[bool_mask]) > 0:   # superclass zero-suppresses so we have to do as well
            self.detector_rawids.append(rawid_ak[bool_mask])
        return self._add_events(bool_mask, raw, self.max_entries)
    def draw(self):
        # first we need to find a single drawable detector (WF browser cannot draw
        # different detectors per-event)
        # do that in temporary lists, so we keep the original data
        for _ in range(self.cycle):
            detector, files, entries, nr_entries, _ = self._singularize()
            super()._draw(files, entries, nr_entries, self.max_entries_drawn, detector, self.verbosity)
            self.blacklist.append(detector)
    def _singularize(self):
        # let's take the first detetor showing up in our list. This is ofc arbitrary.
        select_rawid = self._select_rawid(ak.flatten(self.detector_rawids, axis=None))#self.detector_rawids[0][0][0]
        print("Selected", get_key_for_rawid(self.channelmap, select_rawid))
        new_files = []
        new_entries = []
        new_rawids = []
        new_entries_nr = 0
        #print(select_rawid)
        for file, entries, rawids in zip(self.files, self.entries, self.detector_rawids):
            entries = entries[ak.any(rawids == select_rawid, axis=-1)]
            rawids = rawids[ak.any(rawids == select_rawid, axis=-1)] # just for completeness
            if len(entries) > 0:
                new_files.append(file)
                new_entries.append(entries)
                new_rawids.append(rawids)
                new_entries_nr += len(entries)
        if len(new_files) == 0:
            raise RuntimeError("reduced to 0...")
        return get_key_for_rawid(self.channelmap, select_rawid), new_files, new_entries, new_entries_nr, new_rawids 
    def _select_rawid(self, rawids_array:ak.Array):
        assert rawids_array.ndim == 1
        for rawid in rawids_array:
            detectorname = get_key_for_rawid(self.channelmap, rawid)
            if detectorname not in self.blacklist:
                return rawid