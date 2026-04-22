


def testingImports():
    from dataclasses import dataclass
    from skimage.io import imread, imsave 
    import SimpleITK as sitk     
    import tifffile
    import os
    import torch    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("running on the GPU")
    from skimage.io import imread, imsave
    from cellpose import models
    import pandas as pd
    from alive_progress import alive_bar
    from scipy import stats
    from skimage.exposure import rescale_intensity
    import numpy as np
    from scipy.stats import zscore
    import subprocess
    from datetime import datetime
    from scipy import ndimage
    from skimage.filters import difference_of_gaussians


from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
import json
from typing import Any, Dict
import re

@dataclass
class UserInputs:
    ##--- General Parameters --##    
    segmentationChannel: int = field(default=0, metadata={"help": "Channel index used for segmentation input."})
    requiredChannels: list[int] = field(default_factory=lambda: [1, 2], metadata={"help": "Channel indices that must pass the intensity-vs-background test for a crop to be kept. segmentationChannel is never tested. Indices >= channel count in an image are skipped for that image."})
    signalThreshold: float = field(default=0.25, metadata={"help": "Threshold applied to signal when generating segmentation classes."})
    verboseMessages: bool = field(default=True, metadata={"help": "If True, print verbose debugging messages."})    
    classNames: list[str] = field(default_factory=lambda: ["Control", "LOF"],
        metadata={"help": "List of class names for analysis. For example ['Control', 'LOF'] or ['CGN_Control', 'GNP_nipblLOF']. These names are used for labeling the classes in the analysis and should correspond to the conditions/groups in your dataset. Separate conditions by _ or spaces or -."})

    ##--- Cellpose Parameters ----##
    flow_threshold:float = field(default = 0.4, metadata={"help": "Cellpose flow threshold parameter. Lower this is you see merged cell masks."})
    cellprob_threshold: float = field(default=0.0, metadata={"help": "Cellpose cell probability threshold parameter. Lower values will yield more masks, including more false positives potentially."})
    tile_norm_blocksize: int = field(default=100, metadata={"help": "Cellpose tile normalization block size parameter. Reduce if running into memory usage issues."})
    minSize: int = field(default=9000, metadata={"help": "Minimum cell size parameter."})
    zAxis: int = field(default=0, metadata={"help": "Z-axis parameter for 3D segmentation."})
    buffer: int = field(default=10, metadata={"help": "X-Y padding (pixels/voxels) added around the detected region."})
    imgZCropValue: int = field(default=10, metadata={"help": "Crop margin for the z-dimension for the entire image. cropStart=cropValue, cropEnd=min(-cropValue,-1)."})


    @staticmethod
    def parse_name(name: str) -> list[str]:
        parts = re.split(r"[_\-\s]+", name)
        return [
            re.sub(r"[^a-z0-9]", "", part.lower())
            for part in parts
            if part.strip()
        ]

    @property
    def classPatterns(self) -> list[list[str]]:
        return [self.parse_name(name) for name in self.classNames]


    @property
    def cropStart(self) -> int:
        return self.imgZCropValue

    @property
    def cropEnd(self) -> int:
        return min(-self.imgZCropValue, -1)



def getSegmentationUserInputs(filePath, saveFolder, user_inputs_json=None):
    if user_inputs_json is not None:
        userInputList = load_user_inputs_json(Path(user_inputs_json))
    else:
        userInputList = UserInputs()
    ##make new folder and save all results there
    if saveFolder is None:
        
        parentFolder = filePath.parent
        saveFolderName = f"{filePath.name}_Segmentation_ch{userInputList.segmentationChannel}"
        saveFolder = parentFolder.joinpath(saveFolderName)
        if not saveFolder.exists():
            saveFolder.mkdir()
        else:
            i = 0
            while saveFolder.exists():
                i += 1
                saveFolderName = f"{filePath.name}_Segmentation_ch{userInputList.segmentationChannel}_{i}"
                saveFolder = parentFolder.joinpath(saveFolderName)
            saveFolder.mkdir()

        printUserInputs(userInputList, filePath)
        if userInputList.verboseMessages:
            print("now making a new folder to save results in...")
            print(f"save folder: {saveFolder}")
    return userInputList, saveFolder



def printUserInputs(userInputList, filePath):
    from dataclasses import fields
    fileList = list(filePath.glob("*.tif"))
    print(f"""Welcome to 3D Bag of Visual Words Step One: Segmentation! \n
          Your file path has: {len(fileList)} files. Your user input parameters are                        
          """
          )
    for f in fields(userInputList):
            print(f"{f.name} = {getattr(userInputList, f.name)}")
            h = f.metadata.get("help")
            if h:
                print(f"    # {h}")
   


def user_inputs_to_dict(user_inputs, include_help: bool = False) -> Dict[str, Any]:
    """
    Convert a dataclass instance to a plain dict.
    Optionally include a _help block populated from field metadata.
    """
    if not is_dataclass(user_inputs):
        raise TypeError("user_inputs_to_dict expects a dataclass instance")

    data = asdict(user_inputs)  # type: ignore # fields only (properties like cropStart/cropEnd are not included)

    if include_help:
        data["_help"] = {
            f.name: f.metadata.get("help", "")
            for f in fields(user_inputs)
        }

    return data


def save_user_inputs_json(user_inputs, out_path, include_help: bool = True) -> Path:
    """
    Save user inputs to JSON with nice formatting.
    Returns the Path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = user_inputs_to_dict(user_inputs, include_help=include_help)

    # In case you later add Path objects or other non-JSON types
    def _json_default(o):
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    out_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return out_path



def load_user_inputs_json(json_path, cls = UserInputs, merge_with_defaults: bool = True):
    """
    Load user inputs from JSON and return an instance of `cls` (a dataclass).
    
    - cls: the dataclass type (e.g., UserInputs)
    - merge_with_defaults: if True, allows JSON to specify only some fields
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    # Remove helper block if present
    data.pop("_help", None)

    # Keep only keys that are actual dataclass fields
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    if merge_with_defaults:
        defaults = cls()                 # start from defaults
        defaults.__dict__.update(filtered)  # overwrite provided values
        return defaults
    else:
        # Requires JSON to contain all required fields (yours do, because defaults exist)
        return cls(**filtered)



