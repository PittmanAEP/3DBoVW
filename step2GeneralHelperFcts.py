

def testingImports():
    from dataclasses import dataclass, field, fields, is_dataclass, asdict
    from pathlib import Path
    import json
    from typing import Any, Dict
    import re
    exit("Testing imports worked! You can now run the full pipeline.")





from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
import json
from typing import Any, Dict
import re

@dataclass
class UserInputs():
    ##---General Parameters---##
    chToClassify: int = field(default = 1, metadata={"help": "The channel to classify."})
    siftNormalization: str = field(default="asinh", metadata={"help": "How do you want to normalize the SIFT descriptors? Options are 'asinh', '5-95', or 'none'. The 5-95 method scales the descriptor values based on the 5th and 95th percentiles, which can help mitigate the influence of outliers and improve the robustness of the descriptors."})
    hogNormalization: str = field(default="asinh", metadata={"help": "How do you want to normalize the HOG descriptors? Options are 'asinh', '5-95', or 'none'. The 5-95 method scales the descriptor values based on the 5th and 95th percentiles, which can help mitigate the influence of outliers and improve the robustness of the descriptors."})
    removeSmallBadCrops: bool = field(default = False, metadata={"help": "Do you want to remove small crops that are smaller than a certain threshold?"})
    splitSmallAndLarge: bool = field(default=False, metadata={"help": "Do you want to split the images into small and large categories for processing?"})
    rgbMask: bool = field(default=False, metadata={"help": "Do you want to use an RGB mask for image processing?"})
    segmentationAvailable: bool = field(default=False, metadata={"help": "Do you already have segmentation masks available for your images?"})
    segmentationChannel: int = field(default=0, metadata={"help": "If you have segmentation masks available, which channel are they in?"})
    nJobs: int = field(default=10, metadata={"help": "How many parallel jobs do you want to run when processing HOG features? Kirby's data = 10, Chris' data = 7"})
                
    transferFiles: bool = field(default=True, metadata={"help": "Do you want to transfer the processed files to a different location?"})
    zDriveSaveFolder: str = field(default="/research/groups/solecgrp/home/apittman1/Data_Analysis/UnsupervisedClassification/3D_HOG/NipblLOF/conCGN_lofCGN_HPC_results/conCGN_dateTesting_equalFileNumber/", metadata={"help": "If you want to transfer the processed files to a different location, specify the path to the folder where you want to save the files. Make sure to include the trailing slash."})
    
    useSegmentationMasks: bool = field(default=True, metadata={"help": "Do you want to use segmentation masks for image processing?"})
    singleChannel: bool = field(default=False, metadata={"help": "Do you want to process only a single channel?"})
    calcAttnToMask: bool = field(default = False,  metadata={"help": "Do you want to calculate the overlap of attention to masks?"})

    #--- SIFT parameters --- ##
    siftKeypointsLocation: str = field(default="all", metadata={"help": "Where are the keypoints localized to? Areas of high signal or low signal?"})
    siftThreshold: float = field(default=0.03, metadata={"help": "Default 0.03, decrease to find more keypoints, 0.09 worked well for subset of Chris' 3D data"})
    siftEdgeR: int = field(default=6, metadata={"help": "Default 6, increase to find more 'edge like' keypoints"})
    ablateMethod: str = field(default="level", metadata={"help": "Word, family, or level"})

    ##---HOG parameters ---##
    blockSize: int = field(default = 2, metadata={"help": "The number of cells in each block for HOG feature extraction. A block is a larger region that contains multiple cells, and the HOG features are normalized within each block. A common choice is 2, which means each block will contain 2x2 cells."})
    cellSize: int = field(default = 2, metadata={"help": "The size of each cell in the HOG feature extraction. A common choice is 2, which means each cell will be 2x2x2 voxels."})
    blockOverlap: float = field(default = 0.5, metadata={"help": "The overlap between adjacent blocks in the HOG feature extraction. A value of 0.5 means that blocks will overlap by 50%."})
    blockOccupancy: float = field(default = 0.5, metadata={"help": "The occupancy of each block in the HOG feature extraction. A value of 0.5 means that each block will be filled to 50% capacity."})
    zModifier: float = field(default = 0.5, metadata={"help": "A modifier for the z-axis in the HOG feature extraction."})
    thetaHistogramBins: int = field(default = 18, metadata={"help": "The number of bins for the theta histogram in the HOG feature extraction. A common choice is 18, which means the theta histogram will have 18 bins."})
    phiHistogramBins: int = field(default = 36, metadata={"help": "The number of bins for the phi histogram in the HOG feature extraction. A common choice is 36, which means the phi histogram will have 36 bins."})

    #----Dictionary parameters---##
    dictionarySize: int = field(default=200, metadata={"help": "The size of the visual words dictionary."})
    sparsityAlpha: float = field(default = 0.1, metadata={"help": "The sparsity alpha parameter for dictionary construction."}) 
    lassoAlpha: float = field(default = 1.0, metadata={"help": "The lasso alpha parameter for dictionary construction."})
    codebookFile: bool = field(default = True, metadata={"help": "Whether to look for a pre-existing codebook. if false, it will generate a new codebook. If true, it will look for a codebook with the name specified by the parameters above. If it can't find a codebook with that name, it will generate a new codebook."}) 
    allImgCodebookFile: bool = field(default = True, metadata={"help": "Whether to use all images for codebook construction. If codebook file is true then this parameter determines if the code looks for a dictionary made with the entire dataset or one built on a subset of images."})
    normalizeMethod: str = field(default = "tfidf l2", metadata={"help": "The normalization method for the features. Options are: tfidf, l2 norm, none, L2 tfidf, l2 exclude tfidf, tfidf L2, tfidf L2 threshold, l1 norm, tfidf l1, bm25 [l1 works well with sparse, l2 is convential]"})
    normSmooth: bool = field(default = False, metadata={"help": "Whether to smooth the normalized features for Tfidf normalization."})
    normSubLinear: bool = field(default = True, metadata={"help": "Whether to use sub-linear normalization  for Tfidf normalization."})
    pooling: str = field(default="sum", metadata={"help": "The pooling method for sparse word assignment for patch analysis. Options are: sum or max."})
   

    ##--- Logistic Regression Parameters ---##
    runLRSweep: bool = field(default=False, metadata={"help": "Whether to run a hyperparameter sweep for the logistic regression model, managed through Weights and Biases."})
    lrPenalty: str = field(default="l2", metadata={"help": "The penalty term for the logistic regression model."})
    lrC: float = field(default=14.978, metadata={"help": "The regularization strength for the logistic regression model."})
    nSplits: int = field(default=5, metadata={"help": "The number of folds for cross-validation."})
    classWeight: str = field(default="balanced", metadata={"help": "The class weight for the logistic regression model."})
    cGrid: tuple[float, ...] = field(default=(0.1, 0.3, 1.0, 3.0, 10.0), metadata={"help": "The grid of C values for hyperparameter tuning."})
    l1Grid: tuple[float, ...] = field(default=(0.1, 0.3, 0.5, 0.7), metadata={"help": "The grid of l1 values for hyperparameter tuning."})
    numberOfNeighbors: int = field(default=4, metadata={"help": "The number of neighbors to consider for k-NN based methods."})
    classNames: list[str] = field(default_factory=lambda: ["250114", "240710", "240621"],
        metadata={"help": "List of class names for analysis. For example ['Control', 'LOF'] or ['CGN_Control', 'GNP_nipblLOF']. These names are used for labeling the classes in the analysis and should correspond to the conditions/groups in your dataset. Separate conditions by _ or spaces or -."}
    )
    
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

    def get_group_name(self, image_name: str) -> str | None:
        image_tokens = set(self.parse_name(image_name))

        matches = [
            class_name
            for class_name, class_pattern in zip(self.classNames, self.classPatterns)
            if all(token in image_tokens for token in class_pattern)
        ]

        if len(matches) > 1:
            raise ValueError(
                f"Image '{image_name}' matched multiple classes: {matches}"
            )

        return matches[0] if matches else None

@dataclass
class OutputResults():

    reconstructionError: float = field(default = -1.0, metadata={"help": "The reconstruction error for the visual words representation."})
    foundAllImgCodebook: bool = field(default = False, metadata={"help": "Indicates whether a codebook built off of all images was found."})
    codebookFileName: str = field(default = "", metadata={"help": "The name of the codebook file."})
    newCodebook: bool = field(default = False, metadata={"help": "Indicates whether a new codebook was created."})


def getClassificationUserInputs(filePath, generateCodebook = False):
    
    userInputList = UserInputs()
    if generateCodebook:
        userInputList.splitSmallAndLarge = False
        userInputList.codebookFile = False
        userInputList.allImgCodebookFile = True

    ##make new folder and save all results there
    print("now making a new folder to save results in...")
    saveFolder = filePath.joinpath(f"Classify_{userInputList.dictionarySize}w_ch{userInputList.chToClassify}_{userInputList.siftNormalization}_split{userInputList.splitSmallAndLarge}_allImgDict{userInputList.allImgCodebookFile}")
    if not saveFolder.exists():
        saveFolder.mkdir()
    else:
        i = 0
        while saveFolder.exists():
            i += 1
            saveFolderName =f"Classify_{userInputList.dictionarySize}w_ch{userInputList.chToClassify}_{userInputList.siftNormalization}_split{userInputList.splitSmallAndLarge}_allImgDict{userInputList.allImgCodebookFile}_{i}"
            saveFolder = filePath.joinpath(saveFolderName)
        saveFolder.mkdir()


    printUserInputs(userInputList, filePath)

    outputResults = OutputResults()

    extra = {
    "note": "User Inputs for Classification",
    "results_path": saveFolder}

    config = build_run_config(userInputList, extra=extra)

    # Choose where to save (without extension). For example, inside the results folder:
    out_base = saveFolder / "run_config"

    save_run_config(config, str(out_base))

    return userInputList, saveFolder, outputResults







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





def build_run_config(user_inputs, extra=None):
    """
    Combine your dataclass with optional extra fields.
    Everything is converted to basic (serializable) types.
    """
    from datetime import datetime, timezone
    import sys, platform

    cfg = {
        "user_inputs": _to_basic(user_inputs),
        "meta": _to_basic({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        })
    }
    if extra:
        # Put ad-hoc fields under meta to avoid clobbering user_inputs
        cfg["meta"].update(_to_basic(extra)) # type: ignore
    return cfg

def save_run_config(config, out_base_path):
    """
    Pretty-print to console and save:
      - {out_base_path}.yml
      - {out_base_path}.json
    """
    import json
    import yaml

    # Console (YAML is very readable)
    print("\n=== Run Configuration ===")
    print(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))

    # Save YAML
    with open(f"{out_base_path}.yml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    # Save JSON (stable diffs)
    with open(f"{out_base_path}.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)



def _to_basic(x):
    """Recursively convert objects into JSON/YAML-safe types."""
    from dataclasses import asdict, is_dataclass
    from pathlib import Path
    from enum import Enum
    import numpy as np

    if is_dataclass(x):
        x = asdict(x) # type: ignore
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _to_basic(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_basic(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    # Fallback: readable string
    return repr(x)




# def runFromSavePoint(userInputList, filePath):

#     chCellCropLocation = filePath / userInputList.savePointFolder
#     allChFolder = filePath.joinpath("allch_positiveCells")
#     if userInputList.singleChannel:
#         allChList = list(allChFolder.rglob("*.tif"))
#     else:
#         allChList = list(allChFolder.rglob("*hyperstack*.tif"))
#     sampleNumber = len(allChList)

#     return chCellCropLocation, sampleNumber


def generateCodebookAllFiles(filePath,userInputList,saveFolder,outputResults):
    import step2ClassificationHelperFcts as classFct

    chCellCropLocation, sampleNumber = compileChCellCrops(filePath,userInputList,saveFolder)
    allImagesCellFeatureDict = classFct.detectSIFTKeypoints(saveFolder, userInputList)
    classFct.dictionarySavepoint(allImagesCellFeatureDict, saveFolder) 
    allImagesCellFeatureDict = classFct.extractKeypointsFeatures(chCellCropLocation,userInputList, saveFolder, allImagesCellFeatureDict)
    classFct.visualizeAnalyzedLocations(allImagesCellFeatureDict, saveFolder, userInputList)
    codebook, outputResults = classFct.generateSparseDictionary(allImagesCellFeatureDict, userInputList, filePath, outputResults)



def compileChCellCrops(filePath,userInputList, saveFolder):
    from skimage.io import imread, imsave
    import warnings 
    import shutil
    import pandas as pd
    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")

    print("moving files to new folders")
    dataFrameFolder = saveFolder.joinpath("dataframes")
    dataFrameFolder.mkdir(parents = True, exist_ok = True)
    ch2CropFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
    ch2CropFolder.mkdir(exist_ok = True)

    allChFolder = filePath.joinpath("allch_positiveCells")
    if userInputList.singleChannel:
        allChList = list(allChFolder.rglob("*.tif"))
    else:
        allChList = list(allChFolder.rglob("*hyperstack*.tif"))

    if userInputList.rgbMask:
        convertRGBMasktoBinary(allChFolder, userInputList)

    maskList = list(allChFolder.rglob("*ask*.tif"))
    segmentationList = list(allChFolder.rglob("*Seg*.tif")) if userInputList.segmentationAvailable else []
    badStringList = ["eroded", "cleaned", "edt", "rgb"]
    refinedMaskList = [mask for mask in maskList if not any(bad_str in mask.name.lower() for bad_str in badStringList)]

    for file in refinedMaskList:
        shutil.copy2(file, ch2CropFolder.joinpath(file.name))     

    for file in segmentationList:
        shutil.copy2(file, ch2CropFolder.joinpath(file.name)) 

    strChannel = str(userInputList.chToClassify)
    for crop in allChList:
        imgNameOnly = crop.name.rsplit("_hyperstack")[0]
        hyperName = ch2CropFolder.joinpath(crop.name)        
        tmpHyper = imread(crop)        

        if len(tmpHyper.shape) > 3:
            shutil.copy2(crop, hyperName) 
            tmpCrop = tmpHyper[userInputList.chToClassify,:,:,:]
        else:
            tmpCrop = tmpHyper

        finalSaveName = ch2CropFolder / f"{imgNameOnly}_ch{strChannel}.tif" if not userInputList.singleChannel else hyperName
        imsave(finalSaveName, tmpCrop)

    sampleNumber = len(allChList)
    print(f"sample number was {sampleNumber}")
    csvFilePath = filePath / "volume_threshold_summary.csv"
    if csvFilePath.exists():
        volumeData = pd.read_csv(filePath / "volume_threshold_summary.csv")
        volumeData.to_csv(dataFrameFolder / "volume_threshold_summary.csv")
    return ch2CropFolder, sampleNumber





def convertRGBMasktoBinary(allChFolder, userInputList):
    import numpy as np
    import tifffile as tiff
    import step2AnalysisHelperFcts as analysisFct

    rgbMaskList = allChFolder.glob("*rgbmask.tif")
    wiggleRoom = 0.1
    nZ = 18  # repeat count for z-dimension

    for rgbMaskPath in rgbMaskList:
        tmpMask = tiff.imread(rgbMaskPath)
        rgbMask = analysisFct.ensure_channels_last(tmpMask)      
        R_mask = rgbMask[..., 0]
        G_mask = rgbMask[..., 1]
        B_mask = rgbMask[..., 2]

        total_cell_mask = (G_mask > wiggleRoom) | (B_mask > wiggleRoom)
        red_only_bg = (R_mask > wiggleRoom) & (G_mask <= wiggleRoom) & (B_mask <= wiggleRoom)
        total_cell_mask = total_cell_mask & (~red_only_bg)

        # 2D -> 3D (18, y, x)
        mask2d = total_cell_mask.astype(np.uint8)          # values 0/1
        mask3d = np.repeat(mask2d[None, ...], nZ, axis=0)  # (18, y, x)
        # print(f"mask shape was {mask3d.shape}")

        # Save
        outPath = rgbMaskPath.with_name(rgbMaskPath.name.replace("rgbmask", "mask"))
        tiff.imwrite(outPath, mask3d, photometric="minisblack", imagej=True,
            metadata={"axes": "ZCYX"},)





def removeBadSmallBlankCrops(chCellCropLocation, filePath, saveFolder, userInputList):
    import shutil
    from skimage.io import imread
    import numpy as np

    ##find txt file that has list of bad crops
    badFolderLocation = saveFolder.joinpath("badCrops")
    badFolderLocation.mkdir(exist_ok = True)     

    userRemovedCGN = 0
    userRemovedGNP = 0
    lowSignalRemovedCGN = 0   
    lowSignalRemovedGNP = 0 

    txtFileLoc = filePath.joinpath("bad_crop_list.txt")
    if not txtFileLoc.exists():
        print(f"can't find text file that lists bad cell crops. I looked here: {txtFileLoc}...")
        badFileNamesRefined = []
    else:
        with open(txtFileLoc, 'r') as file:
            content = file.read()     
        badFileNames = content.split()        
        badFileNamesRefined = [file.rsplit("_",1)[0] for file in badFileNames]
    fullListOfCellCrops = list(chCellCropLocation.glob("*.tif"))

    for file in fullListOfCellCrops:
        tmpMatchName = file.stem.rsplit("_",1)[0]
        if tmpMatchName in badFileNamesRefined:
            shutil.move(file, badFolderLocation.joinpath(file.name))
            if "CGN" in file.name:
                userRemovedCGN += 1
            if "GNP" in file.name:
                userRemovedGNP += 1
            print(f"bad crop: {file.stem}")
    
    if userInputList.chToClassify == 0:
      print("checking for good Hoechst signal in cell crops")
      imgList = list(chCellCropLocation.glob("*ch0.tif"))
      for img in imgList:
        tmpCh0Img = imread(img)
        tmpch0ImgName = img.stem
        tmpCellposeMaskName = chCellCropLocation.joinpath(tmpch0ImgName.replace("_ch0","_CellposeMask.tif"))
        tmpMask = imread(tmpCellposeMaskName)
        bgPixelsOnly = tmpCh0Img[tmpMask == 0]
        avgBgValue = np.mean(bgPixelsOnly)
        signalPixels = tmpCh0Img[tmpMask > 0]
        avgSignal = np.mean(signalPixels)
        if avgSignal < (avgBgValue + avgBgValue*.25):
            # print(f"Hoecht signal for {tmpch0ImgName} was too low, signal {avgSignal} to bg {avgBgValue}")
            if "CGN" in img.stem:
                lowSignalRemovedCGN += 1
            if "GNP" in img.stem:
                lowSignalRemovedGNP += 1
            strToMove = img.stem.rsplit("_",1)[0]
            listToMove = list(chCellCropLocation.glob(f"*{strToMove}*.tif"))
            for file in listToMove:            
                shutil.move(file, badFolderLocation.joinpath(file.name))

    print(f"CGN cells removed: {userRemovedCGN} cells removed by matching with bad list and {lowSignalRemovedCGN} removed due to low Hoechst signal.")
    print(f"GNP cells removed: {userRemovedGNP} cells removed by matching with bad list and {lowSignalRemovedGNP} removed due to low Hoechst signal.")


# def removeMismatchKirby(chCellCropLocation, saveFolder, userInputList):
#     import pandas as pd
#     import shutil
#     from collections import defaultdict

#     badFolderLocation = saveFolder.joinpath("badCrops")
#     excelFileFolder = saveFolder.parent / "overlapKirbyFiles"
#     listOfKirbyExcel = list(excelFileFolder.glob("*.xlsx"))
#     dictKirby = defaultdict()
#     chToClass = userInputList.chToClassify

#     for file in listOfKirbyExcel:
#         dateName = file.name.rsplit("_Segmentation")[0].lower()
#         dataFrame = pd.read_excel(file)
#         dictKirby[dateName] = dataFrame

#     listOfAllChPosImages = list(chCellCropLocation.rglob(f"*_ch{chToClass}.tif"))
#     movedCountCGN = 0
#     movedCountGNP = 0
#     keptCountGNP = 0
#     keptCountCGN = 0
#     for imgName in listOfAllChPosImages:
#         nameMatchStr = imgName.name.replace(f"_ch{chToClass}.tif", ".tif")
#         tmpDateName = imgName.name.rsplit("_image_")[0].lower()
#         tmpDF = dictKirby[tmpDateName]
#         iouScore = tmpDF.loc[tmpDF["cellName"].str.lower() == nameMatchStr.lower(), "overlapMeasurement"].values[0]
#         keepImg = iouScore > 0.6
#         if not keepImg:
#             listOfImgsToRemove = list(chCellCropLocation.glob(f"{imgName.stem.replace(f'_ch{chToClass}', '')}*"))
#             for fileToRemove in listOfImgsToRemove:
#                 shutil.move(fileToRemove, badFolderLocation.joinpath(fileToRemove.name))
#             if "CGN" in imgName.name:
#                 movedCountCGN += 1
#             if "GNP" in imgName.name:
#                 movedCountGNP += 1
#         else:
#             if "CGN" in imgName.name:
#                 keptCountCGN += 1
#             if "GNP" in imgName.name:
#                 keptCountGNP += 1
            
#     print(f"Total cells removed for not matching Kirby's masks: CGN: {movedCountCGN}, GNP: {movedCountGNP}")
#     print(f"Total cells kept: {keptCountGNP} GNP cells and {keptCountCGN} CGN cells.")


    
 
def sortLargeAndSmallNuclei(chCellCropLocation, filePath, saveFolder, userInputList):
    from skimage.io import imread
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    import shutil

    print("Loading in user preferences for size sorting...")
    dataFrameFolder = saveFolder.joinpath("dataframes")
    dataFrameFolder.mkdir(parents = True, exist_ok = True)
    volumeFileDF = pd.read_csv(filePath.joinpath("volume_threshold_summary.csv"))
    volumeFileDF["Group_match"] = volumeFileDF["Group"].str.replace(" ", "_")

    unusedCropFolderName = saveFolder / f"ch{userInputList.chToClassify}Crops_Unused"
    unusedCropFolderName.mkdir(exist_ok=True)

    imageFilesList = list(chCellCropLocation.glob(f"*ch{userInputList.chToClassify}.tif"))
    for file in imageFilesList:

        tmpCellposeMaskName = chCellCropLocation / file.name.replace(f"ch{userInputList.chToClassify}.tif","CellposeMask.tif")
        hyperstackName = chCellCropLocation / file.name.replace(f"ch{userInputList.chToClassify}.tif","hyperstack.tif")
        # nnMaskName = chCellCropLocation / file.name.replace(f"ch{userInputList.chToClassify}.tif","nnunetmask.tif")
        segMaskName = chCellCropLocation / file.name.replace(f"ch{userInputList.chToClassify}.tif",f"ch{userInputList.segmentationChannel}signalSeg.tif")
        ch1SegMaskName  = filePath / "allch_positiveCells" / file.name.replace(f"ch{userInputList.chToClassify}.tif",f"ch1signalSeg.tif")
        tmpCellposeMask = imread(tmpCellposeMaskName)
        # tmpSegmentation = imread(segMaskName) if userInputList.segmentationAvailable else np.zeros_like(tmpCellposeMask)
        tmpch1Seg = imread(ch1SegMaskName)
        # tmpVolumeSumCellpose = np.sum(tmpCellposeMask)
        # tmpVolumeNonZeroSeg = np.count_nonzero(tmpSegmentation)
        tmpVolume = np.sum((tmpch1Seg != 0).astype("uint8")) #tmpVolumeSumCh1Seg, used to match Step 1 volume calculation
        # tmpVolumeNoneroCellpose = np.count_nonzero(tmpCellposeMask)
        
        matched_group_row = None
        for idx, row in volumeFileDF.iterrows():
            group_tag = row["Group_match"]
            if group_tag in file.name:
                matched_group_row = row
                
                break

        threshold = matched_group_row["Threshold"] # pyright: ignore[reportOptionalSubscript]
        keep_small = matched_group_row["SmallVolume"] # type: ignore
        keep_large = matched_group_row["LargeVolume"] # pyright: ignore[reportOptionalSubscript]
        keep_all = matched_group_row["allKeep"] # type: ignore

        if not keep_all:
            if keep_small and tmpVolume <= threshold:
                continue  # Keep it
            elif keep_large and tmpVolume >= threshold:
                continue  # Keep it

            else:
                # Move all related files to unused folder     
                fileNameMatch = hyperstackName.name.rsplit("_hyperstack")[0]
                fileListToMove = list(chCellCropLocation.glob(f"{fileNameMatch}*"))
                for fileToMove in fileListToMove:
                    shutil.move(fileToMove, unusedCropFolderName / fileToMove.name)
                       
        else:
            print("This condition was not sorted by size.")

    #save volume choice csv file to save folder
    previousCellCount = pd.read_excel(filePath / "removed_cell_numbers.xlsx")
    print(f"counting after volume splits")
    countDict = defaultdict(int)

    listOfCells = list(chCellCropLocation.glob(f"*_ch{userInputList.chToClassify}.tif"))
    for file in listOfCells:
        conditionName = file.name.rsplit("_image_")[0]
        countDict[conditionName] += 1 

    counts_df = pd.DataFrame({
        "condition": list(countDict.keys()),
        "post volume split": list(countDict.values())})        
    cellCountDF = pd.merge(previousCellCount, counts_df, on="condition", how="outer") 
    cellCountDF.to_excel(dataFrameFolder / "removed_cell_numbers_post_split.xlsx")




def calcVolume(maskCrop, xySpace, zSpace):
    import numpy as np
    pixelVolume = np.count_nonzero(maskCrop)
    umVolume = pixelVolume * xySpace* zSpace
    
    return umVolume




def runSegmentation(chCellCropLocation, userInputList):
    import tifffile as tf
    import numpy as np
    from scipy.ndimage import distance_transform_edt, label
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")

    print("Now thresholding signal...")   
    chToThresh = userInputList.segmentationChannel
    folderLoc = chCellCropLocation
    imgFileList = list(folderLoc.rglob("*hyperstack.tif"))
    for file in imgFileList:
        imgNameOnly = file.stem.rsplit("_hyperstack")[0]
        maskName = imgNameOnly + f"_{userInputList.classMaskChoice}.tif"
        tmpMask = tf.imread(folderLoc / maskName)

        edtMask = distance_transform_edt(tmpMask)
        tmpEDTName = folderLoc / maskName.replace(".tif","EDT.tif")
        tf.imwrite(tmpEDTName, edtMask)

        tmpImg = tf.imread(file)[0,:,:,:]
        tmpThresChImg = folderLoc.joinpath(str(file.name).replace("hyperstack",f"ch{chToThresh}"))
        tf.imwrite(tmpThresChImg, tmpImg)
            
        bgOnlyPixels = tmpImg[tmpMask == 0]
        tmpImg[tmpMask == 0] = 0
        tmpZeros = np.zeros((tmpImg.shape))
        tmpImgStd = np.std(tmpImg[tmpImg > 0], axis=0)
        blownOutPixels = tmpImgStd*25
        tmpImgBGPerc = np.nanpercentile(bgOnlyPixels, 50, axis=None)
        
        bgThreshold = tmpImgBGPerc * 6.5
        tmpImgPerc = np.percentile(tmpImg[(tmpImg > 0) & (tmpImg<blownOutPixels)], 50)
        
        tmpZeros[tmpImg >= tmpImgPerc] = 1
        tmpZeros[(tmpImg > tmpImgBGPerc) & (tmpImg < tmpImgPerc)] = 2
        tmpZeros[(tmpZeros == 0) & (tmpMask > 0)] = 2
        tmpZeros[(tmpImg < bgThreshold) & (edtMask < 2)] = 0 # type: ignore                
        tmpName = f"{imgNameOnly}_ch{chToThresh}signalSeg.tif"
        tf.imwrite(folderLoc / tmpName, tmpZeros)





def runSingleScaleClassificationSparse(userInputList,chCellCropLocation,saveFolder,filePath, sampleNumber, outputResults, runFromSavePoint):
    import step2ClassificationHelperFcts as classFct
    import step2AnalysisHelperFcts as anaFct  
    ##--if loading from previous checkpoint ---
    if runFromSavePoint:
        allImagesCellFeatureDict = classFct.dictionaryLoad(saveFolder) 
        val_error = -1000    
        codebook = classFct.loadCodebookSavepoint(filePath, userInputList)   
        allImagesCellFeatureDict, tfidfObject = classFct.normalizeVectors(saveFolder, userInputList,allImagesCellFeatureDict)    

    else:
        #--Look for keypoints, extract info, and generate sparse codebook---##
        allImagesCellFeatureDict = classFct.detectSIFTKeypoints(saveFolder, userInputList)
        classFct.dictionarySavepoint(allImagesCellFeatureDict, saveFolder) 
        allImagesCellFeatureDict = classFct.extractKeypointsFeatures(chCellCropLocation,userInputList, saveFolder, allImagesCellFeatureDict)
        classFct.visualizeAnalyzedLocations(allImagesCellFeatureDict, saveFolder, userInputList)
        codebook, outputResults = classFct.generateSparseDictionary(allImagesCellFeatureDict, userInputList, filePath, outputResults)

        ##-- Compile frequency histograms, normalize ---##
        allImagesCellFeatureDict = classFct.sparseCodeAndComputeFrequency(allImagesCellFeatureDict, codebook, userInputList, saveFolder)  
        allImagesCellFeatureDict, tfidfObject = classFct.normalizeVectors(saveFolder, userInputList,allImagesCellFeatureDict) 

        ##--Save the dictionary and first value (for debugging) ---##
        classFct.dictionarySavepoint(allImagesCellFeatureDict, saveFolder) 
        classFct.saveFirstValueDictionary(allImagesCellFeatureDict, saveFolder)  
        
    ##-- Run linear regression, generate attention maps, analyze attention maps ---##
    if userInputList.runLRSweep:
        runLRSweep(userInputList, allImagesCellFeatureDict, saveFolder)                                                             
    
    anaFct.linearRegressMultiClasses(allImagesCellFeatureDict, saveFolder, userInputList, tfidfObject)
    # anaFct.runHaralickMultiClasses(allImagesCellFeatureDict, saveFolder, userInputList)
    
    if userInputList.calcAttnToMask:
        anaFct.calcAttnMaskOverlap(allImagesCellFeatureDict, saveFolder, userInputList)
    
    classFct.dictionarySavepoint(allImagesCellFeatureDict, saveFolder) 

    finalMetaDataPath = anaFct.runPCAandUmapSparse(allImagesCellFeatureDict, saveFolder, userInputList, sampleNumber) 

    return outputResults, finalMetaDataPath







def runLRSweep(userInputList, allImagesCellFeatureDict, saveFolder):      
    import wandb
    from functools import partial
    sweep_config = {
            "method": "bayes",                           
            "metric": {"name": "cv/auc_mean", "goal": "maximize"},
            "parameters": {
                "normalizeMethod": {"value":  "tfidf l2"},
                "smooth_idf": {"values": [True, False]},
                "sublinear_tf": {"values": [False, True]},
                "penalty": {"values": ["l2", "elasticnet"]},

                # Let Bayes search these as continuous ranges:
                "C":        {"distribution": "log_uniform", "min": 1e-3, "max": 10.0},
                "l1_ratio": {"distribution": "uniform",     "min": 0.0,  "max": 0.8}
            },
            # Optional: stop bad runs early
            "early_terminate": {"type": "hyperband", "min_iter": 3}
        }
    sweep_id = wandb.sweep(sweep_config, project="bovw-lr")

    wandb.agent(
        sweep_id,
        function=partial(sweep_entry, userInputList, allImagesCellFeatureDict, saveFolder),
        count=1000)  

def sweep_entry(userInputList, allImagesCellFeatureDict, saveFolder):
    import wandb
    from pathlib import Path
    import step2AnalysisHelperFcts as anaFct

    # Start the run that the sweep agent controls
    run = wandb.init(project="bovw-lr")  # the agent injects the sweep config here
    cfg = wandb.config

    # Map sweep config -> your userInputList
    userInputList.normalizeMethod = cfg.normalizeMethod
    userInputList.smooth_idf      = cfg.smooth_idf
    userInputList.sublinear_tf    = cfg.sublinear_tf
    userInputList.penalty         = cfg.penalty
    userInputList.C               = float(cfg.C)
    userInputList.l1_ratio        = float(getattr(cfg, "l1_ratio", 0.0))
    userInputList.n_splits        = getattr(userInputList, "n_splits", 5)

    # Use your existing trainer (see change #3 below to avoid double-init)
    anaFct.train_with_wandb(
        project="bovw-lr",
        allImagesCellFeatureDict=allImagesCellFeatureDict,
        savePath=Path(saveFolder) / "sweep_runs",
        userInputList=userInputList
    )





def transferResults(saveFolder, userInputList):
    import subprocess
    print(f"saving results to {userInputList.zDriveSaveFolder}")
    subprocess.run(["cp", "-r", saveFolder, userInputList.zDriveSaveFolder], check=True)
    
