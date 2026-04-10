


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

@dataclass
class UserInputs:
    buffer: int = field(default=10, metadata={"help": "Padding (pixels/voxels) added around the detected region."})
    cropValue: int = field(default=10, metadata={"help": "Crop margin. cropStart=cropValue, cropEnd=min(-cropValue,-1)."})
    segmentationChannel: int = field(default=1, metadata={"help": "Channel index used for segmentation input."})
    signalThreshold: float = field(default=0.25, metadata={"help": "Threshold applied to signal when generating segmentation classes."})
    specifyColNames: str = field(default="n", metadata={"help": "Column name specifier / prefix used for dataframe outputs."})
    chToThreshold: int = field(default=1, metadata={"help": "Channel used for intensity thresholding step."})
    verboseMessages: bool = field(default=True, metadata={"help": "If True, print verbose debugging messages."})
    levelErode: list = field(default_factory=lambda: [2, 2, 2], metadata={"help": "Erosion levels per axis (e.g., [Z,Y,X] or [X,Y,Z] depending on your convention)."})
    nnunetCh: int = field(default=0, metadata={"help": "nnUNet channel index if using nnUNet masks."})
    segmentationMaskChoice: str = field(default="Cellpose", metadata={"help": "Mask source: 'Cellpose' or 'nnUNet'."})
    requireCh2: bool = field(default=True, metadata={"help": "If True, require channel 2 to be present for processing."})
    matchKirby: bool = field(default=False, metadata={"help": "If True, apply Kirby-style matching/normalization steps."})
    filterLowHoechst: bool = field(default=False, metadata={"help": "Filter out samples with low Hoechst intensity."})
    filterSmallSize: bool = field(default=False, metadata={"help": "Filter out samples that are too small."})
    ##cellpose parameters
    flow_threshold:float = field(default = 0.4, metadata={"help": "Cellpose flow threshold parameter. Lower this is you see merged cell masks."})
    cellprob_threshold: float = field(default=0.0, metadata={"help": "Cellpose cell probability threshold parameter. Lower values will yield more masks, including more false positives potentially."})
    tile_norm_blocksize: int = field(default=100, metadata={"help": "Cellpose tile normalization block size parameter. Reduce if running into memory usage issues."})
    minSize: int = field(default=9000, metadata={"help": "Minimum cell size parameter."})
    zAxis: int = field(default=0, metadata={"help": "Z-axis parameter for 3D segmentation."})

    @property
    def cropStart(self) -> int:
        return self.cropValue

    @property
    def cropEnd(self) -> int:
        return min(-self.cropValue, -1)



def getSegmentationUserInputs(filePath, saveFolder):
    
    userInputList = UserInputs()
    ##make new folder and save all results there
    if saveFolder is None:
        print("now making a new folder to save results in...")
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





def compileChCellCrops(saveFolder,userInputList):
    from skimage.io import imread, imsave
    import numpy as np
    import warnings 

    saveName = "ch"+str(userInputList.chToClassify)+"Crops"
    ch2CropFolder = saveFolder.joinpath(saveName)
    if not ch2CropFolder.exists():
        ch2CropFolder.mkdir()

    allChFolder = saveFolder.joinpath("allch_positiveCells")
    allChHyperList = list(allChFolder.rglob("*hyperstack*.tif"))

    adChFolder = saveFolder.joinpath("adjustedChannels")
    chPosFolderName = "ch"+str(userInputList.chToClassify)+"_positiveCells"
    chPosFolderName = adChFolder.joinpath(chPosFolderName)

    maskList = list(chPosFolderName.rglob("*mask*.tif"))

    for file in maskList:
        tmpFile = imread(file)
        imsave(ch2CropFolder.joinpath(file.name), tmpFile)

    for hyperstack in allChHyperList:
        tmpHyper = imread(hyperstack)
        tmpName = hyperstack.stem.replace("_hyperstack","")
        if tmpHyper.shape[-1] < 5:
            tmpHyper = np.transpose(tmpHyper,(3,0,1,2))
        tmpch = tmpHyper[userInputList.chToClassify,:,:,:]
        strMatch = "_ch"+str(userInputList.chToClassify)+".tif"
        imsave(ch2CropFolder.joinpath(tmpName+strMatch), tmpch)

    ch2CellCropList = list(ch2CropFolder.glob("*.tif"))
    return ch2CellCropList, ch2CropFolder



def countCells(folderLoc, cellCountDF, countCondition):
    from pathlib import Path
    import pandas as pd
    from collections import defaultdict

    if countCondition == "all masks":
        print("counting all masks")
        countDict = defaultdict(int)
        folderNames = [p for p in folderLoc.glob("*_image_*") if p.is_dir()]
        listOfNames = [path.name.rsplit("_image_")[0] for path in folderNames]
        conditionList = list(set(listOfNames))

        for condition in conditionList:
            conditionFolderList = [folder for folder in folderNames if condition in folder.name]
            for folder in conditionFolderList:
                cellCrops = list(folder.rglob("*_cell_crop*.tif"))
                refinedCellCrops = [crop for crop in cellCrops if "Cellpose" not in crop.name]
                countDict[condition] += len(refinedCellCrops)
        
        counts_df = pd.DataFrame({
            "condition": list(countDict.keys()),
            "total cellpose masks": list(countDict.values())})        
        cellCountDF = pd.merge(cellCountDF, counts_df, on="condition", how="outer")
    
    if countCondition != "all masks":
        print(f"counting after {countCondition}")
        countDict = defaultdict(int)
        listOfCells = folderLoc.glob("*_ch0.tif")
        for file in listOfCells:
            conditionName = file.name.rsplit("_image_")[0]
            countDict[conditionName] += 1 

        if countCondition == "match Kirby":
            counts_df = pd.DataFrame({
                "condition": list(countDict.keys()),
                "matched with Kirby": list(countDict.values())})        
            cellCountDF = pd.merge(cellCountDF, counts_df, on="condition", how="outer")             
        elif countCondition == "post Hoechst":
            counts_df = pd.DataFrame({
                "condition": list(countDict.keys()),
                "above Hoechst threshold": list(countDict.values())}) 
            cellCountDF = pd.merge(cellCountDF, counts_df, on="condition", how="outer")
        elif countCondition == "3ch positive":
            counts_df = pd.DataFrame({
                "condition": list(countDict.keys()),
                "3-ch positive": list(countDict.values())}) 
            cellCountDF = pd.merge(cellCountDF, counts_df, on="condition", how="outer")
        elif countCondition == "removed small":
            counts_df = pd.DataFrame({
                "condition": list(countDict.keys()),
                "above volume threshold": list(countDict.values())}) 
            cellCountDF = pd.merge(cellCountDF, counts_df, on="condition", how="outer")

    return cellCountDF


def refineSegmentationsSizeMatch(saveFolder, userInputList):
    import pandas as pd
    import shutil
    from collections import defaultdict
    from skimage.io import imread
    import numpy as np

    ##--Set up paths ---
    chCellCropLocation = saveFolder / "allch_positiveCells"
    badFolderLocation = saveFolder.joinpath("badCrops")
    badFolderLocation.mkdir(exist_ok = True)  
    excelFileFolder = saveFolder / "overlapKirbyFiles"
    parameterFileLoc = saveFolder / "cropFilterParameters.txt"

    ##--Initialize counts---
    movedCountCGN = 0
    movedCountGNP = 0
    keptCountGNP = 0
    keptCountCGN = 0
    
    lowSignalRemovedCGN = 0   
    lowSignalRemovedGNP = 0 
    lowSizeRemovedCGN = 0
    lowSizeRemovedGNP = 0 

    ##--Initialize dataframe to keep track of moved cells --- 
    
    cellCountDF = pd.DataFrame(columns=["condition"])
    cellCountDF = countCells(saveFolder, cellCountDF, "all masks")
    cellCountDF = countCells(chCellCropLocation, cellCountDF, "3ch positive")

    ##--Remove low signal and low size if desired ---
    if userInputList.filterLowHoechst:
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
            cropSize = np.sum(tmpMask)            
            if avgSignal < (avgBgValue + avgBgValue*.25):
                # print(f"Hoecht signal for {tmpch0ImgName} was too low, signal {avgSignal} to bg {avgBgValue}")
                if "CGN" in img.stem:
                    lowSignalRemovedCGN += 1
                if "GNP" in img.stem:
                    lowSignalRemovedGNP += 1
                strToMove = img.stem.rsplit("_",1)[0]
                # print(f"string match to move was {strToMove}")
                listToMove = list(chCellCropLocation.glob(f"*{strToMove}_*.tif"))
                for file in listToMove:            
                    shutil.move(file, badFolderLocation.joinpath(file.name))

        cellCountDF = countCells(chCellCropLocation, cellCountDF, "post Hoechst")

    if userInputList.filterSmallSize:
        imgList = list(chCellCropLocation.rglob("*ch0.tif"))
        for img in imgList:
            tmpCh0Img = imread(img)
            tmpch0ImgName = img.stem
            tmpCellposeMaskName = chCellCropLocation.joinpath(tmpch0ImgName.replace("_ch0","_CellposeMask.tif"))
            tmpMask = imread(tmpCellposeMaskName)
            bgPixelsOnly = tmpCh0Img[tmpMask == 0]
            avgBgValue = np.mean(bgPixelsOnly)
            signalPixels = tmpCh0Img[tmpMask > 0]
            avgSignal = np.mean(signalPixels)
            cropSize = np.sum(tmpMask)        
            if cropSize < 12000:
                print(f"Size for {tmpch0ImgName} was too low, size was {cropSize}")
                if "CGN" in img.stem:
                    lowSizeRemovedCGN += 1
                if "GNP" in img.stem:
                    lowSizeRemovedGNP += 1
                strToMove = img.stem.rsplit("_",1)[0]
                listToMove = list(chCellCropLocation.glob(f"*{strToMove}_*.tif"))
                for file in listToMove:            
                    shutil.move(file, badFolderLocation.joinpath(file.name))
        cellCountDF = countCells(chCellCropLocation, cellCountDF, "removed small")
    
    ##--Match Kirby's masks---
    if userInputList.matchKirby:
        listOfKirbyExcel = list(excelFileFolder.glob("*.xlsx"))
        dictKirby = defaultdict()
        chToClass = userInputList.segmentationChannel

        for file in listOfKirbyExcel:
            dateName = file.name.rsplit("_Segmentation")[0].lower()
            dataFrame = pd.read_excel(file)
            dictKirby[dateName] = dataFrame

        listOfAllChPosImages = list(chCellCropLocation.rglob(f"*_ch{chToClass}.tif"))
        
        for imgName in listOfAllChPosImages:
            nameMatchStr = imgName.name.replace(f"_ch{chToClass}.tif", ".tif")
            # print(f"working on {imgName.name}")
            tmpDateName = imgName.name.rsplit("_image_")[0].lower()
            tmpDF = dictKirby[tmpDateName]
            iouScore = tmpDF.loc[tmpDF["cellName"].str.lower() == nameMatchStr.lower(), "overlapMeasurement"].values[0]
            keepImg = iouScore > 0.6
            if not keepImg:
                listOfImgsToRemove = list(chCellCropLocation.glob(f"{imgName.stem.replace(f'_ch{chToClass}', '')}*"))
                for fileToRemove in listOfImgsToRemove:
                    shutil.move(fileToRemove, badFolderLocation.joinpath(fileToRemove.name))
                if "CGN" in imgName.name:
                    movedCountCGN += 1
                if "GNP" in imgName.name:
                    movedCountGNP += 1
            else:
                if "CGN" in imgName.name:
                    keptCountCGN += 1
                if "GNP" in imgName.name:
                    keptCountGNP += 1
        cellCountDF = countCells(chCellCropLocation, cellCountDF, "match Kirby")
            
    
    parameterTextStr = f"""Parameters:
                were crops below a certain volume threshold removed? {userInputList.filterSmallSize}
                were crops with a low Hoechst signal removed? {userInputList.filterLowHoechst}
                were crops that didn't match Kirby's segmentations removed? {userInputList.matchKirby}
          
            CGN cells removed: {lowSizeRemovedCGN} removed due to small size. 
            GNP cells removed: {lowSizeRemovedGNP} removed due to small size.
            CGN cells removed: {lowSignalRemovedCGN} removed due to low Hoechst signal.
            GNP cells removed: {lowSignalRemovedGNP} removed due to low Hoechst signal.                    
            Total cells removed for not matching Kirby's masks: CGN: {movedCountCGN}, GNP: {movedCountGNP}")
            Total cells kept: {keptCountGNP} GNP cells and {keptCountCGN} CGN cells.")"""
    
    # print(parameterTextStr)
    
    with open(parameterFileLoc, 'w') as file:
         file.write(parameterTextStr)

    cellCountDF.to_excel(saveFolder / "removed_cell_numbers.xlsx")









# def removeBadSmallBlankCrops(saveFolder, userInputList):
#     import shutil
#     from skimage.io import imread
#     import numpy as np

#     chCellCropLocation = saveFolder / "allch_positiveCells"

#     ##find txt file that has list of bad crops
#     badFolderLocation = saveFolder.joinpath("badCrops")
#     badFolderLocation.mkdir(exist_ok = True)     

#     lowSignalRemovedCGN = 0   
#     lowSignalRemovedGNP = 0 

#     lowSizeRemovedCGN = 0
#     lowSizeRemovedGNP = 0 
    
#     print("checking for good Hoechst signal in cell crops")
#     imgList = list(chCellCropLocation.rglob("*ch0.tif"))
#     for img in imgList:
#         tmpCh0Img = imread(img)
#         tmpch0ImgName = img.stem
#         tmpCellposeMaskName = chCellCropLocation.joinpath(tmpch0ImgName.replace("_ch0","_CellposeMask.tif"))
#         tmpMask = imread(tmpCellposeMaskName)
#         bgPixelsOnly = tmpCh0Img[tmpMask == 0]
#         avgBgValue = np.mean(bgPixelsOnly)
#         signalPixels = tmpCh0Img[tmpMask > 0]
#         avgSignal = np.mean(signalPixels)
#         cropSize = np.sum(tmpMask)
#         if userInputList.filterLowHoechst:
#             if avgSignal < (avgBgValue + avgBgValue*.25):
#                 print(f"Hoecht signal for {tmpch0ImgName} was too low, signal {avgSignal} to bg {avgBgValue}")
#                 if "CGN" in img.stem:
#                     lowSignalRemovedCGN += 1
#                 if "GNP" in img.stem:
#                     lowSignalRemovedGNP += 1
#                 strToMove = img.stem.rsplit("_",1)[0]
#                 listToMove = list(chCellCropLocation.rglob(f"*{strToMove}*.tif"))
#                 for file in listToMove:            
#                     shutil.move(file, badFolderLocation.joinpath(file.name))
#         if userInputList.filterSmallSize:
#             if cropSize < 12000:
#                 print(f"Size for {tmpch0ImgName} was too low, size was {cropSize}")
#                 if "CGN" in img.stem:
#                     lowSizeRemovedCGN += 1
#                 if "GNP" in img.stem:
#                     lowSizeRemovedGNP += 1
#                 strToMove = img.stem.rsplit("_",1)[0]
#                 listToMove = list(chCellCropLocation.rglob(f"*{strToMove}*.tif"))
#                 for file in listToMove:            
#                     shutil.move(file, badFolderLocation.joinpath(file.name))
    
#     print(f"CGN cells removed: {lowSizeRemovedCGN} removed due to small size.")
#     print(f"GNP cells removed: {lowSizeRemovedGNP} removed due to small size.")
#     print(f"CGN cells removed: {lowSignalRemovedCGN} removed due to low Hoechst signal.")
#     print(f"GNP cells removed: {lowSignalRemovedGNP} removed due to low Hoechst signal.")


# def removeMismatchKirby(saveFolder, userInputList):
#     import pandas as pd
#     import shutil
#     from collections import defaultdict

#     chCellCropLocation = saveFolder / "allch_positiveCells"

#     badFolderLocation = saveFolder.joinpath("badCrops")
#     excelFileFolder = saveFolder / "overlapKirbyFiles"
#     listOfKirbyExcel = list(excelFileFolder.glob("*.xlsx"))
#     dictKirby = defaultdict()
#     chToClass = userInputList.segmentationChannel

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

