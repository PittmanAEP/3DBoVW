

def addBufferToCrop(nonZero,buffer,origMasks):
    masks = origMasks
    if nonZero[0].min() - buffer/2 > 0:
        z_low = nonZero[0].min() - (buffer/2)
    else:
        z_low = 0
    if nonZero[0].max() + buffer/2 < masks.shape[0]:
        z_high = nonZero[0].max() + (buffer/2)
    else:
        z_high = masks.shape[0]
    
    if nonZero[1].min() - buffer > 0:
        y_low = nonZero[1].min() - buffer
    else:
        y_low = 0
    if nonZero[1].max() + buffer < masks.shape[1]:
        y_high = nonZero[1].max() + buffer
    else:
        y_high = masks.shape[1]
    
    if nonZero[2].min() - buffer > 0:
        x_low = nonZero[2].min() - buffer
    else:
        x_low = 0
    if nonZero[2].max() + buffer < masks.shape[2]:
        x_high = nonZero[2].max() + buffer
    else:
        x_high = masks.shape[2]

    boundingBox = [int(z_low),int(z_high), int(y_low),int(y_high), int(x_low),int(x_high)]
        
    return boundingBox


def generateMetaDataArray(saveFolder, userInput):
    import numpy as np
    import pandas as pd
    from skimage.io import imread
    if userInput.verboseMessages:
        print("Generating metadata array now...")
    allChPosFolder = saveFolder.joinpath("allch_positiveCells")
    allChPosFileList = list(allChPosFolder.rglob("*hyperstack.tif"))
    numberSplits = len(allChPosFileList[0].stem.split("_"))
    splitColumns = [f"name_{i}" for i in range(numberSplits)]
    rows = []
    for fileName in allChPosFileList:
        tmpWholeImg = imread(fileName)
        tmpImg = tmpWholeImg[0] if tmpWholeImg.ndim == 4 else tmpWholeImg
        parts = fileName.stem.split("_")
        splitDict = dict(zip(splitColumns, parts))
        rows.append({
            **splitDict,
            "baseName": fileName.stem,
            "Volume[pixels]": calcVolume(tmpImg,1,1),
            "avgIntensity": np.mean(tmpImg),
            "split": fileName.name.split("_"), 
        })

    finalMetaDataFrame = pd.DataFrame(rows)
    
    cellCropSNRLoc = saveFolder.joinpath("cell_intensities_SNR.xlsx")
    cellCropSNRDF = pd.read_excel(cellCropSNRLoc)
    # merged_df = pd.merge(finalMetaDataFrame, cellCropSNRDF, on='baseName', how='inner')

    metaDataFileName = saveFolder / f"{saveFolder.name}_CellPose_Seg_ch{userInput.segmentationChannel}_metadata.xlsx"
    finalMetaDataFrame.to_excel(metaDataFileName)
    
    return metaDataFileName


def calcVolume(maskCrop, xySpace, zSpace):
    import numpy as np
    pixelVolume = np.count_nonzero(maskCrop)
    umVolume = pixelVolume * xySpace* zSpace    
    return umVolume


def calcAreaToVolumeRatio(tmpMask, maskName):
    import numpy as np
    from skimage.measure import marching_cubes, mesh_surface_area

    lowSigMask = (tmpMask == 2).astype("uint8")
    highSigMask = (tmpMask == 1).astype("uint8")
    allSigMask = (tmpMask !=  0).astype("uint8")
    allSigVol = np.sum(allSigMask)

    noLowSignal = False
    noHighSignal = False
    
    if np.sum(lowSigMask) == 0:
        print(f"no low signal mask found, image was {maskName}")
        noLowSignal = True
        lowSigVol = 0 
        lowSigVolRatio = 0
        lowSigRatio = 0
    if np.sum(highSigMask) == 0:
        print(f"no high signal mask found, image was {maskName}")
        noHighSignal = True
        highSigVol = 0
        highSigVolRatio = 0
        highSigRatio = 0

    if not noLowSignal:
        vertsLow, facesLow, _, _ = marching_cubes(lowSigMask, level=0)
        lowSigSurfaceArea = mesh_surface_area(vertsLow, facesLow)
        lowSigVol = np.sum(lowSigMask)
        lowSigVolRatio = lowSigVol / allSigVol
        lowSigRatio = lowSigSurfaceArea / lowSigVol
    else:
        lowSigRatio = -1

    if not noHighSignal:
        vertsHigh, facesHigh, _, _ = marching_cubes(highSigMask, level=0)
        highSigSurfaceArea = mesh_surface_area(vertsHigh, facesHigh)
        highSigVol = np.sum(highSigMask) 
        highSigVolRatio = highSigVol / allSigVol
        highSigRatio = highSigSurfaceArea / highSigVol
    else:
        highSigRatio = -1
    
    if noHighSignal and noLowSignal:
        print(f"bad segmentation found for: {maskName}")
        allSigRatio = -1 
    else:        
        vertsAll, facesAll, _, _ = marching_cubes(allSigMask, level=0)
        allSigSurfaceArea = mesh_surface_area(vertsAll, facesAll)        
        allSigRatio = allSigSurfaceArea / allSigVol

    return lowSigRatio, highSigRatio, allSigRatio, lowSigVol, highSigVol, lowSigVolRatio, highSigVolRatio, allSigVol



def sortCellCropsByChannelSignal(userInputList, saveFolder):
    from skimage.io import imread, imsave
    import tifffile as tiff
    import numpy as np
    import warnings
    import pandas as pd
    from alive_progress import alive_bar
    from scipy import stats
    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")

    signalThreshold = userInputList.signalThreshold
    seg_ch = userInputList.segmentationChannel
    required = list(dict.fromkeys(userInputList.requiredChannels))

    def passes_intensity_threshold(avg, bg):
        if np.isnan(avg):
            return False
        return avg > (bg + bg * signalThreshold)

    def channel_positive_folder(c):
        p = saveFolder.joinpath(f"adjustedChannels/ch{c}_positiveCells")
        p.mkdir(parents=True, exist_ok=True)
        return p

    folderPaths = list(saveFolder.glob("img*"))
    intensityResults = []

    folderLocAllChPositive = saveFolder.joinpath("allch_positiveCells")
    folderLocAllChPositive.mkdir(exist_ok=True)

    if userInputList.verboseMessages:
        print("Now sorting cell crops by signal in various channels...")

    for folder in folderPaths:
        imgNameOnly = folder.name.rsplit("img_")[1]
        mask = folder / f"{imgNameOnly}_CellposeMasks.tif"
        tmpMask = imread(mask)
        if userInputList.verboseMessages:
            print(f"Working on...{imgNameOnly}")

        tmpCropFolderLoc = folder.joinpath("crops")
        tmpCellposeMaskFolder = folder.joinpath("maskCropsCellpose")

        tmpHyperstackWholeImg = tiff.imread(folder / f"{imgNameOnly}.tif")
        if tmpHyperstackWholeImg.ndim == 4:
            channelIndex = [i for i, dim in enumerate(tmpHyperstackWholeImg.shape) if dim <= 5]
            tmpHyperstack = np.moveaxis(tmpHyperstackWholeImg, channelIndex, 0)
        else:
            tmpHyperstack = tmpHyperstackWholeImg
        if tmpHyperstack.ndim == 3:
            tmpHyperstack = tmpHyperstack[np.newaxis, ...]
        nChannel = tmpHyperstack.shape[0]

        # Required channels for gating: user list minus segmentation channel; skip indices not in this image
        channels_required = [
            c for c in required
            if c != seg_ch and 0 <= c < nChannel
        ]
        
        if userInputList.verboseMessages:
            skipped = [c for c in required if c >= nChannel or c < 0]
            if skipped:
                print(f"  (skipping required channels not in stack {nChannel} ch): {skipped}")
            print(f"  intensity gate on channels (excl. seg ch {seg_ch}): {channels_required}")

        whole_img_bg = np.empty(nChannel, dtype=np.int32)
        for channel in range(nChannel):
            tmpChannel = np.array(tmpHyperstack[channel, :, :, :], copy=True)
            tmpChannel[tmpMask != 0] = 0
            outside = tmpChannel[tmpChannel > 0]
            if outside.size == 0:
                whole_img_bg[channel] = 0
            else:
                imgAvg = int(np.mean(outside))
                imgMode = int(stats.mode(outside, axis=None)[0])  # type: ignore
                whole_img_bg[channel] = imgAvg if imgMode == 0 else imgMode

        if userInputList.verboseMessages:
            parts = [f"ch{i}: {whole_img_bg[i]}" for i in range(nChannel)]
            print("  whole image bg " + ", ".join(parts))

        numCellID = int(np.max(tmpMask))
        with alive_bar(numCellID) as bar:
            for cellID in range(1, numCellID + 1):
                cellIDExistsBool = len(list(tmpCropFolderLoc.glob(f"*_cell_crop_{cellID}.tif"))) > 0
                if cellID in tmpMask and cellIDExistsBool:
                    tmpCellCrop = imread(
                        list(tmpCropFolderLoc.glob(f"{imgNameOnly}_cell_crop_{cellID}.tif"))[0]
                    )
                    if tmpCellCrop.ndim == 3:
                        tmpCellCrop = tmpCellCrop[np.newaxis, ...]
                    tmpMaskPath = next(
                        tmpCellposeMaskFolder.glob(f"{imgNameOnly}_cell_crop_{cellID}_CellposeMask.tif")
                    )
                    tmpMaskCrop = imread(tmpMaskPath)

                    avg = np.full(nChannel, np.nan, dtype=np.float64)
                    for c in range(nChannel):
                        tmpImg = np.copy(tmpCellCrop[c, :, :, :])
                        tmpImg[tmpMaskCrop == 0] = 0
                        inside = tmpImg[tmpImg > 0]
                        if inside.size > 0:
                            avg[c] = np.mean(inside)

                    for c in range(nChannel):
                        if c == seg_ch:
                            continue
                        if passes_intensity_threshold(avg[c], whole_img_bg[c]):
                            fpos = channel_positive_folder(c)
                            imsave(
                                fpos.joinpath(f"{imgNameOnly}_cell_crop_{cellID}_hyperstack.tif"),
                                tmpCellCrop,
                            )
                            imsave(
                                fpos.joinpath(f"{imgNameOnly}_cell_crop_{cellID}_CellposeMask.tif"),
                                tmpMaskCrop,
                            )

                    all_required = all(
                        passes_intensity_threshold(avg[c], whole_img_bg[c]) for c in channels_required
                    )

                    if all_required:
                        tmpCropName = f"{imgNameOnly}_cell_crop_{cellID}"
                        row = {"baseName": tmpCropName}
                        for c in range(nChannel):
                            bg = float(whole_img_bg[c])
                            a = float(avg[c])
                            row[f"wholeImgCh{c}"] = int(whole_img_bg[c])
                            row[f"cellCropCh{c}"] = a if not np.isnan(a) else np.nan
                            row[f"ch{c}SNR"] = (a / bg) if (bg != 0 and not np.isnan(a)) else np.nan
                        intensityResults.append(row)

                        imsave(
                            folderLocAllChPositive / f"{imgNameOnly}_cell_crop_{cellID}_hyperstack.tif",
                            tmpCellCrop,
                        )
                        imsave(
                            folderLocAllChPositive / f"{imgNameOnly}_cell_crop_{cellID}_ch0.tif",
                            tmpCellCrop[0, :, :, :],
                        )
                        imsave(
                            folderLocAllChPositive / f"{imgNameOnly}_cell_crop_{cellID}_CellposeMask.tif",
                            tmpMaskCrop,
                        )

                bar()

    pd.DataFrame(intensityResults).to_excel(saveFolder.joinpath("cell_intensities_SNR.xlsx"))


