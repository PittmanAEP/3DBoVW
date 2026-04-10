
def stitchnnUNetImg(dataLoc, buffer):
    from skimage.io import imread, imsave
    import numpy as np
    import re
    import warnings
    import SimpleITK as sitk
    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")

    nnUNetMasksFolder = dataLoc.joinpath("nnUNetFolders/nnUNetPostProcessed")
    cellposeMaskNames = list(dataLoc.rglob("*_CellposeMasks.tif"))

    saveLocNnunetMasks = dataLoc.joinpath("nnUNetMasks")
    saveLocNnunetMasks.mkdir(exist_ok=True)

    for cellposeFile in cellposeMaskNames:
        folderName = cellposeFile.name.split("_CellposeMasks.tif")[0]
        print("Working on..."+str(folderName))        

        cellPoseMaskFile = imread(cellposeFile)
        nnunetFiles = list(nnUNetMasksFolder.glob("*.nrrd"))
        stitchedImage = np.zeros(cellPoseMaskFile.shape)

        for mask in nnunetFiles:
            ##make sure that the file comes from the right image            
            name = mask.stem 
            tmpCellNumber = name.split("_")[-1]
            sourceFileName = name.rsplit("_",1)[0]
            
            if folderName in sourceFileName:
                tmpMask = sitk.ReadImage(mask)
                tmpMaskArray = sitk.GetArrayFromImage(tmpMask)  
                cellPoseCopy = np.copy(cellPoseMaskFile)
                cellPoseCopy[cellPoseCopy != int(tmpCellNumber)] = 0

                nonZeroList = np.nonzero(cellPoseCopy)
                zLow, zHigh, yLow, yHigh, xLow, xHigh = addBufferToCrop(nonZeroList, buffer=buffer, origMasks=cellPoseCopy)
                tmpBlock = stitchedImage[zLow:zHigh, yLow:yHigh, xLow:xHigh]
                tmpBlock[tmpBlock == 0] += tmpMaskArray[tmpBlock == 0]*int(tmpCellNumber)
                # tmpBlock[tmpBlock == 0] += tmpMaskArray[tmpBlock == 0]*int(tmpCellNumber)
                stitchedImage[zLow:zHigh, yLow:yHigh, xLow:xHigh] = tmpBlock

                imsave(saveLocNnunetMasks.joinpath("nnUNetMask_"+name+".tif"), tmpMaskArray)
                
        saveName = cellposeFile.parent.joinpath(folderName+"_nnUNetMasks.tif")
        imsave(saveName, stitchedImage)
        


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

    ##---- Get User Preferences
    signalThreshold = userInputList.signalThreshold
    folderPaths = list(saveFolder.glob("img*"))
    #---- Set up folder paths
    intensityResults = []

    folderLocCh1Positive = saveFolder.joinpath("adjustedChannels/ch1_positiveCells")
    folderLocCh1Positive.mkdir(parents=True, exist_ok = True)
    
    folderLocAllChPositive = saveFolder.joinpath("allch_positiveCells")
    folderLocAllChPositive.mkdir(exist_ok = True)

    #--- Sort cells into folders for positive signal
    print("Now sorting cell crops by signal in various channels...")
    for folder in folderPaths:
        imgNameOnly = folder.name.rsplit('img_')[1]
        mask = folder / f"{imgNameOnly}_CellposeMasks.tif"
        tmpMask = imread(mask) 
        print(f"Working on...{imgNameOnly}")

        tmpImageFolder = folder
        tmpCropFolderLoc = tmpImageFolder.joinpath("crops")
        tmpCellposeMaskFolder = tmpImageFolder.joinpath("maskCropsCellpose")

        tmpHyperstackWholeImg = tiff.imread(folder / f"{imgNameOnly}.tif")
        channelIndex = [i for i, dim in enumerate(tmpHyperstackWholeImg.shape) if dim <= 5]
        tmpHyperstack = np.moveaxis(tmpHyperstackWholeImg, channelIndex, 0)
        nChannel = tmpHyperstack.shape[0]
        
        if nChannel >= 3:
            folderLocCh2Positive = saveFolder.joinpath("adjustedChannels/ch2_positiveCells")
            folderLocCh2Positive.mkdir(exist_ok = True)

        if nChannel > 3:
            folderLocCh3Positive = saveFolder.joinpath("adjustedChannels/ch3_positiveCells")
            folderLocCh3Positive.mkdir(exist_ok = True)

        whole_img_bg = np.empty(nChannel, dtype=np.int32)
        for channel in range(nChannel):
            tmpChannel = tmpHyperstack[channel,:,:,:]
            tmpChannel[tmpMask != 0] = 0
            imgAvg = int(np.mean(tmpChannel[tmpChannel>0]))
            imgMode = int(stats.mode(tmpChannel[tmpChannel>0],axis=None)[0]) # type: ignore
            whole_img_bg[channel] = imgAvg if imgMode == 0 else imgMode

        vals = [whole_img_bg[i] if i < len(whole_img_bg) else None for i in range(4)]
        wholeImgAvgCh0, wholeImgAvgCh1, wholeImgAvgCh2, wholeImgAvgCh3 = vals

        # print(f"whole image avgs: ch0: {wholeImgAvgCh0}, ch1: {wholeImgAvgCh1}, ch2: {wholeImgAvgCh2}, ch3: {wholeImgAvgCh3}")
        
        numCellID = int(np.max(tmpMask))
        with alive_bar(numCellID) as bar:
            for cellID in range(1, numCellID+1):
                cellIDExistsBool = len(list(tmpCropFolderLoc.glob(f"*_cell_crop_{cellID}.tif"))) > 0
                if cellID in tmpMask and cellIDExistsBool:
                    tmpCellCrop = imread(list(tmpCropFolderLoc.glob(f"{imgNameOnly}_cell_crop_{cellID}.tif"))[0])
                    
                    tmpMaskPath = next(tmpCellposeMaskFolder.glob(f"{imgNameOnly}_cell_crop_{cellID}_CellposeMask.tif"))
                    tmpMaskCrop = imread(tmpMaskPath)
                    
                    ch1Positive = False
                    ch2Positive = False
                    ch3Positive = False 

                    tmpImgCh0 = np.copy(tmpCellCrop[0,:,:,:])
                    tmpImgCh0[tmpMaskCrop == 0] = 0
                    filteredCh0 = tmpImgCh0
                    # print(f"{tmpImageName} image and channel 0 mean: {np.mean(filteredCh0)} for cell {cellID}")
                    avgCh0 = np.mean(filteredCh0[filteredCh0>0]) 
                    tmpImgCh1 = np.copy(tmpCellCrop[1,:,:,:])
                    tmpImgCh1[tmpMaskCrop == 0] = 0
                    filteredCh1 = tmpImgCh1
                    avgCh1 = np.mean(filteredCh1[filteredCh1>0])
                    # print(f"{tmpImageName} image and channel 1 mean: {np.mean(filteredCh1)} for cell {cellID}")

                    if avgCh1 > (wholeImgAvgCh1 + wholeImgAvgCh1*signalThreshold):
                        ch1Positive = True
                        # print(f"Cell number {cellID} is positive for channel 1")
                        imsave(folderLocCh1Positive.joinpath(imgNameOnly+"_cell_crop_"+str(cellID)+"_hyperstack.tif"), tmpCellCrop)
                        imsave(folderLocCh1Positive.joinpath(f"{imgNameOnly}_cell_crop_{cellID}_CellposeMask.tif"), tmpMaskCrop)

                    if nChannel > 2:
                        # filteredCh2 = filteredHyperstack[2,:,:,:]
                        tmpImgCh2 = np.copy(tmpCellCrop[2,:,:,:])
                        tmpImgCh2[tmpMaskCrop == 0] = 0
                        filteredCh2 = tmpImgCh2 
                        avgCh2 = np.mean(filteredCh2[filteredCh2>0])
                        # print(f"{tmpImageName} image and channel 2 mean: {np.mean(filteredCh2)}")
                        if userInputList.requireCh2:
                            if avgCh2 > (wholeImgAvgCh2 + wholeImgAvgCh2*signalThreshold):
                                ch2Positive = True
                                # print(f"Cell number {cellID} is positive for channel 2")
                                imsave(folderLocCh2Positive.joinpath(f"{imgNameOnly}_cell_crop_{cellID}_hyperstack.tif"), tmpCellCrop)
                                imsave(folderLocCh2Positive.joinpath(f"{imgNameOnly}_cell_crop_{cellID}_CellposeMask.tif"), tmpMaskCrop)
                        else:
                            #user doesn't require ch2 so this is always positive 
                            ch2Positive = True
                    if nChannel > 3:
                        tmpImgCh3 = np.copy(tmpCellCrop[3,:,:,:])
                        tmpImgCh3[tmpMaskCrop == 0] = 0
                        filteredCh3 = tmpImgCh3
                        avgCh3 = np.mean(filteredCh3[filteredCh3>0])
                        # print(f"{tmpImageName} image and channel 3 mean: {np.mean(filteredCh3)} for cell {cellID}")
                        if avgCh3 > (wholeImgAvgCh3 + wholeImgAvgCh3*signalThreshold):
                            ch3Positive = True
                            # print(f"Cell number {cellID} is positive for channel 3")
                            imsave(folderLocCh3Positive.joinpath(f"{imgNameOnly}_cell_crop_{cellID}_hyperstack.tif"), tmpCellCrop)
                            imsave(folderLocCh3Positive.joinpath(f"{imgNameOnly}_cell_crop_{cellID}_CellposeMask.tif"), tmpMaskCrop)

                    fourChBool = nChannel==4 and ch1Positive and ch2Positive and ch3Positive
                    threeChBool = nChannel==3 and ch1Positive and ch2Positive

                    if fourChBool or threeChBool:
                        tmpCropName = f"{imgNameOnly}_cell_crop_{cellID}"
                        if fourChBool:
                            intensityResults.append({
                                "baseName": tmpCropName,
                                "wholeImgCh0": wholeImgAvgCh0,
                                "cellCropCh0": avgCh0,
                                "ch0SNR": avgCh0 / wholeImgAvgCh0, # pyright: ignore[reportOperatorIssue]
                                "wholeImgCh1": wholeImgAvgCh1,
                                "cellCropCh1": avgCh1,
                                "ch1SNR": avgCh1 / wholeImgAvgCh1, # pyright: ignore[reportOperatorIssue]
                                "wholeImgCh2": wholeImgAvgCh2,
                                "cellCropCh2": avgCh2,
                                "ch2SNR": avgCh2 / wholeImgAvgCh2, # pyright: ignore[reportOperatorIssue]
                                "wholeImgCh3": wholeImgAvgCh3,
                                "cellCropCh3": avgCh3,
                                "ch3SNR": avgCh3 / wholeImgAvgCh3, # pyright: ignore[reportOperatorIssue]
                            })
                        if threeChBool:
                            intensityResults.append({
                                "baseName": tmpCropName,
                                "wholeImgCh0": wholeImgAvgCh0,
                                "cellCropCh0": avgCh0,
                                "ch0SNR": avgCh0 / wholeImgAvgCh0, # pyright: ignore[reportOperatorIssue]
                                "wholeImgCh1": wholeImgAvgCh1,
                                "cellCropCh1": avgCh1,
                                "ch1SNR": avgCh1 / wholeImgAvgCh1, # pyright: ignore[reportOperatorIssue]
                                "wholeImgCh2": wholeImgAvgCh2,
                                "cellCropCh2": avgCh2,
                                "ch2SNR": avgCh2 / wholeImgAvgCh2 # pyright: ignore[reportOperatorIssue]
                            })
                        imsave(folderLocAllChPositive / f"{imgNameOnly}_cell_crop_{cellID}_hyperstack.tif", tmpCellCrop)
                        imsave(folderLocAllChPositive / f"{imgNameOnly}_cell_crop_{cellID}_ch0.tif", tmpCellCrop[0,:,:,:])
                        imsave(folderLocAllChPositive / f"{imgNameOnly}_cell_crop_{cellID}_CellposeMask.tif", tmpMaskCrop)

                    bar()
        
        dataFrame = pd.DataFrame(intensityResults)
        dataFrame.to_excel(saveFolder.joinpath("cell_intensities_SNR.xlsx")) 



