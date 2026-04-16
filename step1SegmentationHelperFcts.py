
def runCellposeSegmentation(filePath, userInputList, saveFolder):
    import torch    
    from skimage.io import imsave
    import tifffile as tiff
    from cellpose import models, io
    from alive_progress import alive_bar
    import numpy as np
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")
    if userInputList.verboseMessages:
        if torch.cuda.is_available():
            print("cellpose is running on the GPU") 

    fileNames = list(filePath.glob("*.tif"))
    print("number of files: " + str(len(fileNames)))    
    model = models.CellposeModel(gpu=True)

    with alive_bar(len(fileNames)) as bar:
        for file in fileNames:
            tmpHyperstack = tiff.imread(file)
            if tmpHyperstack.ndim == 4:
                channelIndex = [i for i, dim in enumerate(tmpHyperstack.shape) if dim <= 5]
                tmpHyperstack = np.moveaxis(tmpHyperstack, channelIndex, 0)
            if tmpHyperstack.ndim == 3:
                tmpHyperstack = tmpHyperstack[np.newaxis, ...]
                userInputList.segmentationChannel = 0

            tmpFile = tmpHyperstack[userInputList.segmentationChannel,userInputList.cropStart:userInputList.cropEnd,:,:]
            tmpHyperstack = tmpHyperstack[:,userInputList.cropStart:userInputList.cropEnd,:,:]
            name = file.stem
            adjustFolderName = saveFolder.joinpath("adjustedChannels")
            adjustFolderName.mkdir(exist_ok = True)
            tiff.imwrite(adjustFolderName.joinpath(f"{name}_ch{userInputList.segmentationChannel}crop.tif"), tmpFile)
            
            imageSaveFolder = saveFolder.joinpath(f"img_{name}")
            imageSaveFolder.mkdir(exist_ok = True)
            tmpHyperName = imageSaveFolder.joinpath(file.name)
            tiff.imwrite(tmpHyperName, tmpHyperstack)

            if userInputList.verboseMessages:
                print(f"working on...{name}")
                print("Now running cellpose SAM...")   
                print(f"flow_threshold: {userInputList.flow_threshold}")
                print(f"cellprob_threshold: {userInputList.cellprob_threshold}")
                print(f"tile_norm_blocksize: {userInputList.tile_norm_blocksize}")
                print(f"minSize: {userInputList.minSize}")
                print(f"zAxis: {userInputList.zAxis}")

            flow_threshold = userInputList.flow_threshold
            cellprob_threshold = userInputList.cellprob_threshold
            tile_norm_blocksize = userInputList.tile_norm_blocksize
            minSize = userInputList.minSize
            zAxis = userInputList.zAxis

            masks, flows, styles = model.eval(tmpFile, batch_size=32, 
                                    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                    do_3D=True, z_axis = zAxis,
                                    min_size = minSize,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize}) # type: ignore
            if userInputList.verboseMessages:
                io.logger_setup()
 
            cropFolderName = imageSaveFolder.joinpath("crops")
            cropFolderName.mkdir(exist_ok = True)
            maskFolderName = imageSaveFolder.joinpath("maskCropsCellpose")
            maskFolderName.mkdir(exist_ok = True)

            imsave(imageSaveFolder.joinpath(f"{name}_CellposeMasks.tif"), masks)

            if userInputList.verboseMessages:
                print("cropping out individual cells now...")

            cropDictionary, maskDictionary, avgVolume = cropOutCellsFromMask(masks, tmpHyperstack, userInputList.buffer, imageName=name)

            for key, crop in cropDictionary.items():
                saveName = cropFolderName.joinpath(f"{key}.tif")   
                tiff.imwrite(saveName, crop, imagej=True) 
            
            for key,crop in maskDictionary.items():
                saveName = maskFolderName.joinpath(f"{key}.tif")               
                imsave(saveName, crop)
            
            csvFolderName = saveFolder.joinpath("dataframes")
            csvFolderName.mkdir(exist_ok = True)

            ##now let's generate a bounding box image from the cellpose masks....
            boundingBoxImage = addBoundingBox(cellPoseMaskFile=masks, cropFolderLoc=cropFolderName, buffer=userInputList.buffer)
            imsave(adjustFolderName.joinpath(f"{name}_boundingBox.tif"), boundingBoxImage)
            bar()
    


def cropOutCellsFromMask(maskImg, cellImg, buffer, imageName):
    import numpy as np

    volumeList = [] #np.zeros((0, 2), dtype=object)

    cropDictionary = {}
    maskDictionary = {}
    maxValue = np.max(maskImg)

    if cellImg.ndim == 4 and cellImg.shape[-1] < 5:
        cellImg = np.transpose(cellImg, (3,0,1,2))
    if cellImg.ndim == 3:
        cellImg = cellImg[np.newaxis, ...]

    for cropNumber in range(1, maxValue+1):
        cellName = imageName + "_cell_crop_" + str(cropNumber)
        cellCropMask = (maskImg == cropNumber).astype(np.uint8)
        nonZeroList = np.nonzero(cellCropMask)
        if (len(nonZeroList[0]) != 0):
            zLow, zHigh, yLow, yHigh, xLow, xHigh = addBufferToCrop(nonZeroList, buffer, maskImg)
            
            cropDictionary[cellName] = cellImg[:,zLow:zHigh, yLow:yHigh, xLow:xHigh]
            maskDictionary[cellName+"_CellposeMask"] = cellCropMask[zLow:zHigh, yLow:yHigh, xLow:xHigh] / cropNumber
            ##save volume number
            tmpCrop = cellCropMask[zLow:zHigh, yLow:yHigh, xLow:xHigh] 
            volumeList.append(calcVolume(tmpCrop,1,1))

    ##now that we've computed the volume for each crop let's get the average value
    avgVolume = np.median(volumeList)


    return cropDictionary, maskDictionary, avgVolume



def countNonZeroSlices(cellCropMask):
    import numpy as np
    nonZeroSlices = 0
    for zSlice in range(0,cellCropMask.shape[0]):
        if np.any(cellCropMask[zSlice,:,:] == 1):
            nonZeroSlices += 1
    return nonZeroSlices


def calcVolume(maskCrop, xySpace, zSpace):
    import numpy as np
    pixelVolume = np.count_nonzero(maskCrop)
    umVolume = pixelVolume * xySpace* zSpace
    
    return umVolume

def addBufferToCrop(nonZero,buffer,origMasks):
    import numpy as np

    zmin, ymin, xmin = (ax.min() for ax in nonZero)
    zmax, ymax, xmax = (ax.max() for ax in nonZero)

    zpad = buffer / 2
    ypad = buffer
    xpad = buffer

    z_low  = np.clip(zmin - zpad, 0, origMasks.shape[0])
    z_high = np.clip(zmax + zpad, 0, origMasks.shape[0])

    y_low  = np.clip(ymin - ypad, 0, origMasks.shape[1])
    y_high = np.clip(ymax + ypad, 0, origMasks.shape[1])

    x_low  = np.clip(xmin - xpad, 0, origMasks.shape[2])
    x_high = np.clip(xmax + xpad, 0, origMasks.shape[2])

    return tuple(map(int, (z_low, z_high, y_low, y_high, x_low, x_high)))



def addBoundingBox(cellPoseMaskFile, cropFolderLoc, buffer):
    import numpy as np
    import re
    print("creating bounding box image now...")
    print(cropFolderLoc)
    cropFiles = list(cropFolderLoc.glob("*.tif"))
    boxImage = np.zeros(cellPoseMaskFile.shape)

    maxValue = len(cropFiles)-1
    for loc in range(0,maxValue):        
        file = cropFiles[loc]
        name = file.stem
        maskNumber = re.split(r'_',name)[-1]
        maskNumber = int(maskNumber)
        
        cellPoseCopy = np.copy(cellPoseMaskFile)
        cellPoseCopy[cellPoseCopy != maskNumber] = 0

        nonZeroList = np.nonzero(cellPoseCopy)
        boundingBox = addBufferToCrop(nonZeroList, buffer=buffer, origMasks=cellPoseCopy)
        boxImage = addOutlineCrop(boxImage, boundingBox, maskNumber)

    return boxImage


def addOutlineCrop(boundingBoxImage, boundingBox, cellNumber):
    import numpy as np
    boundingBoxImageUpdate = np.copy(boundingBoxImage)
    zLow = boundingBox[0]
    zHigh = boundingBox[1]
    yLow = boundingBox[2]
    yHigh = boundingBox[3]
    xLow = boundingBox[4]
    xHigh = boundingBox[5]
    ##go through every z slice
    for slice in range(zLow, zHigh):
        ##start at y min/max and add number to all x values
        for xStep in range(xLow, xHigh):
            boundingBoxImageUpdate[slice][yLow][xStep] = cellNumber
            if yHigh == boundingBoxImage.shape[1]:
                boundingBoxImageUpdate[slice][yHigh-1][xStep] = cellNumber
            else:
                boundingBoxImageUpdate[slice][yHigh][xStep] = cellNumber
        for yStep in range(yLow, yHigh):
            boundingBoxImageUpdate[slice][yStep][xLow] = cellNumber
            if xHigh == boundingBoxImage.shape[2]:
                boundingBoxImageUpdate[slice][yStep][xHigh-1] = cellNumber
            else:
                boundingBoxImageUpdate[slice][yStep][xHigh] = cellNumber
    return boundingBoxImageUpdate


