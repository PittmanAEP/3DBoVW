##--- Dependancies ---####
from pathlib import Path
import step2GeneralHelperFcts as genFct
import step2streamlitHelperFcts as streamFct
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="3D BoVW Step 2: Classification")
    
    # Backward compatible: if omitted, we prompt like you do now
    p.add_argument(
        "--imagePath",
        nargs="?",
        type=Path,
        help="Folder containing input .tif files"
    )

    p.add_argument(
        "--savepoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="Existing results folder to resume from (use absolute or relative path). This will restart the anaysis after the Cellpose segmentation step"
    )

    p.add_argument(
        "--testEnv",
        action="store_true",
        help = "test packages in environment, run if you just made a new environment or added new packages and want to check they work before running the full pipeline"
    )

    p.add_argument(
        "--generateCodebook",
        action="store_true",
        help = "Generate the codebook from the full (non-split) dataset"
    )

    return p.parse_args()


def resolve_savepoint(images_dir: Path, sp: Path | None) -> Path | None:
    if sp is None:
        return None
    sp = Path(sp).expanduser()
    # If user passes just a folder name, treat it as relative to images_dir.parent
    if not sp.is_absolute():
        sp = images_dir.parent / sp
    return sp



def main():
    args = parse_args()
    filePath = args.imagePath or Path(input("Where are your images located?\n"))
    filePath = Path(filePath).expanduser()

    if args.testEnv:
        genFct.testingImports()
    
    if args.generateCodebook:
        runFromSavePoint = False
        userInputList, saveFolder, outputResults = genFct.getClassificationUserInputs(filePath, generateCodebook = True)
        userInputList.splitSmallAndLarge = False
        userInputList.codebookFile = False
        userInputList.allImgCodebookFile = True
        parameterFileLoc = genFct.save_user_inputs_json(userInputList, saveFolder / "user_inputs.json", include_help = True)

        genFct.generateCodebookAllFiles(filePath,userInputList,saveFolder,outputResults)        
        print("Codebook generated. Exiting.")
        exit()

    if not filePath.exists():
        raise SystemExit(f"Sorry that path doesn't exist: {filePath}")

    saveFolder = resolve_savepoint(filePath, args.savepoint)
    if saveFolder is not None and not saveFolder.exists():
        raise SystemExit(f"Savepoint folder does not exist: {saveFolder}") 


    if saveFolder is None:
        userInputList, saveFolder, outputResults = genFct.getClassificationUserInputs(filePath)
        parameterFileLoc = genFct.save_user_inputs_json(userInputList, saveFolder / "user_inputs.json", include_help = True)
        chCellCropLocation, sampleNumber = genFct.compileChCellCrops(filePath,userInputList,saveFolder)
        runFromSavePoint = False
    else:
        ##--Running from savepoint, grab user inputs from there ---
        userInputList = genFct.load_user_inputs_json(saveFolder / "user_inputs.json")
        parameterFileLoc = saveFolder / "user_inputs.json"
        chCellCropLocation = filePath.joinpath("allch_positiveCells")
        runFromSavePoint = True

    if userInputList.removeSmallBadCrops:
        genFct.removeBadSmallBlankCrops(chCellCropLocation, filePath, saveFolder, userInputList)

    if userInputList.splitSmallAndLarge:
        genFct.sortLargeAndSmallNuclei(chCellCropLocation, filePath, saveFolder, userInputList)

    outputResults, finalMetaDataPath = genFct.runSingleScaleClassificationSparse(userInputList,chCellCropLocation,saveFolder,filePath, sampleNumber, outputResults, runFromSavePoint)

    streamlitAppFolder = saveFolder.joinpath("StreamAppClass")
    streamlitAppFolder.mkdir(exist_ok = True)  

    streamFct.writeStreamlitAppSparseClassification(streamlitAppFolder, finalMetaDataPath, parameterFileLoc, userInputList.zDriveSaveFolder) 

    if userInputList.transferFiles:
        genFct.transferResults(saveFolder, userInputList)



if __name__ == "__main__":
    main()










##----User Inputs -----###
# filePath = pathlib.Path(input("Where are your segmentation results located?\n"))
# if not filePath.exists:
#     print("Sorry that path doesn't exist. Try again?")
#     filePath = pathlib.Path(input('Where is your data located?\n')) 
# userInputList, saveFolder, outputResults = genFct.getClassificationUserInputs(filePath)
# if not userInputList.runFromSavePoint:
#     chCellCropLocation, sampleNumber = genFct.compileChCellCrops(filePath,userInputList,saveFolder)
# else:
#     chCellCropLocation, sampleNumber = genFct.runFromSavePoint(userInputList, filePath)

##-----Refine cell crops -----###

# if userInputList.matchKirbyMasks:
#     genFct.removeMismatchKirby(chCellCropLocation, saveFolder, userInputList)
# if userInputList.rerunSegmentation:
#     genFct.runSegmentation(chCellCropLocation, userInputList)
# if userInputList.splitSmallAndLarge:
#     genFct.sortLargeAndSmallNuclei(chCellCropLocation, filePath, saveFolder, userInputList)


# if userInputList.clusteringAl == "sparse":
#     ##---If loading from previous save point ----##
#     saveFolder = filePath.joinpath(userInputList.savePointFolder) if userInputList.runFromSavePoint else saveFolder
    
#     outputResults, finalMetaDataPath = genFct.runSingleScaleClassificationSparse(userInputList,chCellCropLocation,saveFolder,filePath, sampleNumber, outputResults)


##--- Make parameter file and streamlit apps ---###
# parameterFileName = genFct.writeParameterFile(filePath,saveFolder,userInputList, outputResults)
# streamlitAppFolder = saveFolder.joinpath("StreamAppClass")
# streamlitAppFolder.mkdir(exist_ok = True)  

# wordSparsePath = "word_signal_occupancy_sparse.xlsx" if userInputList.clusteringAl == "sparse" else "word_signal_occupancy.xlsx"
# wordOccDataLoc = streamlitAppFolder.parent.joinpath(wordSparsePath)

# if userInputList.nuclearVolume:
#     streamFct.writeStreamlitAppSparseClassificationNuclear(streamlitAppFolder, finalMetaDataPath, parameterFileName, userInputList.zDriveSaveFolder) 
#     streamFct.writeStreamlitOutputApp(streamlitAppFolder, finalMetaDataPath,parameterFileName, userInputList.zDriveSaveFolder)
# else:
#     streamFct.writeStreamlitAppSparseClassification(streamlitAppFolder, finalMetaDataPath, parameterFileName, userInputList.zDriveSaveFolder) 

# if userInputList.transferFiles:
#     genFct.saveResultsToZDrive(saveFolder, userInputList)

