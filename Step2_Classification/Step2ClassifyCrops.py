##--- Dependancies ---####
from pathlib import Path
import step2GeneralHelperFcts as genFct
import step2streamlitHelperFcts as streamFct
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="3D BoVW Step 2: Classification")
    
    # Backward compatible: if omitted, we prompt like you do now
    p.add_argument(
        "--imagepath",
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

    p.add_argument(
        "--user-inputs-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON file with classification parameters (same shape as user_inputs.json written to the output folder). "
        "Omitted keys use built-in defaults. Ignored when --savepoint is used.",
    )

    p.add_argument(
        "--init-from-images",
        type=Path,
        default=None,
        metavar="DIR",
        help="Folder of .tif/.tiff crops when Step 1 was skipped. Creates a new dataset root "
        "(see --init-output-root) with allch_positiveCells/ and copies images, then exits. "
        "Re-run Step 2 with --imagepath set to that dataset root.",
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

    if args.init_from_images is not None:
        src = Path(args.init_from_images).expanduser()
        if not src.is_dir():
            raise SystemExit(f"Not a directory: {src}")
        out_root = src.parent / f"{src.name}_for_classification"
        try:
            n = genFct.prepare_layout_from_raw_images(src, out_root)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        print(
            f"Copied {n} image file(s) into:\n  {out_root / 'allch_positiveCells'}\n"
            f"Run Step 2 classification with:\n  --imagepath \"{out_root}\""
        )
        exit()

    filePath = args.imagepath or Path(input("Where are your images located?\n"))
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
        cfg_json = args.user_inputs_json
        if cfg_json is not None:
            cfg_json = Path(cfg_json).expanduser()
            if not cfg_json.is_file():
                raise SystemExit(f"User inputs JSON not found: {cfg_json}")
        # userInputList, saveFolder = genf.getSegmentationUserInputs(
        #     filePath, saveFolder, user_inputs_json=cfg_json
        # )
        userInputList, saveFolder, outputResults = genFct.getClassificationUserInputs(filePath, user_inputs_json=cfg_json)
        parameterFileLoc = genFct.save_user_inputs_json(userInputList, saveFolder / "user_inputs_classification.json", include_help = True)
        chCellCropLocation, sampleNumber = genFct.compileChCellCrops(filePath,userInputList,saveFolder)
        runFromSavePoint = False
    else:
        ##--Running from savepoint, grab user inputs from there ---
        _, _, outputResults = genFct.getClassificationUserInputs(filePath)
        userInputList = genFct.load_user_inputs_json(saveFolder / "user_inputs.json")
        parameterFileLoc = saveFolder / "user_inputs.json"
        chCellCropLocation = filePath.joinpath("allch_positiveCells")
        sampleNumber = len(list(chCellCropLocation.glob("*.tif")))
        runFromSavePoint = True

    if userInputList.removeSmallBadCrops:
        genFct.removeBadSmallBlankCrops(chCellCropLocation, filePath, saveFolder, userInputList)

    if userInputList.splitSmallAndLarge:
        genFct.sortLargeAndSmallNuclei(chCellCropLocation, filePath, saveFolder, userInputList)

    outputResults, finalMetaDataPath = genFct.runSingleScaleClassificationSparse(userInputList,chCellCropLocation,saveFolder,filePath, sampleNumber, outputResults, runFromSavePoint)

    streamlitAppFolder = saveFolder.joinpath("StreamAppClass")
    streamlitAppFolder.mkdir(exist_ok = True)  

    streamFct.writeStreamlitAppSparseClassification(streamlitAppFolder, finalMetaDataPath, parameterFileLoc, userInputList) 

    if userInputList.transferFiles:
        genFct.transferResults(saveFolder, userInputList)
    
    if userInputList.runStreamlitApp:
        streamFct.runStreamlitApp(streamlitAppFolder)



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

