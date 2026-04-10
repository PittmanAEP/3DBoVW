##Dependancies####
from pathlib import Path
import step1GeneralHelperFcts as genf
import step1SegmentationHelperFcts as seghf
import step1AnalysisHelperFcts as anahf
import step1streamlitHelperFcts as streamhf
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="3D BoVW Step 1: Segmentation")

    # Backward compatible: if omitted, we prompt like you do now
    p.add_argument(
        "--imagePath",
        nargs="?",
        type=Path,
        help="Folder containing input .tif files"
    )

    # This is the style you said you like: flag + path
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
        genf.testingImports()

    if not filePath.exists():
        raise SystemExit(f"Sorry that path doesn't exist: {filePath}")

    saveFolder = resolve_savepoint(filePath, args.savepoint)
    if saveFolder is not None and not saveFolder.exists():
        raise SystemExit(f"Savepoint folder does not exist: {saveFolder}")  

    if saveFolder is None:
        userInputList, saveFolder = genf.getSegmentationUserInputs(filePath, saveFolder)
        parameterFileLoc = genf.save_user_inputs_json(userInputList, saveFolder / "user_inputs.json", include_help = True)
        ##---Run Cellpose (+ optional nnUNet) segmentation ---
        seghf.runCellposeSegmentation(filePath, userInputList, saveFolder)
    else:
        ##--Running from savepoint, grab save folder and user inputs from there ---
        userInputList = genf.load_user_inputs_json(saveFolder / "user_inputs.json")
        parameterFileLoc = saveFolder / "user_inputs.json"


    #-----Refine cell crops -----
    anahf.sortCellCropsByChannelSignal(userInputList, saveFolder)
    genf.refineSegmentationsSizeMatch(saveFolder, userInputList)

    ##--Generate metadata array ----
    metaDataFileName = anahf.generateMetaDataArray(saveFolder, userInputList)

    ##-- Prepare Streamlit visualization app ---
    streamlitFolder = saveFolder.joinpath("StreamlitApp")
    streamlitFolder.mkdir(exist_ok=True) 
    streamhf.moveTifsToFolder(saveFolder, streamlitFolder)
    finalMetaDataPath = saveFolder / metaDataFileName
    streamhf.prepareImgsForSegmentationMaxProj(saveFolder)
    streamhf.writeStreamlitAppSegmentation(streamlitFolder,finalMetaDataPath, parameterFileLoc)

if __name__ == "__main__":
    main()

