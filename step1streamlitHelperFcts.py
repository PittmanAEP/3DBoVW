

def prepareImgsForSegmentationMaxProj(saveFolder, userInputList):
    import numpy as np
    from skimage.io import imread, imsave
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")
    if userInputList.verboseMessages:
        print("generating max projections for visualization")
    folderPaths = list(saveFolder.glob("img*"))
    maxProjectionFolder = saveFolder.joinpath("maxProjections")
    maxProjectionFolder.mkdir(exist_ok = True)
    for folder in folderPaths:
        imgNameOnly = folder.name.rsplit('img_')[1]
        mask = f"{imgNameOnly}_CellposeMasks.tif"

        tmpImg = imread(folder / f"{imgNameOnly}.tif")
        tmpMask = imread(folder / mask)

        tmpImgMax = np.max(tmpImg, axis=0)
        tmpMaskMax = np.max(tmpMask, axis=0)
        saveImgName = maxProjectionFolder / f"{imgNameOnly}_max.tif"
        saveMaskName = maxProjectionFolder / mask.replace(".tif", "_max.tif")

        imsave(saveImgName, tmpImgMax)
        imsave(saveMaskName, tmpMaskMax)


def moveTifsToFolder(saveFolder, streamlitFolder, userInputList):
    from skimage.io import imread, imsave
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")
    if userInputList.verboseMessages:
        print("copying full image files to streamlit folder")
    fileLoc = saveFolder / "maxProjections"
    fileList = fileLoc.rglob("*.tif")
    refinedList = [filePath for filePath in fileList if "Cellpose" not in filePath.name]
    for file in refinedList:
        imgName = file.name.replace("_max.tif", "")
        fullPath = saveFolder / imgName / f"{imgName}.tif"
        tmpFile = imread(fullPath)
        imsave(streamlitFolder / f"{imgName}.tif" ,tmpFile)

        fullMaskPath = saveFolder / imgName / f"{imgName}_CellposeMasks.tif"
        tmpMask = imread(fullMaskPath)
        imsave(streamlitFolder / f"{imgName}_CellposeMasks.tif", tmpMask)


def writeStreamlitAppSegmentation(streamlitAppFolder,finalMetaDataPath, parameterFileName, userInputList):
    import json
    import subprocess

    class_names_json = json.dumps(userInputList.classNames)
    if userInputList.verboseMessages:
        print("Writing Streamlit Script Now...")
    script = f"""
import altair as alt
import streamlit as st
import pandas as pd
import plotly.express as px
import tifffile as tiff
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates
import json

# ---- Setup paths ----
dataLoc = Path("{finalMetaDataPath}")
jsonFile = Path("{parameterFileName}")
imageBasePath = dataLoc.parent / "allch_positiveCells"
maxProjectionLoc = dataLoc.parent / "maxProjections"
wholeImgBasePath = maxProjectionLoc.parent

listOfWholeImgs = [p.name for p in wholeImgBasePath.glob("img_*") if p.is_dir()]

# ---- Define Functions ----
# Class/condition labels from Step 1 user_inputs (classNames); substring match on baseName
conditionList = {class_names_json}

def detect_condition(name):
    name_lower = name.lower()
    for condition in sorted(conditionList, key=len, reverse=True):
        if condition.lower() in name_lower:
            return condition
    return "Unassigned"



def normalize_contrast(image, min_val, max_val):
    clipped = np.clip(image, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val + 1e-8)  # avoid division by zero

def create_overlayed_pil(image, imgName, mask=None, show_mask=True, alpha_val=128):
    from PIL import Image
    import numpy as np

    min_val, max_val = np.percentile(image, (2, 98))
    norm_img = normalize_contrast(image, min_val, max_val)
    display_img = (norm_img * 255).astype(np.uint8)
    display_pil = Image.fromarray(display_img).convert("RGBA")

    if show_mask and mask is not None:
        rgba_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
        base_name = imgName.rsplit("img_", 1)[1]
        for cell_id in np.unique(mask):
            if cell_id == 0:
                continue  # skip background

            crop_name = f"{{base_name}}_cell_crop_{{cell_id}}_hyperstack"
            is_bad = crop_name in st.session_state.bad_list or crop_name not in existing_crop_names

            # Choose red for bad, blue for good
            color = (255, 0, 0) if is_bad else (0, 0, 255)

            # Create boolean mask for current ID
            current_mask = mask == cell_id
            for c in range(3):  # R, G, B
                rgba_mask[..., c][current_mask] = color[c]
            rgba_mask[..., 3][current_mask] = alpha_val

        mask_pil = Image.fromarray(rgba_mask, mode="RGBA")
        return Image.alpha_composite(display_pil, mask_pil)

    return display_pil



def get_clicked_cell_id(coords, mask, display_width):
    orig_width, orig_height = mask.shape[1], mask.shape[0]
    scale_x = orig_width / display_width
    scale_y = orig_height / display_width
    x = int(coords['x'] * scale_x)
    y = int(coords['y'] * scale_y)
    return mask[y, x]

def display_cell_crop_viewer(selected_image, cell_id, cropsLoc):
    import tifffile as tiff
    import matplotlib.pyplot as plt

    fullImgCropName = cropsLoc / f"{{selected_image}}_cell_crop_{{cell_id}}_hyperstack.tif"
    fullMaskCropName = cropsLoc / f"{{selected_image}}_cell_crop_{{cell_id}}_CellposeMask.tif"

    if fullImgCropName.exists():
        overlayCropMask = st.checkbox("Show cellpose mask", value=True)
        tmpImgVol = tiff.imread(fullImgCropName)[1,:,:,:]
        tmpMaskVol = tiff.imread(fullMaskCropName)
        zIndexCropViewer = st.slider("Z-slice to display", 0, tmpImgVol.shape[0]-1, round(tmpImgVol.shape[0]//2), key="crop_zIndex")
        alphaCropViewer = st.slider("Mask transparency", 0.0, 1.0, 0.5, key="crop_alpha")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(tmpImgVol[zIndexCropViewer], cmap='gray')
        if overlayCropMask:
            cellposeMasked = np.ma.masked_where(tmpMaskVol[zIndexCropViewer] == 0, tmpMaskVol[zIndexCropViewer])
            ax.imshow(cellposeMasked, cmap='summer', alpha=alphaCropViewer)
        ax.set_title(f"Image {{selected_image}}_cell_crop_{{cell_id}}.tif")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write(f"{{fullImgCropName}} wasn't included in the final analysis/can't be found")

@st.cache_data
def convert_df_to_excel(df):
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer)
        return output.getvalue()

def assign_color(row):
    name_key = row["names"] 
    if name_key in bad_list or name_key not in existing_crop_names:
        return "red"
    if name_key in clicked_data:
        return "green"
    else:
        return "blue"


# ---- Load Data ----
dataDF = pd.read_excel(dataLoc, index_col=0)
dataDF["condition"] = dataDF["baseName"].apply(detect_condition)
dataDF = dataDF.rename(columns={{"Volume[pixels]": "Volume_pixels"}})
existing_crop_names = set()
for f in imageBasePath.glob(f"*cell_crop_*_hyperstack.tif"):
    existing_crop_names.add(f.stem)

# ---- Initialize all the session state variables ----
if "selected_image" not in st.session_state:
    # Default to first image in your dataframe if none selected
    st.session_state.selected_image = dataDF.iloc[0]['baseName'] + ".tif"
    st.session_state.selected_cellposemask = dataDF.iloc[0]['baseName'].replace("hyperstack", "CellposeMask.tif")
if "clicked" not in st.session_state:
    st.session_state.clicked = set()
if "bad_list" not in st.session_state:
    st.session_state.bad_list = set()
if "last_clicked" not in st.session_state:
    st.session_state.last_clicked = None
if "filteredDF" not in st.session_state:
    st.session_state.filteredDF = dataDF.copy()
if "channel_selected" not in st.session_state:
    st.session_state.channel_selected = 0
if "removed_crops" not in st.session_state:
    st.session_state.removed_crops = set()
# Initialize session state index
if "img_index" not in st.session_state:
    st.session_state.img_index = 0
# Filter out removed crops
if st.session_state.removed_crops:
    dataDF = dataDF[~dataDF.index.isin(st.session_state.removed_crops)]


# ----- Set up data frames -------
palette = ["#FF007F", "#0080FF", "#00C853", "#FFB300"]  # up to 4 colors

color_map = {{
    condition: palette[i % len(palette)]
    for i, condition in enumerate(conditionList)
}}
color_map["Unassigned"] = "#9E9E9E"

# --- Handle plot click ---
clicked_data = st.session_state.get("clicked")
bad_list = st.session_state.get("bad_list")

howToTab, sizePlots, imageTab = st.tabs(["How to use", "Segmentation Results", "Images"])

with howToTab:
    st.markdown(
        r'''
### Overview

This app helps you **review 3D segmentation results**, compare **nuclear volumes** across conditions, set **volume-based filters**, and **flag individual cell crops** to exclude from further work. Data are loaded from your run's metadata spreadsheet and image folders next to it.

---

### Segmentation Results tab

- **Violin and histogram charts** plot nuclear volume (`Volume_pixels`) by **condition**. Conditions are inferred from each cell's `baseName` by substring match against `conditionList` (copied from `classNames` in your Step 1 `user_inputs.json` when this app was generated). Names that match no class are labeled **Unassigned**.
- **KDE plot** shows the distribution of volumes per condition, with **dashed vertical lines** for the thresholds you set in the sidebar of that section.
- Use **Set volume threshold per group** to enter a volume cutoff for each condition. The table below the chart shows how many cells fall **below** vs **above** each threshold.
- **Select volume type to keep for each group** uses one radio choice per condition: keep **small** volumes, **large** volumes, **none**, or **all**. These choices are stored in the summary table (boolean columns).
- Click **Save summary as CSV to disk** to write `volume_threshold_summary.csv` in the **same parent folder as your metadata Excel file** (next to `allch_positiveCells`, etc.). Use that file in downstream steps; nothing is written until you press save.

---

### Images tab

- Pick a **whole-image folder** (`img_*`) from the dropdown or use **Next Image** / the **Image Selection** buttons at the bottom.
- Toggle **Show cellpose mask** to overlay segment labels on the max projection or a single Z-slice (**Choose View Mode**).
- **Overlay colors:** **blue** masks are crops that appear in the analysis (matching hyperstack files under `allch_positiveCells`). **red** masks are excluded (e.g. you removed them) or have no matching crop file.
- **Click the overlaid image** to select a cell ID. The **right column** shows that crop's Z-stack viewer (optional Cellpose overlay on the crop).
- **Remove crop from dataset** adds that crop to the **excluded list** (session memory), colors it red on the whole-image view, and removes it from tables driven by `dataDF` for this session. It does **not** delete files on disk.

---

### Sidebar (all tabs)

- The sidebar shows a **single crop** corresponding to the app's **default spreadsheet selection** (`selected_image` / channel / Z / optional Cellpose overlay) and reports **average intensity inside the mask** for the chosen channel.
- **User excluded crops** lists every crop you marked with **Remove crop from dataset**, sorted by name.
- **Save excluded list to txt** writes `excluded_crops.txt` in the **metadata parent folder** (one hyperstack stem per line). Use that list to exclude the same crops in later pipelines. Until you save, excluded crops exist only in this browser session.

---

### Tips

- Refreshing or closing the page clears **session-only** state (excluded crops, threshold UI state). Save the CSV and the excluded-crops text file when you are done.
- If the sidebar warns that an image is missing, check that paths and filenames match what Step 1 produced (`allch_positiveCells`, hyperstack naming).
'''
    )

#-- Tab 2, plots
with sizePlots:
    figVolViolin = px.violin(
            st.session_state.filteredDF,
            x="condition",
            y="Volume_pixels",
            color="condition",
            box=True
        )

    figVolViolin.update_layout(
        title="Nuclear Size",
        xaxis_title="condition",
        yaxis_title="Volume (pixels)"
    )

    figVolViolin.update_xaxes(type='category')

    st.plotly_chart(figVolViolin, use_container_width=True)



    fig = px.histogram(
    st.session_state.filteredDF,
    x="Volume_pixels",
    color="condition",
    nbins=10,
    barmode="overlay",  # you can also try 'group'
    opacity=0.7
)

    fig.update_layout(
        title="Nuclear Volume by Species and Condition",
        xaxis_title="Nuclear Volume (pixels)",
        yaxis_title="Cell Count",
        bargap=0.05
    )

    st.plotly_chart(fig, use_container_width=True)
   # --- Step 4: Build Altair density plot ---
    
    group_means = dataDF.groupby("condition")["Volume_pixels"].mean().to_dict()
    tabChart, tabInput = st.columns([0.75,0.25])

    with tabInput:
        # Create a dictionary to store user-defined thresholds
        split_points = {{}}
        st.subheader("Set volume threshold per group")
        for group, mean_val in group_means.items():
            split_points[group] = st.number_input(
                f"Threshold for {{group}}",
                value=float(round(mean_val, 2)),
                step=1.0,
                format="%.2f",
                key=f"thresh_{{group}}"
            )

    with tabChart:            
        base = alt.Chart(dataDF)
        # Add density transform
        density = base.transform_density(
            density="Volume_pixels",       # Column to apply KDE on
            groupby=["condition"],        # Group by this column for separate curves
            as_=["volume", "density"]  # Output column names for KDE result
        )
        split_df = pd.DataFrame({{
            "condition": list(split_points.keys()),
            "threshold": list(split_points.values())
        }})

        split_lines = alt.Chart(split_df).mark_rule(strokeDash=[4, 3], size=2).encode(
            x="threshold:Q",
            color=alt.Color("condition:N", legend=None),
            tooltip=["condition", "threshold"]
        )

        # Create the chart
        kde_chart = density.mark_area(
            opacity=0.5
        ).encode(
            x=alt.X("volume:Q", title="Nuclear Volume (pixels)"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("condition:N", title="condition")  # condition coloring
        )

        # Display in Streamlit
        full_chart = (kde_chart + split_lines).properties(
        title="KDE of Nuclear Volumes with Group Means",
        width=600,
        height=400)
        
        st.altair_chart(full_chart, use_container_width=True)
    
        count_rows = []

        for group, threshold in split_points.items():
            subset = dataDF[dataDF["condition"] == group]
            below = (subset["Volume_pixels"] < threshold).sum()
            above = (subset["Volume_pixels"] >= threshold).sum()
            count_rows.append({{
                "condition": group,
                "Threshold": threshold,
                "Below Threshold": below,
                "Above Threshold": above
            }})

        summary_df = pd.DataFrame(count_rows)
        st.subheader("Cell counts per group based on thresholds")
        st.dataframe(summary_df.style.format({{"Threshold": "{{:.2f}}"}}))

        st.subheader("Select volume type to keep for each group")

        # Initialize lists to store user selections
        small_vol_list = []
        large_vol_list = []
        noKeepList = []
        allKeepList = []

        for group in summary_df["condition"]:
            selection = st.radio(
                label=f"{{group}}: Which volumes to keep?",
                options=["SmallVolume", "LargeVolume", "None", "All"],
                horizontal=True,
                key=f"volume_choice_{{group}}"
            )
            small_vol_list.append(selection == "SmallVolume")
            large_vol_list.append(selection == "LargeVolume")
            noKeepList.append(selection == "None")
            allKeepList.append(selection == "All")

        # Add columns to summary_df
        summary_df["SmallVolume"] = small_vol_list
        summary_df["LargeVolume"] = large_vol_list
        summary_df["noKeep"] = noKeepList
        summary_df["allKeep"] = allKeepList

        # Define file path
        saveLocation = dataLoc.parent / "volume_threshold_summary.csv"

        # Save button
        if st.button("💾 Save summary as CSV to disk"):
            summary_df.to_csv(saveLocation, index=False)
            st.success(f"Saved to: {{saveLocation}}")




with imageTab:
    colMax, colIndi = st.columns([1,1])
    with colMax: 
        image_names = listOfWholeImgs
        
        # Dropdown with default index
        colText, colButton = st.columns([0.75,0.25])
        with colButton:
            if st.button("Next Image"):
                st.session_state.img_index = (st.session_state.img_index + 1) % len(image_names)
        with colText:
            selected_image = st.selectbox("Select an image:", image_names, index=st.session_state.img_index)

        # Update the session index if user picks from dropdown
        current_index = image_names.index(selected_image)
        if current_index != st.session_state.img_index:
            st.session_state.img_index = current_index

        imagePath = maxProjectionLoc.parent / selected_image / f"{{selected_image.replace('img_', '')}}.tif"
        maskPath = maxProjectionLoc.parent / selected_image / f"{{selected_image.replace('img_', '')}}_CellposeMasks.tif"
        tmpImg = tiff.imread(imagePath)[1,:,:,:]
        # print(f"image {{imagePath}} was read in with shape {{tmpImg.shape}}")
        tmpMask = tiff.imread(maskPath)
        

        colCheck, colRadio = st.columns([1,1])
        with colCheck:            
            show_features = st.checkbox("Show cellpose mask", value=True, key="wholeImgCheckbox")
        with colRadio:
            view_option = st.radio(
                "Choose View Mode:",
                ["Max Projection", "Single Z-Slice"],
                index=0)  # 0 = first item
        
              
        
        z_indices = np.argmax(tmpImg, axis=0)
        display_width = 300

        if view_option == "Max Projection":
            img = np.max(tmpImg, axis=0)
            mask = np.max(tmpMask, axis=0)
        else:
            zIndexCrop = st.slider("Z-slice to display", 0, tmpImg.shape[0]-1, round(tmpImg.shape[0]//2))
            img = tmpImg[zIndexCrop]
            mask = tmpMask[zIndexCrop]

        overlayed_img = create_overlayed_pil(img,selected_image, mask=mask, show_mask=show_features)
        coords = streamlit_image_coordinates(overlayed_img, key="pil", width=display_width)
        if show_features:
            st.markdown(
                "<div style='font-size:0.85rem;margin-top:0.35rem;'>"
                "<span style='display:inline-block;width:11px;height:11px;background:#0000FF;"
                "vertical-align:middle;margin-right:5px;border-radius:2px;'></span>"
                "Included in analysis&nbsp;&nbsp;&nbsp;"
                "<span style='display:inline-block;width:11px;height:11px;background:#FF0000;"
                "vertical-align:middle;margin-right:5px;border-radius:2px;'></span>"
                "Not included"
                "</div>",
                unsafe_allow_html=True,
            )

        with colIndi:
            if coords is not None:
                cell_id = get_clicked_cell_id(coords, mask, display_width)
                display_cell_crop_viewer(selected_image.replace('img_', ''), cell_id, imageBasePath)
            
                crop_key = f"{{selected_image.replace('img_', '')}}_cell_crop_{{cell_id}}"

                if st.button("Remove crop from dataset"):
                    st.session_state.removed_crops.add(crop_key)
                    st.session_state.bad_list.add(f"{{crop_key}}_hyperstack")
                    st.success(f"{{crop_key}} removed from dataset.")
                    st.rerun()

    st.markdown("### 📂 Image Selection")

    for i, img_name in enumerate(listOfWholeImgs):
        # is_analyzed = img_name in st.session_state.imageFilterSummary
        is_current = i == st.session_state.img_index

        # label = f"✅ {{img_name}}" if is_analyzed else img_name
        if is_current:
            img_name = f"**{{img_name}}**"  # bold in markdown

        if st.button(img_name, key=f"img_btn_{{i}}"):
            st.session_state.img_index = i
            st.rerun()

# ---- Sidebar image viewer ----
with st.sidebar:
    st.header("Images and parameter info")
    

    selected_image_path = imageBasePath.joinpath(st.session_state.selected_image) 
    if selected_image_path.exists():        
        hyperImageVolume = tiff.imread(selected_image_path)
        if hyperImageVolume.shape[0] < 5:
            hyperImageVolume = np.transpose(hyperImageVolume, (1,2,3,0))  
    else:
        st.warning(f"Image not found: {{st.session_state.selected_image}}")
    selected_cellposemask_path = imageBasePath.joinpath(st.session_state.selected_cellposemask)

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Mask transparency", 0.0, 1.0, 0.5, key="sidebar_alpha")
    with col2:
        zIndex = st.slider("Z-slice to display", 0, hyperImageVolume.shape[0]-1, round(hyperImageVolume.shape[0]//2), key="sidebar_zIndex")
    
    cellposeMaskVolume = tiff.imread(selected_cellposemask_path)      
    st.text("Which mask(s) do you want to display?")
    show_cellpose = st.checkbox("cellpose mask", value=False)

    available_channels = list(range(hyperImageVolume.shape[-1]))
    st.session_state.channel_selected = st.selectbox("Choose image channel", available_channels, key="channel_select")
    imageVolume = hyperImageVolume[:, :, :, st.session_state.channel_selected]

    filteredImageVolume = np.where(cellposeMaskVolume, imageVolume, 0)  
    avgFilteredIntValue = np.mean(filteredImageVolume[filteredImageVolume>0]) 

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(imageVolume[zIndex], cmap='gray')
    if show_cellpose:
        cellposeMasked = np.ma.masked_where(cellposeMaskVolume[zIndex] == 0, cellposeMaskVolume[zIndex])
        ax.imshow(cellposeMasked, cmap='summer', alpha=alpha)
    ax.set_title(f"Image {{st.session_state.selected_image}}")
    ax.axis("off")
    st.pyplot(fig)
    st.text(f"Avg Intensity (inside mask) for this channel was {{round(avgFilteredIntValue)}}")

    st.divider()

    st.subheader("User excluded crops")
    bad_sorted = sorted(st.session_state.bad_list)
    if bad_sorted:
        st.caption(f"{{len(bad_sorted)}} crop(s) marked for exclusion.")
        for name in bad_sorted:
            st.text(name)
    else:
        st.caption("None yet. Remove crops from the Images tab to add them here.")

    if st.button("Save excluded list to txt", key="save_bad_crops_sidebar"):
        out_path = dataLoc.parent / "excluded_crops.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\\n".join(sorted(st.session_state.bad_list)))
        st.success(f"Saved {{len(st.session_state.bad_list)}} name(s) to {{out_path}}")

    

    
    """

    fileName = streamlitAppFolder.joinpath("segmentationApp.py")
    with open(fileName, "w") as f:
        f.write(script)

    fileLocation = streamlitAppFolder.joinpath("segmentationApp.py")
    subprocess.run(["streamlit", "run", fileLocation])




