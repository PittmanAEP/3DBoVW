

def writeStreamlitAppSparseClassification(streamlitAppFolder, finalMetaDataPath, parameterFileName, zDriveLoc):
    import subprocess

    print("Writing Streamlit Script Now...")
    finalMetaDataPathZDrive = zDriveLoc / finalMetaDataPath.parent /  finalMetaDataPath.name
    parameterFileNameZDrive = zDriveLoc / finalMetaDataPath.parent /  parameterFileName.name


    script = f"""


import json
from matplotlib.patches import Rectangle
import streamlit as st
import pandas as pd
import plotly.express as px
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates
import heapq
import joblib

from PIL import Image

# ---- Setup paths ----
dataLoc = Path("{finalMetaDataPathZDrive}")
chToAnalyzeStr = dataLoc.parent.stem.rsplit("_ch")[1][0]
txtFile = Path("{parameterFileNameZDrive}")
imageBasePath = list(dataLoc.parent.glob("ch*Crops"))[0]
dataFrameFolder = dataLoc.parent / "dataframes"
savePointsFolder = dataLoc.parent / "savepoints"
umapScanPath = dataFrameFolder / "umap_neighbor_scan.xlsx"
clusterSummaryPath = dataFrameFolder / "cluster_summary_normalizedFrequencyVector.xlsx"
wholeDictionaryPath = savePointsFolder / "allImagesCellFeatureDict.joblib"
lrEvalPath = dataLoc.parent / "logreg_eval"
outputPath = dataLoc.parent / "outputGraphs"


#---- Define Functions --- ###

def patch_bbox(loc_zyx, size, imgBounds):
    z, y, x = loc_zyx
    r = size // 2
    y0, y1 = int(max(0,y-r)), int(min(y+r, imgBounds[1]))
    x0, x1 = int(max(0,x-r)), int(min(x+r, imgBounds[2]))
    z0, z1 = int(max(0,z-r)), int(min(z+r, imgBounds[0]))

    return (z0, z1, y0, y1, x0, x1)



def get_condition_columns(df: pd.DataFrame) -> list[str]:
    # canonical condition names (lowercase for case-insensitive match)
    canonical_conditions = [
        "control_prenetrin", "control_postnetrin", "par3oe_prenetrin", "par3oe_postnetrin"
    ]
    canon_set = set(canonical_conditions)

    # match DataFrame columns case-insensitively
    matched = []
    for col in df.columns:
        print(col)
        if col.lower() in canon_set:
            matched.append(col)
    print(matched)
    return matched


def expand_cluster_rows(df):
    expanded_rows = []

    # Dynamically detect condition columns: from column index 4 up to 'image_list' column (exclusive)
    # image_list_idx = df.columns.get_loc("image_list")
    condition_cols = get_condition_columns(df)
    df[condition_cols] = df[condition_cols].fillna(0).astype(int)
    print(f"condition rows were {{condition_cols}}")
    for _, row in df.iterrows():
        for condition_col in condition_cols:
            count = row[condition_col]
            print(f"count was {{count}} of type {{type(count)}}")
            for _ in range(count):
                expanded_rows.append({{
                    'num_clusters': row['num_clusters'],
                    'cluster_id': row['cluster_id'],
                    'purity': row['purity'],
                    'condition': condition_col,
                    'image_list': row['image_list']
                }})
    
    return pd.DataFrame(expanded_rows)


def plot_expanded_cluster_composition(df_expanded, k_value, plot_title):
    df_k = df_expanded[df_expanded['num_clusters'] == k_value].copy()
    cluster_ids = sorted(df_k['cluster_id'].unique())

    fig = go.Figure()

    # Add jittered scatter points per image
    jitter_strength = 0.1
    np.random.seed(0)
    df_k['x_jitter'] = df_k['cluster_id'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df_k))
    df_k['y_jitter'] = np.random.uniform(-jitter_strength, jitter_strength, size=len(df_k))  # optional: fixed for horizontal spread
    
    condition_labels = df_k['condition'].unique()
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'teal']
    color_map = {{cond: default_colors[i % len(default_colors)] for i, cond in enumerate(condition_labels)}}


    for condition in condition_labels:
        df_cond = df_k[df_k['condition'] == condition]
        fig.add_trace(go.Scatter(
            x=df_cond['x_jitter'],
            y=df_cond['y_jitter'],
            mode='markers',
            name=condition,
            marker=dict(
                color=color_map[condition],
                size=df_cond['purity'] * 10,
                opacity=0.7,
                line=dict(width=0.5, color='black')
            ),
            hovertext=[
                f"Cluster: {{row['cluster_id']}}<br>Purity: {{row['purity']:.2f}}<br>Condition: {{row['condition']}}"
                for _, row in df_cond.iterrows()
            ],
            hoverinfo="text",
            customdata=df_cond[['image_list']]
        ))

    fig.update_layout(
        title=plot_title,
        yaxis=dict(
                title="Condition",
                showticklabels=False,
                showgrid=False,
                zeroline=False),
        xaxis=dict(title="Cluster ID", tickmode='linear'),
        plot_bgcolor='white',
        margin=dict(l=60, r=20, t=40, b=60),
        height=300
    )

    return fig

def normalize_contrast(image, min_val, max_val):
    clipped = np.clip(image, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val + 1e-8)  # avoid division by zero

def create_overlayed_pil(image, mask=None, show_mask=True, alpha_val=128):
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    min_val, max_val = np.percentile(image, (2, 98))
    norm_img = normalize_contrast(image, min_val, max_val)
    display_img = (norm_img * 255).astype(np.uint8)
    display_pil = Image.fromarray(display_img).convert("RGBA")

    if show_mask and mask is not None:
        unique_ids = np.unique(mask)
        colormap = plt.get_cmap("nipy_spectral")

        # Normalize IDs to 0–1 for colormap
        norm_mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # Generate RGBA mask using colormap
        rgba_mask = (colormap(norm_mask) * 255).astype(np.uint8)
        rgba_mask[..., 3] = (mask > 0) * alpha_val  # Set alpha only for non-zero regions
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

    fullImgCropName = cropsLoc / f"{{selected_image}}_cell_crop_{{cell_id}}.tif"
    fullMaskCropName = cropsLoc / f"{{selected_image}}_cell_crop_{{cell_id}}_cellposeMask.tif"
    
    if fullImgCropName.exists():
        overlayCropMask = st.checkbox("Show cellpose mask", value=True)
        tmpImgVol = tiff.imread(fullImgCropName)
        tmpMaskVol = tiff.imread(fullMaskCropName)
        zIndex = st.slider("Z-slice to display", 0, tmpImgVol.shape[0]-1, round(tmpImgVol.shape[0]//2))
        alpha = st.slider("Mask transparency", 0.0, 1.0, 0.5)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(tmpImgVol[zIndex], cmap='gray')
        if overlayCropMask:
            cellposeMasked = np.ma.masked_where(tmpMaskVol[zIndex] == 0, tmpMaskVol[zIndex])
            ax.imshow(cellposeMasked, cmap='summer', alpha=alpha)
        ax.set_title(f"Image {{selected_image}}_cell_crop_{{cell_id}}.tif")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("That cell wasn't included in the final analysis")


def normalizeSparseCodes(dictionary):

    for imageName, imageData in dictionary.items():
        sparseCodes = imageData["sparseCodes"]
        totalCodes = np.sum(sparseCodes)
        imageData["normSparseCodes"] = sparseCodes / totalCodes

    return dictionary



def top_patches_for_word_global(allImgDictionary, word_idx: int, top_k: int = 25, per_image_cap: int | None = 5):
    
    # Find the top patches (across ALL images) for a given word (0-based index).
    # Returns a list of dicts with: image_key, patch_index, score, location, patch_size.

    # - allImgDictionary[image_key]['sparseCodes'] : (n_patches, n_words) ndarray
    # - allImgDictionary[image_key]['location']    : list[(z,y,x)]
    # - allImgDictionary[image_key]['patch_size']  : list[int] (per patch)
    # - per_image_cap limits how many patches we take from each image (for speed).
    
    heap = []  # min-heap of (score, image_key, patch_idx, loc, size)

    for image_key, imageData in allImgDictionary.items():
        codes = imageData.get("normSparseCodes")
        codes = np.asarray(codes)

        scores = codes[:, word_idx]  # shape (n_patches,)
        n_patches = scores.shape[0]
        k_local = min(per_image_cap or top_k, n_patches)

        # top k for this image
        idx_local = np.argpartition(scores, -k_local)[-k_local:]
        idx_local = idx_local[np.argsort(scores[idx_local])[::-1]]

        locs  = imageData.get("location")
        sizes = imageData.get("patch_size")

        for i in idx_local:
            s = float(scores[i])
            loc = tuple(locs[i])
            size = sizes[i]

            item = (s, image_key, int(i), loc, size)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, item)

    # sort descending by score
    heap.sort(key=lambda x: x[0], reverse=True)
    return [
        {{"score": float(s), "image_key": img, "patch_index": idx, "location": loc, "patch_size": size}}
        for (s, img, idx, loc, size) in heap
    ]


def summarize_kde_from_df(df: pd.DataFrame):
    x = df["x"].to_numpy()
    ctrl = df["density_control"].to_numpy()
    lof  = df["density_lof"].to_numpy()

    # normalize so they integrate to 1
    ctrl_area = np.trapz(ctrl, x) + 1e-12
    lof_area  = np.trapz(lof, x) + 1e-12
    ctrl_pdf = ctrl / ctrl_area
    lof_pdf  = lof  / lof_area

    def stats(pdf):
        mean = np.trapz(x * pdf, x)
        var  = np.trapz((x - mean)**2 * pdf, x)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1] + 1e-12
        median = np.interp(0.5, cdf, x)
        mode = x[np.argmax(pdf)]
        q10 = np.interp(0.10, cdf, x)
        q90 = np.interp(0.90, cdf, x)
        return dict(mean=mean, var=var, median=median, mode=mode, q10=q10, q90=q90)

    return stats(ctrl_pdf), stats(lof_pdf)




# ---- Load Data ----
dataDF = pd.read_excel(dataLoc)
for col in dataDF.columns:
    if dataDF[col].isin(["CON", "LOF", "control", "NIPBLOF", "NIPBLLOF"]).all():
        dataDF.rename(columns={{col: "condition"}}, inplace=True)
        break

umapScanDF = pd.read_excel(umapScanPath)

dataDF["color_group"] = dataDF["condition"].apply(
    lambda x: "lof" if "lof" in str(x).lower() else "control")

allImgDictionary = joblib.load(wholeDictionaryPath)
allImgDictionary = normalizeSparseCodes(allImgDictionary)


# ---- Handle click and session state ----
if "selected_image" not in st.session_state or "selected_mask" not in st.session_state or "selected_segmentation" not in st.session_state or "selected_cellposemask" not in st.session_state:
    # Default to first image in your dataframe if none selected
    st.session_state.selected_image = dataDF.iloc[0]['image_name'] # + ".tif"
    st.session_state.selected_mask = dataDF.iloc[0]['image_name'].rsplit("_",1)[0]+"_nnunetmask.tif"
    st.session_state.selected_segmentation = dataDF.iloc[0]['image_name'].rsplit("_",1)[0]+"_ch1signalSeg.tif"
    st.session_state.selected_cellposemask = dataDF.iloc[0]['image_name'].rsplit("_",1)[0]+"_CellposeMask.tif"
    st.session_state.selected_keypoints = dataDF.iloc[0]['image_name'].rsplit("_",1)[0]+"_keypointsOutlineRGB.tif"

if "apply_filter_occ" not in st.session_state:
    st.session_state.apply_filter_occ = False
if "reset_filter_occ" not in st.session_state:
    st.session_state.reset_filter_occ = False
if "apply_filter_freq" not in st.session_state:
    st.session_state.apply_filter_freq = False
if "reset_filter_freq" not in st.session_state:
    st.session_state.reset_filter_freq = False
if "ai_explanations" not in st.session_state:
    st.session_state.ai_explanations = {{}}  # key: feature idx, value: text



#---- Set up tabs -------
tab1_umap, tabClusterSum, tab4_patches, tab_LR_info  = st.tabs(["UMAP/PCA Results","Cluster Summary", "Image Patches", "Logistic Regression"])

with tab1_umap: 
    ##--- Load Data ---

    # ---- Scatter plot with click support ----
    st.header("Unsupervised classification results", divider="gray")
    st.subheader("2D UMAP results")
    neighbor_numbers = sorted(umapScanDF['neighborNumber'].unique())    
    default_neighbor = neighbor_numbers.index(4)

    # Generate dropdown options: any column that does NOT contain 'word'
    color_columns = [col for col in umapScanDF.columns if "word" not in col.lower()]
    # color_columns = [col for col in dataDF.columns if "word" not in col.lower()]
    default_color = "condition" if "condition" in color_columns else color_columns[0]
    default_shape = "CGN" if "CGN" in color_columns else color_columns[0]

    colN, colC, colShape, colSize = st.columns([1,1,1,1])
    with colN:
        selected_neighbor = st.selectbox("UMAP Neighbor Number", neighbor_numbers, index=default_neighbor)
    with colC:
        selected_color_col = st.selectbox("Color points by:", color_columns, index=color_columns.index(default_color))
    with colShape:
        selected_shape = st.selectbox("Shape points by:", color_columns, index=color_columns.index(default_shape))
    with colSize:
        selected_size = st.selectbox("Size points by:", color_columns, index=color_columns.index(color_columns[0]))
    # Step 3: Filter dataframe to selected neighbor number
    filtered_df = umapScanDF[umapScanDF['neighborNumber'] == selected_neighbor]
    # filtered_df = dataDF[dataDF['neighborNumber'] == selected_neighbor]
    color_map = {{
        "control": "#FF007F",  # Rose
        "lof": "#008080"       # Teal
        }}
    figUmap = px.scatter(filtered_df, 
                    x='umapx',
                    y='umapy',
                    size = selected_size,
                    hover_data="image_name",
                    custom_data="image_name",
                    symbol=selected_shape,
                    color=selected_color_col, 
                    color_discrete_map=color_map,
                    size_max= 10)
    
    figUmap.update_traces(
        selected=dict(marker=dict(opacity=1, size=10)),     # Keep selected points fully visible
        unselected=dict(marker=dict(opacity=1))             # Keep unselected points fully visible too
    )

    figUmap.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        height=500,
        width=700
    )
    event_dict = st.plotly_chart(figUmap, on_select="rerun", selection_mode="points")
    selected_points = event_dict

    
    # -----Update reactive values -------
    if selected_points and selected_points.get("selection", {{}}).get("points"):
        clicked_point = selected_points["selection"]["points"][0]
        image_name = clicked_point["customdata"][0]  
        st.session_state.selected_image = image_name 
        st.session_state.selected_mask = image_name.rsplit("_",1)[0]+"_nnunetmask.tif"
        st.session_state.selected_cellposemask = image_name.rsplit("_",1)[0]+"_CellposeMask.tif"
        st.session_state.selected_segmentation = image_name.rsplit("_",1)[0]+"_ch1signalSeg.tif"
        st.session_state.selected_keypoints = image_name.rsplit("_",1)[0]+"_keypointsOutlineRGB.tif"



with tabClusterSum:    
    clusterSummaryDF = pd.read_excel(clusterSummaryPath)
    with st.container(horizontal_alignment="center", border=True):
        st.header("Cluster Size: 2")
        clusterSummaryDFExpand = expand_cluster_rows(clusterSummaryDF)
        figCluster2 = plot_expanded_cluster_composition(clusterSummaryDFExpand, 2, "")
        event_data_cluster2 = st.plotly_chart(figCluster2, on_select="rerun", selection_mode="points")
        # Filter the summary dataframe for k=2
        cluster_info = clusterSummaryDF[clusterSummaryDF['num_clusters'] == 2]
        # Drop 'image_list' column before display
        cluster_info_display = cluster_info.drop(columns=['image_list'])
        # Show the result
        st.dataframe(cluster_info_display)

        if event_data_cluster2 and event_data_cluster2.get("selection", {{}}).get("points"):
            clicked_point = event_data_cluster2["selection"]["points"][0]
            image_name = clicked_point["customdata"][0]
            st.write(f"🖼️ Cells in this cluster are: `{{image_name}}`")

    with st.container(horizontal_alignment="center", border=True):
        st.header("Cluster Size: 3")
        clusterSummaryDFExpand = expand_cluster_rows(clusterSummaryDF)
        figCluster3 = plot_expanded_cluster_composition(clusterSummaryDFExpand, 3, "")
        event_data_cluster3 = st.plotly_chart(figCluster3, on_select="rerun", selection_mode="points")
        # Filter the summary dataframe for k=2
        cluster_info = clusterSummaryDF[clusterSummaryDF['num_clusters'] == 3]
        # Drop 'image_list' column before display
        cluster_info_display = cluster_info.drop(columns=['image_list'])
        # Show the result
        st.dataframe(cluster_info_display)
        
        if event_data_cluster3 and event_data_cluster3.get("selection", {{}}).get("points"):
            clicked_point = event_data_cluster3["selection"]["points"][0]
            image_name = clicked_point["customdata"][0]
            st.write(f"🖼️ Cells in this cluster are: `{{image_name}}`")
    
    with st.container(horizontal_alignment="center", border=True):
        selected_point_cluster4 = st.session_state.get('clicked_point_cluster4', None)
        st.header("Cluster Size: 4")
        clusterSummaryDFExpand = expand_cluster_rows(clusterSummaryDF)
        figCluster4 = plot_expanded_cluster_composition(clusterSummaryDFExpand, 4, "")
        event_data_cluster4 = st.plotly_chart(figCluster4, on_select="rerun", selection_mode="points")
        # Filter the summary dataframe for k=2
        cluster_info = clusterSummaryDF[clusterSummaryDF['num_clusters'] == 4]
        # Drop 'image_list' column before display
        cluster_info_display = cluster_info.drop(columns=['image_list'])
        # Show the result
        st.dataframe(cluster_info_display)

        if event_data_cluster4 and event_data_cluster4.get("selection", {{}}).get("points"):
            clicked_point = event_data_cluster4["selection"]["points"][0]
            image_name = clicked_point["customdata"][0]
            st.write(f"🖼️ Cells in this cluster are: `{{image_name}}`")





with tab4_patches:
    
    st.header("Visualize patches and associated sparse codes")
    wordColumns = [col for col in dataDF.columns if col.startswith("word_")]
    dictionarySize = len(wordColumns)
    wordOnlyDF = dataDF[wordColumns]
    summedWordValues = wordOnlyDF.sum(axis=0)
    thresholdWords = st.radio("Do you want to only show the top 95 percentile words?", ["yes", "no"])
    if thresholdWords == "yes":
        percentileThreshold = np.percentile(summedWordValues, 95)
        wordsAboveThres = summedWordValues[summedWordValues >= percentileThreshold]
        wordsToUse = [int(word.rsplit("_")[1]) for word in wordsAboveThres.index]
    colSelectWord, colTopK = st.columns([1,1])
    with colSelectWord:
        if thresholdWords == "no":
            selectedWord = st.number_input("Which word do you want to see?", min_value=0, max_value=dictionarySize, value=1, step=1)
        else:
            selectedWord = st.selectbox("Which word do you want to see?", options=wordsToUse)
    with colTopK:
        top_k = st.slider("How many patches to pull up?", min_value=1, max_value=10, value=5, step=1)

    results = top_patches_for_word_global(allImgDictionary, selectedWord, top_k, 5)
    df = pd.DataFrame(results)
    st.dataframe(df, hide_index=True)

    uniqueIDCount = 0 
    for rec in results:
        uniqueIDCount += 1
        uniqueIDWholeImgSlider = f"whole_key_{{uniqueIDCount}}"
        uniqueIDPatchSlides = f"patch_key_{{uniqueIDCount}}"
        uniqueIDCheckbox = f"check_key_{{uniqueIDCount}}"
        uniqueIDPathHisto = f"histo_key_{{uniqueIDCount}}"
        uniqueIDImgHisto = f"imgHisto_key_{{uniqueIDCount}}"
        img_key   = rec["image_key"]
        i         = rec["patch_index"]
        loc_zyx   = rec["location"]
        psize     = rec["patch_size"]
        score     = rec["score"]
        image_entry = allImgDictionary[img_key]
        codes = image_entry["sparseCodes"]  
        normFreqVector = image_entry["normalizedFrequencyVector"]
        normSparseCodes = image_entry["normSparseCodes"]
        tmpImg = tiff.imread(imageBasePath / img_key)
        segImg = tiff.imread(imageBasePath / img_key.replace("ch1", "ch1signalSeg"))
        imgSize = tmpImg.shape

        with st.container(horizontal_alignment = "center", border = True):
            colImg, colPatch = st.columns([1,1])
            z0, z1, y0, y1, x0, x1 = patch_bbox(loc_zyx, psize, imgSize)
            midZ = int(round(z1 - z0) / 2)
            zPatchSize = z1 - z0
            colText, colButton = st.columns([0.75,0.25])
            with colText:
                st.write(img_key)
            with colButton:
                patchSegOverlay = st.checkbox("Show segmentation overlay", value=False, key = uniqueIDCheckbox)
            with colImg:                
                zSliderWholeImg = st.slider("Z-slice, whole img", 0, imgSize[0]-1, midZ+z0, key=uniqueIDWholeImgSlider)
                fig, ax = plt.subplots(figsize=(1,1))
                in_slice = (z0 <= zSliderWholeImg < z1)
                if in_slice:
                    rect = Rectangle(
                            (x0, y0),              # (x, y) origin in image coords
                            x1 - x0,               # width
                            y1 - y0,               # height
                            linewidth=1,
                            edgecolor='yellow',
                            facecolor='none',
                            linestyle='--',
                            alpha=1.0
                        )
                    ax.add_patch(rect)
                ax.set_axis_off()
                ax.imshow(tmpImg[zSliderWholeImg], cmap='gray')
                st.pyplot(fig)

            with colPatch:
                zSliderPatches = st.slider("Z-slice", 0, zPatchSize, midZ, key = uniqueIDPatchSlides)      
                tmpSegCrop = segImg[z0:z1, y0:y1, x0:x1]
                seg_masked_patch = np.ma.masked_where(tmpSegCrop[zSliderPatches] == 0, tmpSegCrop[zSliderPatches])
                tmpCrop1 = tmpImg[z0:z1, y0:y1, x0:x1]
                fig, ax = plt.subplots(figsize=(1,1))
                ax.imshow(tmpCrop1[zSliderPatches], cmap='gray')
                if patchSegOverlay:
                    ax.imshow(seg_masked_patch, cmap='spring', alpha=0.25)
                st.pyplot(fig)
            

            vec = np.asarray(normSparseCodes[i])
            x_idx = np.arange(0,dictionarySize,1)
            y_val = vec
            df_bar = pd.DataFrame({{
                "Word": x_idx,
                "Count": y_val}})
            
            df_bar["Selected"] = np.where(df_bar["Word"] == int(selectedWord),
                              "Selected word", "Other")

            sparseHisto = px.bar(df_bar,
                x = "Word",
                y = "Count",
                color="Selected",                           # color by category
                color_discrete_map={{
                    "Selected word": "gold",
                    "Other": "lightgray"
                }})
            

            imageFreq = np.asarray(normFreqVector)
            x_words = np.arange(0,dictionarySize,1)
            y_val = imageFreq
            df_bar_wholeImg = pd.DataFrame({{
                "Word": x_words,
                "Count": y_val}})
            
            df_bar_wholeImg["Selected"] = np.where(df_bar_wholeImg["Word"] == int(selectedWord),
                              "Selected word", "Other")

            sparseHistoWholeImg = px.bar(df_bar_wholeImg,
                x = "Word",
                y = "Count",
                color="Selected",                           # color by category
                color_discrete_map={{
                    "Selected word": "gold",
                    "Other": "lightgray"
                }})
            
            pathHisto, imgHisto = st.columns([1,1])
            with pathHisto:
                st.plotly_chart(sparseHisto, use_container_width=True, key = uniqueIDPathHisto)
            with imgHisto:
                st.plotly_chart(sparseHistoWholeImg, use_container_width=True, key = uniqueIDImgHisto)
    
with tab_LR_info:
    confusionMatrixPath = lrEvalPath / f"confusion_matrix_ch{{chToAnalyzeStr}}_modeldefault.png"
    rocCurvePath = lrEvalPath / f"roc_curve_ch{{chToAnalyzeStr}}_LR_default.png"
    jsonInfoPath = lrEvalPath / f"metrics_ch{{chToAnalyzeStr}}_lr_default.json"
    with open(jsonInfoPath, "r") as f:
        jsondata = json.load(f)

    st.header("Reports on Logistic Regression Model ")

    st.image(confusionMatrixPath)

    st.image(rocCurvePath)

    st.json(jsondata)



# ---- Sidebar image viewer ----
with st.sidebar:
    st.header("Images and parameter info")
    selected_image_path = imageBasePath.joinpath(st.session_state.selected_image) 
    selected_mask_path = imageBasePath.joinpath(st.session_state.selected_mask)
    selected_cellposemask_path = imageBasePath.joinpath(st.session_state.selected_cellposemask)
    selected_segmentation_path = imageBasePath.joinpath(st.session_state.selected_segmentation)
    selected_keypoints_path = imageBasePath.joinpath(st.session_state.selected_keypoints)
    
    if selected_image_path.exists():
        imageVolume = tiff.imread(selected_image_path)
        if selected_mask_path.exists():
            maskVolume = tiff.imread(selected_mask_path)
        else:
            maskVolume = np.zeros_like(imageVolume)
        cellposeMaskVolume = tiff.imread(selected_cellposemask_path)
        segmentationVolume = tiff.imread(selected_segmentation_path)

        if selected_keypoints_path.exists():
            keypointsVolume = tiff.imread(selected_keypoints_path)
        
        
        show_nnmask = st.checkbox("Show nnunet mask overlay", value=False)
        show_cellposemask = st.checkbox("Show cellpose mask overlay", value=True)
        show_segmentation = st.checkbox("Show segmentation overlay", value=False)
        show_keypoints = st.checkbox("Show keypoints (if saved)?", value = False)
        colZ, colA = st.columns([0.5,0.5])
        with colZ:
            zIndex = st.slider("Z-slice", 0, imageVolume.shape[0]-1, round(imageVolume.shape[0]/2))
        with colA:
            alpha = st.slider("Mask transparency", 0.0, 1.0, 0.5)

        masked = np.ma.masked_where(maskVolume[zIndex] == 0, maskVolume[zIndex])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(imageVolume[zIndex], cmap='gray')
        if show_nnmask:
            masked = np.ma.masked_where(maskVolume[zIndex] == 0, maskVolume[zIndex])
            ax.imshow(masked, cmap='spring', alpha=alpha)
        if show_segmentation:
            seg_masked = np.ma.masked_where(segmentationVolume[zIndex] == 0, segmentationVolume[zIndex])
            ax.imshow(seg_masked, cmap='spring', alpha=alpha)
        if show_cellposemask:
            cellposemasked = np.ma.masked_where(cellposeMaskVolume[zIndex] == 0, cellposeMaskVolume[zIndex])
            ax.imshow(cellposemasked, cmap='summer', alpha=alpha)
        if show_keypoints:            
            keypointsMasked = np.ma.masked_where(keypointsVolume[zIndex] == 0, keypointsVolume[zIndex])
            ax.imshow(keypointsMasked, cmap='summer', alpha=alpha)

        ax.set_title(f"Overlay Z Slice {{zIndex}}")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning(f"Image not found: {{st.session_state.selected_image}}")


    st.divider()

    
    # Show parameters
    if txtFile.exists():
        with open(txtFile, "r") as file:
            file_contents = file.read()
    else:
        st.write(f"txt file not found, I looked here: {{txtFile}}")

    st.text_area("Parameters from run", file_contents, height=300)

    

    """
    fileName = streamlitAppFolder.joinpath("classificationApp.py")
    with open(fileName, "w") as f:
        f.write(script)

    fileLocation = streamlitAppFolder.joinpath("classificationApp.py")

