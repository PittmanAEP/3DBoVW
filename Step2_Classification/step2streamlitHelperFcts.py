

def writeStreamlitAppSparseClassification(streamlitAppFolder, finalMetaDataPath, parameterFileName, userInputList):
    import subprocess
    import json
    print("Writing Streamlit Script Now...")
    zDriveLoc = userInputList.zDriveSaveFolder
    finalMetaDataPathZDrive = zDriveLoc / finalMetaDataPath.parent /  finalMetaDataPath.name
    parameterFileNameZDrive = zDriveLoc / finalMetaDataPath.parent /  parameterFileName.name
    class_names_json = json.dumps(userInputList.classNames)

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
import re
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

canonical_conditions = {class_names_json}
segmentation_mask_suffix = {json.dumps(userInputList.segmentationMaskString)}

#---- Define Functions --- ###
def normalize_key(text: str) -> str:
    '''Lowercase and remove non-alphanumeric characters for flexible matching.'''
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def tokenize_condition(condition: str) -> list[str]:
    '''
    Split a condition like 'Par3OE_prenetrin' into robust match tokens.
    ['par3oe', 'prenetrin']
    '''
    return [normalize_key(tok) for tok in re.split(r"[_\s]+", str(condition)) if tok]


def build_condition_metadata(canonical_conditions: list[str]) -> list[dict]:
    '''
    Store canonical condition names plus normalized tokens.
    Sort most specific first so longer/more detailed names win.
    '''
    metadata = []
    for cond in canonical_conditions:
        metadata.append({{
            "name": cond,
            "tokens": tokenize_condition(cond),
            "norm": normalize_key(cond)
        }})

    metadata.sort(key=lambda d: (-len(d["tokens"]), -len(d["norm"])))
    return metadata


def infer_condition_from_image_name(image_name: str, condition_metadata: list[dict]) -> str:
    '''
    Match an image name to one canonical condition.
    Returns 'Unmatched' if nothing fits.
    '''
    image_norm = normalize_key(image_name)

    matches = []
    for item in condition_metadata:
        if all(tok in image_norm for tok in item["tokens"]):
            matches.append(item["name"])

    if not matches:
        return "Unmatched"

    # because metadata is sorted most specific first, take first match
    return matches[0]


def build_condition_color_map(canonical_conditions: list[str]) -> dict:
    '''
    Create a stable color map for however many conditions the user provides.
    '''
    palette = (
        px.colors.qualitative.Safe
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Dark24
    )

    color_map = {{
        cond: palette[i % len(palette)]
        for i, cond in enumerate(canonical_conditions)
    }}
    color_map["Unmatched"] = "#BDBDBD"
    return color_map


def add_condition_and_color_columns(
    df: pd.DataFrame,
    image_col: str,
    condition_metadata: list[dict],
    color_map: dict
) -> pd.DataFrame:
    '''
    Add:
      - condition
      - condition_color
    by matching image names to canonical conditions.
    '''
    df = df.copy()
    df["condition"] = df[image_col].astype(str).apply(
        lambda x: infer_condition_from_image_name(x, condition_metadata)
    )
    df["condition_color"] = df["condition"].map(color_map).fillna("#BDBDBD")
    return df


def get_condition_columns(df: pd.DataFrame, canonical_conditions: list[str]) -> tuple[list[str], dict]:
    '''
    Find condition columns in a dataframe by normalized string match.
    Returns:
      matched_cols: original dataframe column names that matched
      rename_map: original_name -> canonical_name
    '''
    canonical_lookup = {{normalize_key(cond): cond for cond in canonical_conditions}}

    matched_cols = []
    rename_map = {{}}

    for col in df.columns:
        norm_col = normalize_key(col)
        if norm_col in canonical_lookup:
            matched_cols.append(col)
            rename_map[col] = canonical_lookup[norm_col]

    return matched_cols, rename_map


def patch_bbox(loc_zyx, size, imgBounds):
    z, y, x = loc_zyx
    r = size // 2
    y0, y1 = int(max(0,y-r)), int(min(y+r, imgBounds[1]))
    x0, x1 = int(max(0,x-r)), int(min(x+r, imgBounds[2]))
    z0, z1 = int(max(0,z-r)), int(min(z+r, imgBounds[0]))

    return (z0, z1, y0, y1, x0, x1)



def expand_cluster_rows(df: pd.DataFrame, canonical_conditions: list[str]) -> pd.DataFrame:
    expanded_rows = []

    matched_cols, rename_map = get_condition_columns(df, canonical_conditions)

    if not matched_cols:
        raise ValueError(
            f"No condition columns matched. Expected something like: {{canonical_conditions}}"
        )

    df = df.copy().rename(columns=rename_map)

    canonical_matched_cols = [rename_map[col] for col in matched_cols]
    df[canonical_matched_cols] = df[canonical_matched_cols].fillna(0).astype(int)

    for _, row in df.iterrows():
        for condition_col in canonical_matched_cols:
            count = row[condition_col]
            for _ in range(count):
                expanded_rows.append({{
                    "num_clusters": row["num_clusters"],
                    "cluster_id": row["cluster_id"],
                    "purity": row["purity"],
                    "condition": condition_col,
                    "image_list": row["image_list"]
                }})

    return pd.DataFrame(expanded_rows)


def plot_expanded_cluster_composition(df_expanded, k_value, plot_title, color_map, canonical_conditions):
    df_k = df_expanded[df_expanded['num_clusters'] == k_value].copy()
    cluster_ids = sorted(df_k['cluster_id'].unique())

    fig = go.Figure()

    jitter_strength = 0.1
    np.random.seed(0)
    df_k['x_jitter'] = df_k['cluster_id'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df_k))
    df_k['y_jitter'] = np.random.uniform(-jitter_strength, jitter_strength, size=len(df_k))

    # preserve user-defined condition order
    condition_labels = [c for c in canonical_conditions if c in df_k['condition'].unique()]
    if "Unmatched" in df_k['condition'].unique():
        condition_labels.append("Unmatched")

    for condition in condition_labels:
        df_cond = df_k[df_k['condition'] == condition]
        fig.add_trace(go.Scatter(
            x=df_cond['x_jitter'],
            y=df_cond['y_jitter'],
            mode='markers',
            name=condition,
            marker=dict(
                color=color_map.get(condition, "#BDBDBD"),
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
            zeroline=False
        ),
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
    fullMaskCropName = cropsLoc / f"{{selected_image}}_cell_crop_{{cell_id}}_{{segmentation_mask_suffix}}.tif"
    
    if fullImgCropName.exists():
        overlayCropMask = st.checkbox("Show segmentation mask", value=True)
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
umapScanDF = pd.read_excel(umapScanPath)

condition_metadata = build_condition_metadata(canonical_conditions)
condition_color_map = build_condition_color_map(canonical_conditions)

# Add condition + color columns by matching image_name
dataDF = add_condition_and_color_columns(
    dataDF,
    image_col="image_name",
    condition_metadata=condition_metadata,
    color_map=condition_color_map
)

umapScanDF = add_condition_and_color_columns(
    umapScanDF,
    image_col="image_name",
    condition_metadata=condition_metadata,
    color_map=condition_color_map
)


allImgDictionary = joblib.load(wholeDictionaryPath)
allImgDictionary = normalizeSparseCodes(allImgDictionary)


# ---- Handle click and session state ----
if "selected_image" not in st.session_state or "selected_mask" not in st.session_state or "selected_cellposemask" not in st.session_state:
    # Default to first image in your dataframe if none selected
    st.session_state.selected_image = dataDF.iloc[0]['image_name'] # + ".tif"
    st.session_state.selected_cellposemask = dataDF.iloc[0]['image_name'].rsplit("_",1)[0]+"_" + segmentation_mask_suffix + ".tif"
    st.session_state.selected_keypoints = dataDF.iloc[0]['image_name'].rsplit("_",1)[0]+"_keypointsOutlineRGB.tif"

if "apply_filter_occ" not in st.session_state:
    st.session_state.apply_filter_occ = False
if "reset_filter_occ" not in st.session_state:
    st.session_state.reset_filter_occ = False
if "apply_filter_freq" not in st.session_state:
    st.session_state.apply_filter_freq = False
if "reset_filter_freq" not in st.session_state:
    st.session_state.reset_filter_freq = False


#---- Set up tabs -------
tab1_umap, tabClusterSum, tab4_patches, tab_LR_info, tab_help = st.tabs(
    ["UMAP Results", "Cluster Summary", "Image Patches", "Logistic Regression", "How to Use"])

with tab1_umap: 
    # ---- Scatter plot with click support ----
    st.header("Unsupervised classification results", divider="gray")
    st.subheader("2D UMAP results")
    neighbor_numbers = sorted(umapScanDF['neighborNumber'].unique())    
    default_neighbor = neighbor_numbers.index(4)

    # Generate dropdown options: any column that does NOT contain 'word'
    color_columns = [col for col in umapScanDF.columns if "word" not in col.lower()]
    default_color = "condition" if "condition" in color_columns else color_columns[0]
    default_shape = "condition" if "condition" in color_columns else color_columns[0]

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
    scatter_kwargs = dict(
    data_frame=filtered_df,
    x='umapx',
    y='umapy',
    size=selected_size,
    hover_data="image_name",
    custom_data="image_name",
    symbol=selected_shape,
    color=selected_color_col,
    size_max=10
    )

    if selected_color_col == "condition":
        scatter_kwargs["color_discrete_map"] = condition_color_map
        scatter_kwargs["category_orders"] = {{"condition": canonical_conditions + ["Unmatched"]}}

    figUmap = px.scatter(**scatter_kwargs)

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
        st.session_state.selected_cellposemask = image_name.rsplit("_",1)[0]+"_" + segmentation_mask_suffix + ".tif"
        st.session_state.selected_keypoints = image_name.rsplit("_",1)[0]+"_keypointsOutlineRGB.tif"



with tabClusterSum:    
    clusterSummaryDF = pd.read_excel(clusterSummaryPath)
    st.header("Cluster Summary", divider="gray")

    st.info(
        '''
        **How to interpret this panel**

        Cluster membership is determined by running **k-means on the image embedding vectors in their original feature space**,
        not on the 2D UMAP coordinates. For each tested cluster number, this panel shows which images were assigned to each cluster
        and how the experimental conditions are represented within those clusters.
        '''
    )

    st.caption(
        "The scattered points are used only to visually separate overlapping entries within each cluster. "
        "Their vertical position and horizontal jitter do not represent meaningful spatial coordinates."
    )

    for clusterNumber in range(2, 6):
        with st.container(horizontal_alignment="center", border=True):
            st.subheader(f"Cluster Size: {{clusterNumber}}")
            clusterSummaryDFExpand = expand_cluster_rows(clusterSummaryDF, canonical_conditions)
            figCluster = plot_expanded_cluster_composition(
                                            clusterSummaryDFExpand,
                                            clusterNumber,
                                            "",
                                            condition_color_map,
                                            canonical_conditions)
            event_data_cluster = st.plotly_chart(figCluster, on_select="rerun", selection_mode="points")
            # Filter the summary dataframe for k=2
            cluster_info = clusterSummaryDF[clusterSummaryDF['num_clusters'] == clusterNumber]
            # Drop 'image_list' column before display
            cluster_info_display = cluster_info.drop(columns=['image_list'])
            # Show the result
            st.dataframe(cluster_info_display)

            if event_data_cluster and event_data_cluster.get("selection", {{}}).get("points"):
                clicked_point = event_data_cluster["selection"]["points"][0]
                image_name = clicked_point["customdata"][0]
                st.write(f"🖼️ Cells in this cluster are: `{{image_name}}`")



with tab4_attn:
    st.header("Attention Maps", divider="gray")
    listOfWholeImgs = dataDF['image_name'].unique()
    tmpImgName = listOfWholeImgs[st.session_state.img_index]
    tmpAttnName = tmpImgName.replace(".tif", "") + "_attn_"
    tmpAttnPathOptions = list(attentionMapFolder.glob(f"{{tmpAttnName}}*.tif"))
    tmpAttnPath = next(
        (p for p in tmpAttnPathOptions if "raw" not in p.name.lower()), None)
    tmpImgPath = imageBasePath.joinpath(tmpImgName)
    tmpImg = tiff.imread(tmpImgPath)
    tmpAttn = tiff.imread(tmpAttnPath)
    tmpImgShape = tmpImg.shape
    show_attnMap = st.checkbox("Show attention map", value=True)
    colZ, colA = st.columns([0.5,0.5])
    with colZ:
        tmpZValue = st.slider("Z-slice", 0, tmpImgShape[0]-1, round(tmpImgShape[0]/2))
    with colA:
        alpha = st.slider("Mask transparency", 0.0, 1.0, 0.5)
    fig, ax = plt.subplots(figsize=(6, 6))   
    ax.imshow(tmpImg[tmpZValue], cmap='gray') 
    st.write(f"Image: {{tmpImgName}}")
    if show_attnMap:
        attnMasked = np.ma.masked_where(tmpAttn[tmpZValue] == 0, tmpAttn[tmpZValue])
        im_attn = ax.imshow(attnMasked, cmap="coolwarm", alpha=alpha, vmin=-1, vmax=1)
        cbar = fig.colorbar(im_attn, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention value")
    st.pyplot(fig)

    st.markdown("### 📂 Image Selection")

    name_to_img_index = {{str(name): i for i, name in enumerate(listOfWholeImgs)}}
    img_condition_df = (
        dataDF[["image_name", "condition"]]
        .drop_duplicates(subset=["image_name"])
        .sort_values("image_name")
    )

    sel_cols = st.columns(len(canonical_conditions))
    for col_idx, cond_label in enumerate(canonical_conditions):
        with sel_cols[col_idx]:
            st.markdown(f"**{{cond_label}}**")
            cond_names = img_condition_df.loc[
                img_condition_df["condition"] == cond_label, "image_name"
            ]
            for img_name in cond_names:
                i = name_to_img_index[str(img_name)]
                is_current = i == st.session_state.img_index
                label = f"**{{img_name}}**" if is_current else str(img_name)
                if st.button(label, key=f"img_btn_attn_{{i}}"):
                    st.session_state.img_index = i
                    st.rerun()

    unmatched_names = img_condition_df.loc[
        img_condition_df["condition"] == "Unmatched", "image_name"
    ]
    if len(unmatched_names) > 0:
        st.markdown("**Unmatched**")
        for img_name in unmatched_names:
            i = name_to_img_index[str(img_name)]
            is_current = i == st.session_state.img_index
            label = f"**{{img_name}}**" if is_current else str(img_name)
            if st.button(label, key=f"img_btn_attn_{{i}}"):
                st.session_state.img_index = i
                st.rerun()


    
with tab_LR_info:
    confusionMatrixPath = lrEvalPath / "confusion_matrix_oof.png"
    rocCurvePath = lrEvalPath / "roc_curve_oof.png"
    jsonInfoPath = lrEvalPath / "training_report.json"
    classFilePath = lrEvalPath / "classification_report_oof.csv"
    with open(jsonInfoPath, "r") as f:
        jsondata = json.load(f)

    st.header("Reports on Logistic Regression Model ")

    st.image(confusionMatrixPath)
    if rocCurvePath.exists():
        st.image(rocCurvePath)
    else:
        st.write(f"ROC curve not found, I looked here: {{rocCurvePath}}")

    st.json(jsondata)

    if classFilePath.exists():
        report_df = pd.read_csv(classFilePath, index_col=0)
        st.dataframe(report_df)
    else:
        st.write(f"Classification report not found, I looked here: {{classFilePath}}")

with tab_help:
    st.header("How to Use This App", divider="gray")
    st.caption("A quick guide to navigating the 3D BoVW results viewer.")

    st.info(
        '''
        This app is designed to help you explore the outputs of the 3D BoVW pipeline.
        You can move from global clustering patterns, to cluster composition, to
        patch-level dictionary interpretation, and finally to logistic regression results.
        '''
    )

    st.subheader("Recommended workflow")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '''
            **1. Start with UMAP Results**  
            View how images separate in feature space and color points by metadata or condition.

            **2. Click a point of interest**  
            Selecting a point updates the image shown in the sidebar.

            **3. Inspect the sidebar viewer**  
            Scroll through z-slices and optionally overlay masks or saved keypoints.
            '''
        )

    with col2:
        st.markdown(
            '''
            **4. Open Cluster Summary**  
            See how conditions are distributed across clusters and inspect cluster purity.

            **5. Open Image Patches**  
            Investigate what a dictionary word is capturing across the dataset.

            **6. Review Logistic Regression**  
            Check confusion matrices, ROC curves, and classification reports.
            '''
        )

    st.subheader("What each tab does")

    with st.expander("UMAP Results", expanded=True):
        st.markdown(
            '''
            The **UMAP Results** tab shows a 2D embedding of the image-level feature vectors.

            Use this tab to:
            - compare how images group in feature space,
            - change the displayed neighbor number,
            - color, shape, or size points by available metadata,
            - click a point to load that image into the sidebar viewer.
            '''
        )

    with st.expander("Cluster Summary"):
        st.markdown(
            '''
            The **Cluster Summary** tab summarizes how images from each condition are distributed across k-means clusters computed from the full embedding vectors. This view is intended to help you compare cluster composition across different cluster numbers and identify whether clusters are condition-enriched or mixed.

            Use this tab to:

            - compare cluster composition across clustering solutions,
            - inspect cluster purity,
            - click plotted entries to view the images associated with a selected cluster.
            '''
        )

    with st.expander("Attention Maps"):
        st.markdown(
            '''
            The **Attention Maps** tab shows the attention map for the currently selected image.

            Use this tab to:
            - choose image to display
            - choose z-slice to display
            - choose transparency of the attention map
            - show or hide the attention map
            '''

    with st.expander("Logistic Regression"):
        st.markdown(
            '''
            The **Logistic Regression** tab displays downstream classifier outputs.

            This includes:
            - confusion matrix plots,
            - ROC curves when available,
            - JSON training summary,
            - text classification report.
            '''
        )

    st.subheader("Sidebar viewer")
    st.markdown(
        '''
        The sidebar always shows the **currently selected image**.  
        From there you can:
        - scroll through z-slices,
        - overlay the segmentation mask,
        - optionally show saved keypoints,
        - review the saved run parameters.
        '''
    )

    st.subheader("Conditions")
    st.markdown(
        '''
        The condition list shown in this app is written automatically when the pipeline
        generates the app. Image names are matched against those conditions so that
        plots can be colored consistently.
        '''
    )

    st.code("\\n".join(canonical_conditions), language="text")

    st.subheader("Helpful notes")
    st.warning(
        '''
        - To inspect a specific image, click it in **UMAP Results** first.
        - If an image name does not match any condition, it may be labeled **Unmatched**.
        - The **Image Patches** tab is the best place to interpret what individual words represent.
        '''
    )

# ---- Sidebar image viewer ----
with st.sidebar:
    st.header("Images and parameter info")
    selected_image_path = imageBasePath.joinpath(st.session_state.selected_image) 
    selected_cellposemask_path = imageBasePath.joinpath(st.session_state.selected_cellposemask)
    selected_keypoints_path = imageBasePath.joinpath(st.session_state.selected_keypoints)
    
    if selected_image_path.exists():
        imageVolume = tiff.imread(selected_image_path)
        if selected_cellposemask_path.exists():
            cellposeMaskVolume = tiff.imread(selected_cellposemask_path)
        else:
            cellposeMaskVolume = np.zeros_like(imageVolume)

        if selected_keypoints_path.exists():
            keypointsVolume = tiff.imread(selected_keypoints_path)
        
        
        show_cellposemask = st.checkbox("Show segmentation mask overlay", value=True)
        show_keypoints = st.checkbox("Show keypoints (if saved)?", value = False)
        colZ, colA = st.columns([0.5,0.5])
        with colZ:
            zIndex = st.slider("Z-slice", 0, imageVolume.shape[0]-1, round(imageVolume.shape[0]/2))
        with colA:
            alpha = st.slider("Mask transparency", 0.0, 1.0, 0.5)

        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(imageVolume[zIndex], cmap='gray')
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





def runStreamlitApp(streamlitAppFolder):
    import subprocess

    fileLocation = streamlitAppFolder.joinpath("classificationApp.py")
    subprocess.run(["streamlit", "run", fileLocation])