


def visualizeAnalyzedLocations(allImagesCellFeatureDict, saveFolder, userInputList):
    from skimage.io import imread, imsave
    import numpy as np
    visualizeAnalyzedLocationsOutlines(allImagesCellFeatureDict, saveFolder, userInputList, thickness=1)
    chCropFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
    for imageName, imageData in allImagesCellFeatureDict.items():
        tmpImgLoc = chCropFolder / imageName
        tmpImg = imread(tmpImgLoc)
        tmpBlankImg = np.zeros_like(tmpImg)
        listOfLocations = imageData["location"]
        listOfPatches = imageData["patch_size"]
        i=0 
        for loc, patch in zip(listOfLocations, listOfPatches):
            i += 1
            z, y, x = loc
            zLow = z - round(patch/2); zHigh = z + round(patch/2)
            yLow = y - round(patch/2); yHigh = y + round(patch/2)
            xLow = x - round(patch/2); xHigh = x + round(patch/2)
            tmpBlankImg[zLow:zHigh, yLow:yHigh, xLow:xHigh] = i
        
        saveName = imageName.replace(".tif", "patches.tif")
        savePath = chCropFolder / saveName
        imsave(savePath, tmpBlankImg)
        del tmpBlankImg, tmpImg

def visualizeAnalyzedLocationsOutlines(allImagesCellFeatureDict, saveFolder, userInputList, thickness=1):
    from skimage.io import imread, imsave
    import numpy as np

    def clip_box(zl, zh, yl, yh, xl, xh, shape):
        Z, Y, X = shape
        zl = max(0, zl); zh = min(Z, zh)
        yl = max(0, yl); yh = min(Y, yh)
        xl = max(0, xl); xh = min(X, xh)
        return zl, zh, yl, yh, xl, xh

    def draw_box_outline(vol, zl, zh, yl, yh, xl, xh, value, t=1):
        # if the box is too thin, just fill it
        if (zh - zl) <= 2*t or (yh - yl) <= 2*t or (xh - xl) <= 2*t:
            vol[zl:zh, yl:yh, xl:xh] = value
            return

        # 6 faces = a "shell" outline
        vol[zl:zl+t, yl:yh, xl:xh] = value          # z-min face
        vol[zh-t:zh, yl:yh, xl:xh] = value          # z-max face
        vol[zl:zh, yl:yl+t, xl:xh] = value          # y-min face
        vol[zl:zh, yh-t:yh, xl:xh] = value          # y-max face
        vol[zl:zh, yl:yh, xl:xl+t] = value          # x-min face
        vol[zl:zh, yl:yh, xh-t:xh] = value          # x-max face

    chCropFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"

    for imageName, imageData in allImagesCellFeatureDict.items():
        tmpImg = imread(chCropFolder / imageName)
        tmpBlankImg = np.zeros_like(tmpImg)

        listOfLocations = imageData["location"]
        listOfPatches   = imageData["patch_size"]

        for i, (loc, patch) in enumerate(zip(listOfLocations, listOfPatches), start=1):
            z, y, x = map(int, loc)
            p = int(patch)

            # make the box exactly p voxels wide on each axis
            half = p // 2
            zLow = z - half
            yLow = y - half
            xLow = x - half
            zHigh = zLow + p
            yHigh = yLow + p
            xHigh = xLow + p

            zLow, zHigh, yLow, yHigh, xLow, xHigh = clip_box(
                zLow, zHigh, yLow, yHigh, xLow, xHigh, tmpBlankImg.shape
            )

            draw_box_outline(tmpBlankImg, zLow, zHigh, yLow, yHigh, xLow, xHigh, value=i, t=thickness)

        saveName = imageName.replace(".tif", "patches_outline.tif")
        imsave(chCropFolder / saveName, tmpBlankImg)




def compute_reconstruction_error(X, dictionary, alpha):
    from sklearn.decomposition import sparse_encode
    import numpy as np

    codes = sparse_encode(X, dictionary, algorithm="lasso_cd", alpha=alpha)
    X_reconstructed = codes @ dictionary
    error = np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))
    return error


def reconstruction_mse_batched(X, C, D, batch=4096, per_feature=True):
    import numpy as np

    """
    Returns mean squared error. If per_feature=True, it returns MSE averaged
    over samples and features (i.e., divide by N*F). Otherwise it’s per-sample MSE (divide by N).
    """
    n, f = X.shape
    err_sum = 0.0
    for s in range(0, n, batch):
        e = min(s + batch, n)
        # Reconstruct this batch: X_hat = C @ D
        # (C[s:e] is (B, K), D is (K, F) → (B, F))
        R = X[s:e] - C[s:e] @ D
        # Sum of squared residuals in the batch
        err_sum += np.einsum('ij,ij->', R, R)
    denom = (n * f) if per_feature else n
    return err_sum / denom

def transform_batched(estimator, X, batch=4096):
    import numpy as np
    n, _ = X.shape
    K = estimator.components_.shape[0]
    C = np.empty((n, K), dtype=X.dtype)   # codes matrix
    for s in range(0, n, batch):
        e = min(s + batch, n)
        # Under the hood: solver = "lasso_cd" with transform_alpha you set above
        C[s:e] = estimator.transform(X[s:e])
    return C

 
def generateSparseDictionary(allImagesCellFeatureDict, userInputList, filePath, outputResults):
    from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode
    from sklearn.model_selection import train_test_split
    import numpy as np
    import joblib
    import time
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("Starting sparse dictionary learning...")
    start_time = time.time()

    # Create/save paths
    codebookFilesLoc = filePath.joinpath("codebookFiles")
    codebookFilesLoc.mkdir(exist_ok=True)
    sparse_dict_selectImg_filename = f"sparse_dict_{userInputList.sparsityAlpha}_{userInputList.lassoAlpha}_{userInputList.dictionarySize}_ch{userInputList.chToClassify}_{userInputList.siftKeypointsLocation}key_{userInputList.siftThreshold}t_{userInputList.siftEdgeR}r_volsplit{userInputList.splitSmallAndLarge}_{userInputList.siftNormalization}norm.joblib"

    sparse_dict_allimg_filename = f"sparse_dict_{userInputList.sparsityAlpha}_{userInputList.lassoAlpha}_{userInputList.dictionarySize}_ch{userInputList.chToClassify}_{userInputList.siftKeypointsLocation}key_{userInputList.siftThreshold}t_{userInputList.siftEdgeR}r_volsplitFalse_{userInputList.siftNormalization}norm.joblib"
    
    sparse_dict_filename = sparse_dict_allimg_filename if userInputList.allImgCodebookFile else sparse_dict_selectImg_filename

    compiledDescriptors = np.concatenate([np.stack(imgData["feature"]) for imgData in allImagesCellFeatureDict.values()], axis=0)
    # Split into train and validation sets
    X_train, X_val = train_test_split(compiledDescriptors, test_size=0.2, random_state=42)

    sparse_dictionary = None
    if userInputList.codebookFile:
        sparse_dict_fileLoc = codebookFilesLoc / sparse_dict_filename
        if sparse_dict_fileLoc.exists():
            print("Sparse dictionary found, loading now...")
            sparse_dictionary = joblib.load(sparse_dict_fileLoc)             
            outputResults.foundAllImgCodebook = True if userInputList.allImgCodebookFile else False
            outputResults.newCodebook = False
            dict_learner = MiniBatchDictionaryLearning(
                n_components=sparse_dictionary.shape[0],
                transform_algorithm="lasso_cd",
                transform_alpha=userInputList.sparsityAlpha,
                n_jobs=-1
            )
            dict_learner.components_ = np.ascontiguousarray(sparse_dictionary)
            
        else:
            sparse_dictionary = None

    if sparse_dictionary is None:
        outputResults.newCodebook = True
        sparse_dict_filename = sparse_dict_selectImg_filename if userInputList.splitSmallAndLarge else sparse_dict_allimg_filename
        # Initialize MiniBatchDictionaryLearning
        dict_learner = MiniBatchDictionaryLearning(
            n_components=userInputList.dictionarySize,  # initial size (e.g., 500-1000)
            alpha=userInputList.sparsityAlpha,  # controls sparsity
            fit_algorithm="lars",
            max_iter=1700,
            batch_size=1000,
            random_state=42,
            verbose=False,
            n_jobs=-1,
            transform_algorithm="lasso_cd",               # keep same family as before
            transform_alpha=userInputList.sparsityAlpha,
        )

        # Fit the sparse dictionary
        sparse_dictionary = np.ascontiguousarray(dict_learner.fit(X_train).components_)
        # Save dictionary
        joblib.dump(sparse_dictionary, codebookFilesLoc / sparse_dict_filename)
        print(f"Sparse dictionary created and saved. Total time: {time.time() - start_time:.2f}s")

    outputResults.codebookFileName = sparse_dict_filename
    # Compute reconstruction error on validation set

    print("calculating reconstruction error now...")
    # C_val = transform_batched(dict_learner, X_val, batch=4096)
    # val_error = reconstruction_mse_batched(X_val, C_val, sparse_dictionary, batch=4096, per_feature=True)
    val_error = -1
    outputResults.reconstructionError = val_error
    print(f"Reconstruction error on validation set: {val_error:.4f}")
    return sparse_dictionary, outputResults



def loadCodebookSavepoint(filePath, userInputList):
    import joblib

    codebookFilesLoc = filePath.joinpath("codebookFiles")
    sparse_dict_allimg_filename = f"sparse_dict_{userInputList.sparsityAlpha}_{userInputList.lassoAlpha}_{userInputList.dictionarySize}_ch{userInputList.chToClassify}_{userInputList.siftKeypointsLocation}key_{userInputList.siftThreshold}t_{userInputList.siftEdgeR}r_volsplitFalse_{userInputList.siftNormalization}norm.joblib"
    sparse_dictionary = joblib.load(codebookFilesLoc / sparse_dict_allimg_filename)  

    return sparse_dictionary






# # --- Lens A: Dictionary-atom similarity → thresholded graph → families ---

# import numpy as np
# import networkx as nx
# from typing import Dict, List, Tuple, Iterable


# # 1) Normalize atoms and compute cosine-similarity (Gram) matrix
# def atom_cosine_similarity(sparse_dictionary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     import numpy as np
#     """
#     sparse_dictionary: (K, m) from MiniBatchDictionaryLearning.components_
#     Returns:
#         D_unit: (K, m) unit-normalized atoms
#         G: (K, K) cosine similarity matrix (diag == 1)
#     """
#     D = sparse_dictionary  # shape (K, m)
#     norms = np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
#     D_unit = D / norms
#     G = D_unit @ D_unit.T
#     # numerical cleanup
#     np.fill_diagonal(G, 1.0)
#     return D_unit, G

# # 2) Simple redundancy stats (optional but handy)
# def redundancy_stats(G: np.ndarray) -> Dict[str, float]:
    
#     """
#     Returns mutual coherence and a few percentiles of off-diagonal |G|.
#     """
#     K = G.shape[0]
#     off = np.abs(G.copy())
#     off[np.eye(K, dtype=bool)] = 0.0
#     mu = off.max()
#     p95 = np.percentile(off, 95)
#     p99 = np.percentile(off, 99)
#     mean_off = off.sum() / (K*(K-1))
#     return {"mutual_coherence": float(mu), "p95": float(p95), "p99": float(p99), "mean_offdiag": float(mean_off)}

# # 3) Build a thresholded atom graph
# def build_atom_graph(G: np.ndarray, tau: float, use_abs: bool = True) -> nx.Graph:
#     """
#     tau: similarity threshold in [0,1]; edges added when (|G_ij| >= tau) and i != j
#     use_abs: if True, threshold on |G|; else on raw G (signed).
#     """
#     K = G.shape[0]
#     M = np.abs(G) if use_abs else G
#     A = (M >= tau).astype(np.uint8)
#     np.fill_diagonal(A, 0)
#     # Construct graph with weights
#     Gx = nx.Graph()
#     Gx.add_nodes_from(range(K))
#     # Only add edges above threshold
#     ii, jj = np.where(A)
#     for i, j in zip(ii.tolist(), jj.tolist()):
#         if i < j:  # undirected, add once
#             Gx.add_edge(i, j, weight=float(G[i, j]))
#     return Gx

# # 4) Find “families” of words (connected components or community detection)
# def word_families_connected_components(Gx: nx.Graph) -> List[List[int]]:
#     """
#     Fast, parameter-free grouping by graph connectivity.
#     Returns a list of components, each a list of word indices.
#     """
#     return [sorted(list(c)) for c in nx.connected_components(Gx)]

# def word_families_modularity(Gx: nx.Graph) -> List[List[int]]:
#     """
#     Greedy modularity communities (considers edge weights if present).
#     Useful when tau is low and the graph is dense.
#     """
#     from networkx.algorithms.community import greedy_modularity_communities
#     comms = greedy_modularity_communities(Gx, weight="weight")
#     return [sorted(list(c)) for c in comms]

# # 5) (Optional) “Top-k per node” graph variant (keeps each node’s k strongest links)
# def build_topk_graph(G: np.ndarray, k: int = 3, use_abs: bool = True) -> nx.Graph:
#     """
#     Keeps, for each node, edges to its top-k similar neighbors. Symmetrize by union.
#     """
#     K = G.shape[0]
#     S = np.abs(G) if use_abs else G
#     np.fill_diagonal(S, -np.inf)  # exclude self
#     Gx = nx.Graph()
#     Gx.add_nodes_from(range(K))
#     for i in range(K):
#         # indices of top-k neighbors by similarity
#         nbrs = np.argpartition(S[i], -k)[-k:]
#         for j in nbrs:
#             if i == j: 
#                 continue
#             w = float(G[i, j])
#             if Gx.has_edge(i, j):
#                 # keep the stronger weight if already present
#                 Gx[i][j]["weight"] = max(Gx[i][j]["weight"], w, key=abs)
#             else:
#                 Gx.add_edge(i, j, weight=w)
#     return Gx

# def build_mutual_topk_graph(G, k=2, use_abs=True, min_abs_weight=None):
#     import numpy as np, networkx as nx
#     K = G.shape[0]
#     S = np.abs(G) if use_abs else G.copy()
#     np.fill_diagonal(S, -np.inf)  # never pick self

#     # each node's top-k neighbor set (by |similarity|)
#     topk = [set(np.argpartition(S[i], -k)[-k:]) for i in range(K)]

#     Gx = nx.Graph()
#     Gx.add_nodes_from(range(K))

#     for i in range(K):
#         for j in topk[i]:
#             if i < j and i in topk[j]:  # keep only mutual neighbors
#                 w = float(G[i, j])      # keep signed weight
#                 if (min_abs_weight is not None) and (abs(w) < min_abs_weight):
#                     continue
#                 Gx.add_edge(i, j, weight=w)
#     return Gx


# # 6) Convenience wrapper: compute G, make graph, return families + stats
# def compute_word_families(
#     sparse_dictionary: np.ndarray,
#     threshold: float = 0.98,
#     method: str = "components",  # "components" or "modularity"
#     topk: int = 3,            # if set, use top-k graph instead of threshold
#     use_abs: bool = True) -> Dict:
#     """
#     Returns:
#       {
#         'G': cosine-similarity (K,K),
#         'graph': nx.Graph,
#         'families': List[List[int]],
#         'stats': redundancy_stats
#       }
#     """
#     _, G = atom_cosine_similarity(sparse_dictionary)
#     stats = redundancy_stats(G)

#     if topk is not None:
#         # Gx = build_topk_graph(G, k=topk, use_abs=use_abs)
#         Gx = build_mutual_topk_graph(G, k=topk, use_abs=True, min_abs_weight=None)
#     else:
#         Gx = build_atom_graph(G, tau=threshold, use_abs=use_abs)

#     if method == "components":
#         fams = word_families_connected_components(Gx)
#     elif method == "modularity":
#         fams = word_families_modularity(Gx)
#     else:
#         raise ValueError("method must be 'components' or 'modularity'.")
    
#     # Gx_tau = build_atom_graph(G, tau=0.98, use_abs=True)
#     # graph_report(G, Gx_tau, "tau=0.98")

#     # Gx_topk = build_topk_graph(G, k=3, use_abs=True)
#     # graph_report(G, Gx_topk, "topk=3")

#     return {"G": G, "graph": Gx, "families": fams, "stats": stats}

# # 7) (Optional) Small helpers to summarize and export
# def families_summary(families: List[List[int]]) -> List[Tuple[int, int]]:
#     """
#     Returns list of (family_id, size), sorted by size desc.
#     """
#     return sorted([(i, len(f)) for i, f in enumerate(families)], key=lambda x: x[1], reverse=True)

# def family_membership_vector(families: List[List[int]], K: int) -> np.ndarray:
#     """
#     Returns an array fam_id_of_word (length K) indicating which family each word belongs to.
#     """
#     fam_of = -np.ones(K, dtype=int)
#     for fid, fam in enumerate(families):
#         for w in fam:
#             fam_of[w] = fid
#     return fam_of

# def graph_report(G, Gx, name):
#     import numpy as np, networkx as nx
#     K = G.shape[0]
#     cc_sizes = sorted((len(c) for c in nx.connected_components(Gx)), reverse=True)
#     degs = np.array([d for _, d in Gx.degree()])
#     print(
#         f"[{name}] nodes={Gx.number_of_nodes()} edges={Gx.number_of_edges()} "
#         f"density={nx.density(Gx):.4f} avgdeg={degs.mean():.2f} "
#         f"maxdeg={degs.max()} components={len(cc_sizes)} top5={cc_sizes[:5]}"
#     )


# def export_graph_for_streamlit(G: np.ndarray,
#                                Gx: "nx.Graph",
#                                fam_id_of_word: np.ndarray,
#                                save_dir,
#                                layout_seed: int = 42,
#                                min_weight: float | None = None,
#                                topk_per_node: int | None = None,
#                                layout_kwargs: dict | None = None):
#     """
#     Save graph data for later plotting in Streamlit.

#     Args
#     ----
#     G : (K,K) cosine-similarity matrix (signed)
#     Gx : networkx.Graph with 'weight' on edges (built from G)
#     fam_id_of_word : (K,) int array mapping word -> family id
#     save_dir : Path-like output folder
#     layout_seed : seed for spring layout (stable positions across runs)
#     min_weight : if set, drop edges with |weight| < min_weight
#     topk_per_node : if set, keep only the strongest |weight| edges per node (union)
#     layout_kwargs : dict forwarded to nx.spring_layout (e.g., {'k':None, 'iterations':50})

#     Outputs
#     -------
#     nodes.csv: word,family_id,degree,x,y
#     edges.csv: word_i,word_j,weight
#     """
#     import numpy as np, pandas as pd, networkx as nx
#     from pathlib import Path

#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     dataFrameFolder = save_dir.joinpath("dataframes")
#     dataFrameFolder.mkdir(parents = True, exist_ok = True)
#     savepointsFolder = save_dir.joinpath("savepoints")
#     savepointsFolder.mkdir(parents = True, exist_ok = True)

#     K = G.shape[0]

#     # --- copy and optionally filter edges ---
#     H = nx.Graph()
#     H.add_nodes_from(Gx.nodes(data=True))
#     # collect candidate edges
#     edges = []
#     for u, v, d in Gx.edges(data=True):
#         w = float(d.get("weight", 0.0))
#         if (min_weight is not None) and (abs(w) < min_weight):
#             continue
#         edges.append((u, v, w))

#     if topk_per_node is not None:
#         # keep strongest |w| per node, symmetrize by union
#         keep = set()
#         by_node = {i: [] for i in range(K)}
#         for u, v, w in edges:
#             by_node[u].append((v, w))
#             by_node[v].append((u, w))
#         for i in range(K):
#             nbrs = sorted(by_node[i], key=lambda t: abs(t[1]), reverse=True)[:topk_per_node]
#             for j, w in nbrs:
#                 keep.add(tuple(sorted((i, j))))
#         edges = [(i, j, dict(weight=float(G[i, j]))) for (i, j) in keep]
#         H.add_edges_from(edges)
#     else:
#         H.add_weighted_edges_from(edges, weight="weight")

#     # --- stable 2D layout (computed once and reused in Streamlit) ---
#     layout_kwargs = layout_kwargs or {}
#     pos = nx.spring_layout(H, seed=layout_seed, weight="weight", **layout_kwargs)

#     # --- nodes table ---
#     degrees = dict(H.degree()) # type: ignore
#     nodes_df = pd.DataFrame({
#         "word": list(H.nodes()),
#         "family_id": [int(fam_id_of_word[n]) for n in H.nodes()],
#         "degree": [int(degrees[n]) for n in H.nodes()],
#         "x": [float(pos[n][0]) for n in H.nodes()],
#         "y": [float(pos[n][1]) for n in H.nodes()],
#     }).sort_values("word")
#     nodes_df.to_csv(dataFrameFolder / "nodes.csv", index=False)

#     # --- edges table ---
#     e_rows = []
#     for u, v, d in H.edges(data=True):
#         e_rows.append((int(u), int(v), float(d.get("weight", 0.0))))
#     edges_df = pd.DataFrame(e_rows, columns=["word_i", "word_j", "weight"])
#     edges_df.to_csv(dataFrameFolder / "edges.csv", index=False)

#     # lightweight metadata (handy for app display)
#     meta = {
#         "num_nodes": int(H.number_of_nodes()),
#         "num_edges": int(H.number_of_edges()),
#         "layout": "spring",
#         "min_weight": float(min_weight) if min_weight is not None else None,
#         "topk_per_node": int(topk_per_node) if topk_per_node is not None else None,
#         "seed": layout_seed,
#     }
#     with open(savepointsFolder / "graph_meta.json", "w") as f:
#         import json; json.dump(meta, f, indent=2)

#     print(f"Saved nodes/edges to: {savepointsFolder}")




# def findWordFamilies(sparse_dictionary, saveFolder):    
#     import matplotlib
#     matplotlib.use("Agg")   # non-interactive backend
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import networkx as nx
#     import pandas as pd

#     imageSaveFolder = saveFolder.joinpath("outputGraphs")
#     imageSaveFolder.mkdir(parents = True, exist_ok = True)

#     dataFrameFolder = saveFolder.joinpath("dataframes")
#     dataFrameFolder.mkdir(parents = True, exist_ok = True)

#     res = compute_word_families(
#         sparse_dictionary, 
#         threshold=0.85,
#         method="components",
#         topk=3,
#         use_abs=True
#     )

#     G = res["G"]
#     G_tosave = pd.DataFrame(G)
#     G_tosave.to_excel(dataFrameFolder / "full_word_similarity.xlsx")
#     Gx = res["graph"]
#     families = res["families"]
#     stats = res["stats"]

#     # print("Redundancy stats:", stats)
#     # print("Top families (id, size):", families_summary(families)[:10])

#     # Membership vector (word -> family id)
#     K = sparse_dictionary.shape[0]
#     fam_id_of_word = family_membership_vector(families, K)

#     # --- Heatmap of lower triangle only ---
#     order = np.argsort(fam_id_of_word)
#     G_ordered = np.abs(G[np.ix_(order, order)])
#     mask = np.triu(np.ones_like(G_ordered, dtype=bool))  # True for upper triangle
#     G_masked = np.ma.array(G_ordered, mask=mask)

#     plt.figure(figsize=(6,6))
#     cmap = plt.cm.viridis # type: ignore
#     cmap.set_bad(color="white")  # hide masked values
#     im = plt.imshow(G_masked, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
#     plt.title("Atom cosine |G| (lower triangle, reordered by family)")
#     plt.colorbar(im)
#     plt.tight_layout()
#     plt.savefig(imageSaveFolder / "gram_heatmap_lower_triangle.png", dpi=150)
#     plt.close()

#     # --- Thresholded atom graph (for small K) ---
#     pos = nx.spring_layout(Gx, seed=42, weight="weight")
#     plt.figure(figsize=(7,6))
#     nx.draw_networkx_nodes(
#         Gx, pos,
#         node_size=60,
#         node_color=[fam_id_of_word[n] for n in Gx.nodes()], # type: ignore
#         cmap="tab20"
#     )
#     nx.draw_networkx_edges(Gx, pos, alpha=0.2)
#     plt.axis("off")
#     plt.title("Thresholded atom graph")
#     plt.tight_layout()
#     plt.savefig(imageSaveFolder / "atom_graph_t98.png", dpi=150)
#     plt.close()

#     export_graph_for_streamlit(
#         G=G,
#         Gx=Gx,
#         fam_id_of_word=fam_id_of_word,
#         save_dir=saveFolder,
#         min_weight=None,         # optional: drop weak edges
#         topk_per_node=None,      # or e.g., 3 to regularize degree
#         layout_seed=42,
#         layout_kwargs={"iterations": 100}
#     )

#     fam_rows = [{"family_id": i, "size": len(g), "members": " ".join(map(str, g))}
#                 for i, g in enumerate(families)]
#     pd.DataFrame(fam_rows).sort_values("size", ascending=False)\
#                           .to_csv(dataFrameFolder / "families_mutual_top3.csv", index=False)

#     # (Optional) small heatmap or summary print as you already do…
#     print("Redundancy stats:", stats)
#     print("Top families (id, size):", families_summary(families)[:10])
    



def sparseCodeAndComputeFrequency(allImagesCellFeatureDict, sparse_dictionary, userInputList, saveFolder, batch_size=5000):
    from sklearn.decomposition import sparse_encode
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("Sparse coding to convert visual words and compute frequency vectors...")

    dataFrameFolder = saveFolder.joinpath("dataframes")
    dataFrameFolder.mkdir(parents = True, exist_ok = True)

    dictionarySize = sparse_dictionary.shape[0]
    print("Dictionary stats: min", np.min(sparse_dictionary), "max", np.max(sparse_dictionary), "mean", np.mean(sparse_dictionary))
    # print(f"number of images was: {len(allImagesCellFeatureDict)}")
    # Loop over each image with a progress bar
    for imageName, imageData in tqdm(allImagesCellFeatureDict.items(), desc="Images processed"):
        featureArray = np.array(imageData["feature"])

        if featureArray.size == 0:
            print(f"Warning: {imageName} has no features. Skipping.")
            continue

        # Perform sparse coding
        sparse_codes = sparse_encode(
                X=featureArray,
                dictionary=sparse_dictionary,
                algorithm='lasso_cd',
                alpha=userInputList.lassoAlpha,
                n_jobs=-1
            )

        # Compute frequency vector
        if userInputList.pooling == "sum":
            freqVector = np.sum(np.abs(sparse_codes), axis=0)
        elif userInputList.pooling == "max":
            freqVector = np.max(np.abs(sparse_codes), axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {userInputList.pooling}")

        # Handle max-pooling case with no activations
        freqVector[freqVector == -np.inf] = 0.0

        # Store computed frequency vector
        imageData["frequencyVector"] = freqVector
        imageData["sparseCodes"] = sparse_codes

    # --- Compute total word usage across all images ---
    print("Computing total word usage across all features...")
    total_usage = np.zeros(dictionarySize, dtype=float)

    for imageData in allImagesCellFeatureDict.values():
        sparseCodes = imageData["sparseCodes"]  # shape: (n_features, n_words)
        total_usage += np.sum(np.abs(sparseCodes), axis=0)

    # Save usage to CSV
    usage_df = pd.DataFrame({
        "word_index": np.arange(dictionarySize),
        "total_usage": total_usage
    }).sort_values("total_usage", ascending=False)

    usage_df.to_excel(dataFrameFolder.joinpath(f"dictionary_{userInputList.dictionarySize}word_usage.xlsx"), index=False)

    # Save frequency vectors to Excel
    freqVectorArray = np.stack([data["frequencyVector"] for data in allImagesCellFeatureDict.values()])
    freqVectorPd = pd.DataFrame(freqVectorArray)
    saveName = dataFrameFolder.joinpath(f"frequencyVectors_sparse_{userInputList.pooling}.xlsx")
    freqVectorPd.to_excel(saveName)

    return allImagesCellFeatureDict




# def computeChromatinSplitHistograms(allImagesCellFeatureDict, userInputList, saveFolder):
#     from pathlib import Path
#     from skimage.io import imread
#     import numpy as np
#     print("computing split histograms now...")
#     """
#     Computes separate histograms of visual words for chromatin species A and B.

#     Args:
#         allImagesCellFeatureDict: contains image name and information
#         saveFolder: parent folder with all images and analysis
#         userInputList: contains classification channel (chToClassify).

#     Returns:
#         freq_A (np.ndarray): Histogram for chromatin species A (label 1).
#         freq_B (np.ndarray): Histogram for chromatin species B (label 2).
#     """
#     for imageName, imageData in allImagesCellFeatureDict.items():
#         locationArray = np.array(imageData["location"])
        

#         # Construct mask path
#         mask_name = imageName.replace(".tif", "signalSeg.tif")
#         mask_path = saveFolder / "ch1Crops" / mask_name
#         if not mask_path.exists():
#             raise FileNotFoundError(f"Mask not found: {mask_path}")

#         # Load segmentation mask
#         mask = imread(mask_path)
        
#         # Initialize histograms
#         freq_A = np.zeros(userInputList.initialDictSize)
#         freq_B = np.zeros(userInputList.initialDictSize)

#         if userInputList.clusteringAl == "sparse":
#             sparseCodes = np.abs(np.array(imageData["sparseCodes"]))
#             # Loop through all word locations
#             for code_vec, (z, y, x) in zip(sparseCodes, locationArray):
#                 label = mask[int(z), int(y), int(x)]
#                 if label == 1:
#                     freq_A += code_vec
#                 elif label == 2:
#                     freq_B += code_vec

#         if userInputList.clusteringAl == "kmeans":
#             wordList = np.abs(np.array(imageData["word"]))
#             # Loop through all word locations
#             for word, (z, y, x) in zip(wordList, locationArray):
#                 label = mask[int(z), int(y), int(x)]
#                 if label == 1:
#                     freq_A += 1
#                 elif label == 2:
#                     freq_B += 1

#         # Normalize histograms
#         # freq_A /= (np.sum(freq_A) + 1e-8)
#         # freq_B /= (np.sum(freq_B) + 1e-8)
#         num_voxels_A = np.sum([mask[int(z), int(y), int(x)] == 1 for (z, y, x) in locationArray])
#         num_voxels_B = np.sum([mask[int(z), int(y), int(x)] == 2 for (z, y, x) in locationArray])

#         freq_A /= (num_voxels_A + 1e-8)
#         freq_B /= (num_voxels_B + 1e-8)
#         # enrichment2 = freq_B / (freq_A + freq_B + 1e-8)
#         # enrichment2 = freq_B / (freq_A + 1e-8) #(freq_A + freq_B + 1e-8)
#         log_enrichment2 = np.log2((freq_B + 1e-6) / (freq_A + 1e-6))
#         log_enrichment1 = np.log2((freq_A + 1e-6) / (freq_B + 1e-6))

#         fullVector = np.concatenate([freq_A, freq_B])
#         allImagesCellFeatureDict[imageName]["chromatinHisto"] = fullVector
#         allImagesCellFeatureDict[imageName]["chromatinEnrich"] = log_enrichment1
#         allImagesCellFeatureDict[imageName]["chromatinEnrich_2"] = log_enrichment2


#     allImagesCellFeatureDict = zscore_normalize_chromatin_enrichment(allImagesCellFeatureDict)

#     return allImagesCellFeatureDict



# def zscore_normalize_chromatin_enrichment(allImagesCellFeatureDict, enrichment_key="chromatinEnrich", output_key="zNormEnrich"):
#     import numpy as np
#     """
#     Z-score normalizes the chromatin enrichment vectors across all images.

#     Args:
#         allImagesCellFeatureDict (dict): Dictionary with image data containing enrichment vectors.
#         enrichment_key (str): Key where log enrichment vectors are stored.
#         output_key (str): Key to save the z-scored vectors into.

#     Returns:
#         dict: Updated dictionary with z-score normalized enrichment vectors added.
#     """
#     # Stack all enrichment vectors into a matrix
#     enrichment_matrix = np.stack([
#         imageData[enrichment_key] for imageData in allImagesCellFeatureDict.values()
#     ])

#     # Compute mean and std across images for each word
#     mean_vec = np.mean(enrichment_matrix, axis=0)
#     std_vec = np.std(enrichment_matrix, axis=0) + 1e-6  # add epsilon to avoid divide-by-zero

#     # Apply z-score normalization and store
#     for imageName, imageData in allImagesCellFeatureDict.items():
#         enrichment = imageData[enrichment_key]
#         zscored = (enrichment - mean_vec) / std_vec
#         imageData[output_key] = zscored

#     return allImagesCellFeatureDict


def detectSIFTKeypoints(saveFolder, userInputList): 
    from skimage.io import imread
    from collections import defaultdict
    import numpy as np
    from skimage.io import imread, imsave

    print("computing SIFT keypoints and saving annotated images now...")
    cropFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
    listOfImages = list(cropFolder.glob(f"*ch{userInputList.chToClassify}.tif"))
    allImagesCellFeatureDict = defaultdict(dict)
    sigma_list = [1.0, 1.6, 2.2, 3.0]  # scale space

    for imageName in listOfImages:        
        tmpImg = imread(imageName)
        print(f"detecting keypoints for {imageName.stem}")
        if userInputList.siftNormalization == "5-95":
            tmpImgScaled = run5_95_normalization(tmpImg)
            imageSaveName = imageName.with_stem(imageName.stem + "_norm595")
            imsave(imageSaveName, tmpImgScaled)
        if userInputList.siftNormalization == "asinh":
            tmpImgScaled = run_asinh_normalization(tmpImg)
            imageSaveName = imageName.with_stem(imageName.stem + "_normAsinh")
            imsave(imageSaveName, tmpImgScaled)
        if userInputList.siftNormalization == "none":
            tmpImgScaled = tmpImg
        if userInputList.useSegmentationMasks:
            maskName = str(imageName.name).replace(f"ch{userInputList.chToClassify}", "CellposeMask")  
            tmpMask = imread(cropFolder / maskName)
        else:
            tmpMask = np.ones_like(tmpImg)

        dog_pyramid = build_DoG_pyramid(tmpImgScaled, sigma_list, num_octaves=3)
        keypoints = detect_DoG_keypoints(dog_pyramid, sigma_list, tmpMask, userInputList, num_octaves=3)
        save_keypoint_visualizations(imageName, keypoints, sigma_list, cropFolder, userInputList)
        save_3d_keypoint_outlines(imageName, keypoints, sigma_list, cropFolder, userInputList)        
        allImagesCellFeatureDict[imageName.name]["keypoints"] = keypoints

    
    return allImagesCellFeatureDict


def run5_95_normalization(image):
    import numpy as np
    low, high = np.percentile(image, (5, 95))
    image_scaled = np.clip((image - low) / (high - low), 0, 1)
    return image_scaled


def run_asinh_normalization(image):
    import numpy as np
    low, high = np.percentile(image, (25, 98))
    #clip the bottom 25 and the top 2 percentiles to reduce outlier influence, then apply asinh and rescale to [0,1]
    image = np.clip(image, low, high)
    image_asinh = np.arcsinh(image)
    image_scaled = (image_asinh - np.min(image_asinh)) / (np.max(image_asinh) - np.min(image_asinh))
    return image_scaled



def build_DoG_pyramid(volume, sigma_list, num_octaves=3):
    from scipy.ndimage import gaussian_filter, zoom
    import numpy as np

    all_dogs = []
    current_volume = volume.copy()

    for octave in range(num_octaves):
        blurred_volumes = [gaussian_filter(current_volume, sigma=s) for s in sigma_list]
        dog_pyramid = [blurred_volumes[i+1] - blurred_volumes[i] for i in range(len(sigma_list)-1)]
        all_dogs.extend(dog_pyramid)

        # Downsample the volume for the next octave
        current_volume = zoom(current_volume, zoom=0.5, order=1)  # bilinear downsampling

    return all_dogs


def detect_DoG_keypoints(dog_pyramid, sigma_list, tmpMask, userInputList, num_octaves=3):
    import numpy as np
    from scipy.ndimage import maximum_filter, minimum_filter
    #threshold default = 0.03
    threshold = userInputList.siftThreshold
    r = userInputList.siftEdgeR
    keypoints = []
    idx = 0
    for octave in range(num_octaves):
        for s in range(len(sigma_list) - 1):
            dog = dog_pyramid[idx]
            idx += 1
            # --- Find local maxima ---
            local_max = (dog == maximum_filter(dog, size=3)) & (dog > threshold)

            # --- Find local minima ---
            local_min = (dog == minimum_filter(dog, size=3)) & (dog < -threshold)

            # --- Combine both ---
            if userInputList.siftKeypointsLocation == "high":
                extrema = local_min
            if userInputList.siftKeypointsLocation == "low":
                extrema = local_max 
            if userInputList.siftKeypointsLocation == "all":
                extrema = local_max | local_min

            coords = np.argwhere(extrema)

            for z, y, x in coords:
                # Upscale coordinates to match original image resolution
                z_full = int(z * (2 ** octave))
                y_full = int(y * (2 ** octave))
                x_full = int(x * (2 ** octave))

                if (
                    z_full < tmpMask.shape[0]
                    and y_full < tmpMask.shape[1]
                    and x_full < tmpMask.shape[2]
                    and tmpMask[z_full, y_full, x_full] > 0
                    and not is_edge_like_3d(dog, z, y, x, r)
                ):
                    keypoints.append(((octave, s), z_full, y_full, x_full))
    
    return keypoints



def is_edge_like_3d(dog, z, y, x, r):
    #default r: 10, increase to find more points
    import numpy as np
    try:
        # Second-order derivatives
        Dxx = dog[z, y, x+1] + dog[z, y, x-1] - 2 * dog[z, y, x]
        Dyy = dog[z, y+1, x] + dog[z, y-1, x] - 2 * dog[z, y, x]
        Dzz = dog[z+1, y, x] + dog[z-1, y, x] - 2 * dog[z, y, x]

        # Mixed partial derivatives (central differences)
        Dxy = (dog[z, y+1, x+1] - dog[z, y+1, x-1] - dog[z, y-1, x+1] + dog[z, y-1, x-1]) / 4.0
        Dxz = (dog[z+1, y, x+1] - dog[z+1, y, x-1] - dog[z-1, y, x+1] + dog[z-1, y, x-1]) / 4.0
        Dyz = (dog[z+1, y+1, x] - dog[z+1, y-1, x] - dog[z-1, y+1, x] + dog[z-1, y-1, x]) / 4.0

        # Construct 3D Hessian matrix
        H = np.array([
            [Dxx, Dxy, Dxz],
            [Dxy, Dyy, Dyz],
            [Dxz, Dyz, Dzz]
        ])

        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(H)

        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)) or np.any(eigvals == 0):
            return True  # invalid point

        # Sort eigenvalues by absolute magnitude
        abs_eigvals = np.sort(np.abs(eigvals))
        ratio = abs_eigvals[-1] / (abs_eigvals[0] + 1e-8)

        return ratio > r

    except IndexError:
        # Near border — skip
        return True


def save_keypoint_visualizations(imgName, keypoints, sigma_list, cropFolder, userInputList):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import os
    import tifffile
    from skimage.io import imread
    import numpy as np
    from skimage.draw import circle_perimeter

    """
    Saves each z-slice of the volume as a TIFF with overlaid keypoints as red circles.

    Args:
        volume (ndarray): 3D image volume (Z, Y, X).
        keypoints (list): List of (scale_idx, z, y, x) keypoints.
        sigma_list (list): List of Gaussian sigmas corresponding to each scale.
        output_folder (str or Path): Directory where annotated TIFFs will be saved.
        prefix (str): Prefix for output file names.
    """
    imgPath = imgName
    volume = imread(imgPath)
    Z, Y, X = volume.shape
    rgb_stack = np.stack([volume] * 3, axis=-1)  # shape: (Z, Y, X, 3)
    rgb_stack = (rgb_stack / np.max(rgb_stack) * 255).astype(np.uint8)
    keypointsOnly = np.zeros_like(volume)
    for (octave_idx, scale_idx), z, y, x in keypoints:
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            sigma = sigma_list[scale_idx]
            radius = int(round(sigma * (2 ** octave_idx)))

            rr, cc = circle_perimeter(int(y), int(x), radius)        
            # Clip to bounds
            valid = (rr >= 0) & (rr < Y) & (cc >= 0) & (cc < X)
            rr, cc = rr[valid], cc[valid]

            rgb_stack[z, rr, cc, 0] = 255  # red
            rgb_stack[z, rr, cc, 1] = 0
            rgb_stack[z, rr, cc, 2] = 0

            keypointsOnly[z, rr, cc] = 1

    # Save as multi-page TIFF
    save_path = cropFolder / f"{imgName.stem}_keypoints.tif"
    savePathKeypoints = cropFolder / imgName.name.replace(f"_ch{userInputList.chToClassify}.tif", "_keypointsOnly.tif")
    tifffile.imwrite(save_path, rgb_stack)
    tifffile.imwrite(savePathKeypoints, keypointsOnly)


def save_3d_keypoint_outlines(imgName, keypoints, sigma_list, cropFolder, userInputList):
    import numpy as np
    import tifffile
    from skimage.io import imread
    from skimage.draw import circle_perimeter
    from matplotlib.cm import get_cmap
    # Load the volume
    volume = imread(imgName)
    Z, Y, X = volume.shape

    # Normalize and replicate to RGB
    rgb_stack = np.stack([volume]*3, axis=-1)
    rgb_stack = (rgb_stack / np.max(rgb_stack) * 255).astype(np.uint8)

    # Optional: a label mask for internal inspection (not required)
    label_stack = np.zeros((Z, Y, X), dtype=np.uint16)

    # Use a cyclic colormap (can adjust to 'tab20' or others)
    cmap = get_cmap("hsv", len(keypoints))

    for i, ((octave_idx, scale_idx), zc, yc, xc) in enumerate(keypoints, start=1):
        sigma = sigma_list[scale_idx]
        radius = int(round(sigma * (2 ** octave_idx)))

        # Generate distinct color per keypoint
        color = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)

        # Draw 2D circle outlines across z-slices
        for dz in range(-radius, radius + 1):
            z = zc + dz
            if 0 <= z < Z:
                r_xy = int(np.sqrt(radius**2 - dz**2))  # 2D radius in this slice
                if r_xy == 0:
                    continue
                rr, cc = circle_perimeter(int(yc), int(xc), r_xy, shape=(Y, X))
                rgb_stack[z, rr, cc, :] = color
                label_stack[z, rr, cc] = i  # optional: unique label

    # Save RGB visualization
    tmpSaveName = imgName.name.replace(f"_ch{userInputList.chToClassify}", "_keypointsOutlineRGB")
    tifffile.imwrite(cropFolder / tmpSaveName, rgb_stack)
    tifffile.imwrite(cropFolder /  imgName.name.replace(f"_ch{userInputList.chToClassify}", "_keypointsOutlineLabel"), label_stack)


def extractKeypointsFeatures(chCellCropLocation,userInputList, saveFolder, allImagesCellFeatureDict):
    import HOG3D_Keypoints as hg
    from joblib import Parallel, delayed
    
    maskString = "cellposemask"
    allFileList = list(chCellCropLocation.glob("*.tif"))
    maskFileList = [file for file in allFileList if f"{maskString}.tif" in file.name.lower()]
    print("running 3D HOG on keypoints now...")

    def process_single_file(imageName, keypointList):
        print(f"Processing {imageName}")
        tmpCellString = imageName.lower().rsplit(f"_ch{userInputList.chToClassify}", 1)[0]
        if userInputList.useSegmentationMasks:
            cellMask = next((m for m in maskFileList if tmpCellString == m.stem.lower().rsplit(f"_{maskString}", 1)[0]), None)
            if cellMask is None:
                print(f"Warning: no mask found for {imageName.name}")
                return None
        else:
            cellMask = None        
        imagePath = chCellCropLocation / imageName
        fullCellDictionary = hg.HOG_3D_keypoints(imagePath, cellMask, userInputList, keypointList)
        return imageName, fullCellDictionary

    # Run in parallel
    results = Parallel(n_jobs= userInputList.nJobs)(
        delayed(process_single_file)(imageName, data["keypoints"])
        for imageName, data in allImagesCellFeatureDict.items()
        if "keypoints" in data
    )

    # Combine results
    updatedDict = {}
    for result in results:
        if result is not None:
            imageName, fullCellDict = result
            updatedDict[imageName] = fullCellDict

    return updatedDict

    



# def calcWordSignalOccupancySparse(allImagesCellFeatureDict, saveFolder, userInputList):
#     import pandas as pd
#     from skimage.io import imread, imsave
#     import numpy as np
#     from collections import defaultdict

#     segmentationFolder = saveFolder.joinpath(f"ch{userInputList.chToClassify}Crops")
#     dataFrameFolder = saveFolder.joinpath("dataframes")
#     dataFrameFolder.mkdir(parents = True, exist_ok = True)

#     if len(list(segmentationFolder.glob(f"*ch{userInputList.chToThresh}signalSeg.tif"))) == 0:
#         print("segmentation is missing, running now...")
#         thresholdChannel(segmentationFolder, userInputList)

#     result_rows = []   

#     for imageName, imageData in allImagesCellFeatureDict.items():
#         word_contributions = defaultdict(lambda: {
#                                                 "high_positive": 0.0,
#                                                 "low_positive": 0.0,
#                                                 "high_negative": 0.0,
#                                                 "low_negative":  0.0})
#         tmpName = imageName.replace(f"ch{userInputList.chToClassify}.tif", f"ch{userInputList.chToThresh}signalSeg.tif")
#         tmpSegLoc = segmentationFolder / tmpName
#         tmpImgLoc = segmentationFolder / imageName
#         tmpSegmentation = imread(tmpSegLoc)
#         tmpImage = imread(tmpImgLoc)
        
#         locations = imageData["location"]
#         sparseCodes = imageData["sparseCodes"]  # shape: (n_features, n_words)
#         patchSizes = imageData["patch_size"]       

#         for i, (loc, coeffs, size) in enumerate(zip(locations, sparseCodes, patchSizes)):
#             z0, z1, y0, y1, x0, x1 = patch_bbox(loc, size, tmpSegmentation.shape)
#             region = tmpSegmentation[z0:z1, y0:y1, x0:x1]
#             # imsave(saveFolder / imageName.replace(".tif", "_patch.tif"), region)

#             highOnly = np.sum(region[region == 1.0])
#             lowOnly = np.sum(region[region == 2.0])
#             totalCount = np.sum(region[region != 0.0])
#             if totalCount == 0:
#                 continue
#             for word_index, val in enumerate(coeffs):
#                 if val > 0:
#                     word_contributions[word_index]["high_positive"] += (val)*(highOnly/totalCount)
#                     word_contributions[word_index]["low_positive"] += (val)*(lowOnly/totalCount)
#                 if val < 0:
#                     word_contributions[word_index]["high_negative"] += (val)*(highOnly/totalCount)
#                     word_contributions[word_index]["low_negative"] += (val)*(lowOnly/totalCount)

#         for word, contrib in word_contributions.items():
#             highpos = contrib["high_positive"]
#             highneg = contrib["high_negative"]
#             lowpos = contrib["low_positive"]
#             lowneg = contrib["low_negative"]

#             result_rows.append({
#                 "imageName": imageName,
#                 "condition": "control" if "control" in imageName.lower() else "lof",
#                 "word": word,
#                 "high_positive": round(highpos, 3),
#                 "high_negative": round(highneg, 3),
#                 "low_positive": round(lowpos, 3),
#                 "low_negative": round(lowneg, 3)
#             })

#     wordSignalDF = pd.DataFrame(result_rows)
#     wordSignalDF.to_excel(dataFrameFolder / "word_signal_location_sparse.xlsx", index=False)

#     return allImagesCellFeatureDict


def patch_bbox(loc_zyx, size, imgBounds):
    z, y, x = map(int, loc_zyx)
    Z, Y, X = map(int, imgBounds)
    r = int(size) // 2

    # ensure slice length equals `size` (half-open [lo:hi))
    def span(c, r, bound, size):
        if size % 2 == 0:
            lo, hi = c - r, c + r
        else:
            lo, hi = c - r, c + r + 1
        lo = max(0, lo); hi = min(bound, hi)
        return lo, hi

    z0, z1 = span(z, r, Z, size)
    y0, y1 = span(y, r, Y, size)
    x0, x1 = span(x, r, X, size)
    return (z0, z1, y0, y1, x0, x1)


# def generateCodebook(allImagesCellFeatureDict, userInputList, filePath):
#   import numpy as np
#   import time
#   from sklearn.cluster import KMeans, MiniBatchKMeans
#   import faiss   # type: ignore

#   print("creating the codebook now, k = " + str(userInputList.k))
#   start = time.time()
#   ###--- check to see if pre-generate codebook exists
#   saveName = f"codebook_{userInputList.k}k_{userInputList.cellSize}cellsize_{userInputList.zModifier}zMod.csv"
#   codebookFilesLoc = filePath.joinpath("codebookFiles")
#   codebookFilesLoc.mkdir(exist_ok = True)
#   if userInputList.codebookFile and codebookFilesLoc.joinpath(saveName).exists():
#     print("codebook found, loading now...")
#     codebook = np.loadtxt(codebookFilesLoc.joinpath(saveName), delimiter=",")
#   else:
#     print("codebook not found, generating one now....")
#     compiledDescriptors = np.concatenate([np.stack(imgData["feature"]) for imgData in allImagesCellFeatureDict.values()], axis=0)
#     faiss.omp_set_num_threads(10)  # Set based on your CPU cores
#     print("now running FAISS kmeans on flattened vectors...")
#     kmeansFAISS = faiss.Kmeans(d=compiledDescriptors.shape[1], k = userInputList.k, niter=190, verbose=True, nredo=1, seed=123)
#     kmeansFAISS.train(compiledDescriptors)
#     codebook = kmeansFAISS.centroids

#     np.savetxt(codebookFilesLoc.joinpath(saveName), codebook, delimiter=",")
#     np.savetxt(codebookFilesLoc.joinpath("compiledDescriptors.csv"), compiledDescriptors, delimiter=",")
#     print("time to compute codebook..." + str(time.time() - start))

#   return codebook




def convertVisualWordsComputeFrequency(allImagesCellFeatureDict, codebook,userInputList,saveFolder):
  import numpy as np
  from scipy.cluster.vq import vq
  import pandas as pd
  
  print("convert visual words and compute frequency...")
  for imageName, imageData in allImagesCellFeatureDict.items():
      featureArray = np.array(imageData["feature"])
      imgVisualWords, _ = vq(featureArray, codebook)  # Convert features to words
      allImagesCellFeatureDict[imageName]["word"] = imgVisualWords.tolist()  # Store words
  
  # Ensure codebook size is correctly defined
  codebookSize = len(codebook)  # Number of visual words (k)
  for imageName, imageData in allImagesCellFeatureDict.items():
      wordList = imageData["word"]  # Extract the list of words
      freqVector = np.zeros(codebookSize, dtype=np.int16)  # Initialize frequency vector
      # Count occurrences of each word
      for word in wordList:
          # print(f"word was {word}")
          if word is not None:  # Ensure the word has been assigned
              freqVector[word] += 1  

      # Save the frequency vector inside the image dictionary
      allImagesCellFeatureDict[imageName]["frequencyVector"] = freqVector
      del freqVector
  
  freqVectorArray = np.stack([allImagesCellFeatureDict[imageName]["frequencyVector"] for imageName in allImagesCellFeatureDict.keys()])
  freqVectorPd = pd.DataFrame(freqVectorArray)
  saveName = saveFolder.joinpath("frequencyVectors.xlsx")
  freqVectorPd.to_excel(saveName)
  return allImagesCellFeatureDict #visualWords, frequencyVectors,


def saveFirstValueDictionary(allImagesCellFeatureDict, saveFolder):
    import json
    import numpy as np

    print("saving dictionary as a json file now...") 
    savepointsFolder = saveFolder.joinpath("savepoints")
    savepointsFolder.mkdir(parents = True, exist_ok = True)
    save_path = savepointsFolder.joinpath("allImagesCellFeatureDict_firstValueOnly.json")
    # Convert NumPy arrays to lists for JSON compatibility
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):  # Convert arrays
            return obj.tolist()
        elif isinstance(obj, np.int64) or isinstance(obj, np.float32):  # type: ignore # Convert NumPy numbers
            return int(obj)  # or float(obj) if needed
        raise TypeError("Type not serializable")

    first_image_key = next(iter(allImagesCellFeatureDict))  # Get first key
    first_image_data = {first_image_key: allImagesCellFeatureDict[first_image_key]}  # Extract only that image
    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(first_image_data, f, default=convert_numpy, indent=4)
    print(f"Dictionary saved to {save_path}")
 

def dictionarySavepoint(allImagesCellFeatureDict, saveFolder):
    import joblib
    print("saving whole dictionary now....")
    savepointsFolder = saveFolder.joinpath("savepoints")
    savepointsFolder.mkdir(parents = True, exist_ok = True)
    save_path = savepointsFolder / "allImagesCellFeatureDict.joblib"
    joblib.dump(allImagesCellFeatureDict, save_path)
    print("whole dictionary was saved")

def dictionaryLoad(saveFolder):
    import joblib
    print("loading dictionary now...")
    savepointsFolder = saveFolder.joinpath("savepoints")
    savepointsFolder.mkdir(parents = True, exist_ok = True)
    load_path = savepointsFolder / "allImagesCellFeatureDict.joblib"
    allImagesCellFeatureDict = joblib.load(load_path)

    return allImagesCellFeatureDict

def normalizeVectors(saveFolder, userInputList,allImagesCellFeatureDict):
    import numpy as np
    from scipy.special import logsumexp
    from sklearn.feature_extraction.text import TfidfTransformer
    import pandas as pd
    import json

    print("normalizing vectors now...")
    savepointsFolder = saveFolder / "savepoints"
    savepointsFolder.mkdir(parents = True, exist_ok=True)
    

    if userInputList.normalizeMethod.lower() == "tfidf":
        frequencyMatrix = np.stack([imageData["frequencyVector"] for imageData in allImagesCellFeatureDict.values()])  

        tfidf_transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
        tfidfMatrix = tfidf_transformer.fit_transform(frequencyMatrix).toarray() # type: ignore
        for idx, (imageName, imageData) in enumerate(allImagesCellFeatureDict.items()):
            allImagesCellFeatureDict[imageName]["normalizedFrequencyVector"] = tfidfMatrix[idx]

    if userInputList.normalizeMethod.lower() == "tfidf l1":
        freq_matrix = np.stack([imageData["frequencyVector"] for imageData in allImagesCellFeatureDict.values()])
        # Apply TF-IDF transformation
        tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=False, sublinear_tf=False)
        tfidf_matrix = tfidf_transformer.fit_transform(freq_matrix).toarray() # type: ignore
        # Store transformed vectors back
        for i, (imageName, imageData) in enumerate(allImagesCellFeatureDict.items()):
            imageData["normalizedFrequencyVector"] = tfidf_matrix[i]
        
        top_raw = np.argsort(-freq_matrix.sum(0))[:10]
        top_tfidf = np.argsort(-tfidf_matrix.sum(0))[:10]

    
    if userInputList.normalizeMethod.lower() == "tfidf l2":
        frequencyMatrix = np.stack([imageData["frequencyVector"] for imageData in allImagesCellFeatureDict.values()])
        tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=userInputList.normSmooth, sublinear_tf=userInputList.normSubLinear)
        tfidfMatrix = tfidf_transformer.fit_transform(frequencyMatrix).toarray() # type: ignore
        for idx, (imageName, imageData) in enumerate(allImagesCellFeatureDict.items()):
            allImagesCellFeatureDict[imageName]["normalizedFrequencyVector"] = tfidfMatrix[idx]


    if userInputList.normalizeMethod.lower() == "l2 norm":
        frequencyMatrix = np.stack([imageData["frequencyVector"] for imageData in allImagesCellFeatureDict.values()])
        l2_norms = np.linalg.norm(frequencyMatrix, axis=1, keepdims=True)
        # Prevent division by zero (replace zero norms with 1)
        l2_norms[l2_norms == 0] = 1  
        l2_normalized_matrix = frequencyMatrix / l2_norms
        for i, (imageName, imageData) in enumerate(allImagesCellFeatureDict.items()):
            allImagesCellFeatureDict[imageName]["normalizedFrequencyVector"] = l2_normalized_matrix[i]

    if userInputList.normalizeMethod.lower() == "l1 norm":
        frequencyMatrix = np.stack([data["frequencyVector"] for data in allImagesCellFeatureDict.values()])
        l1_norms = np.sum(np.abs(frequencyMatrix), axis=1, keepdims=True)
        l1_norms[l1_norms == 0] = 1
        normalized_matrix = frequencyMatrix / l1_norms

        for i, (imageName, imageData) in enumerate(allImagesCellFeatureDict.items()):
            imageData["normalizedFrequencyVector"] = normalized_matrix[i]

    if userInputList.normalizeMethod == "none":
        for i, (imageName, imageData) in enumerate(allImagesCellFeatureDict.items()):
                allImagesCellFeatureDict[imageName]["normalizedFrequencyVector"] = allImagesCellFeatureDict[imageName]["frequencyVector"]


    if userInputList.normalizeMethod.lower() == "l2 tfidf":
        for imageName, imageData in allImagesCellFeatureDict.items():
            frequencyVector = np.array(imageData["frequencyVector"], dtype=np.float64)
            l2_norm = np.linalg.norm(frequencyVector)
            if l2_norm > 0:
                l2_normalized_vector = frequencyVector / l2_norm 
            else:
                l2_normalized_vector = frequencyVector  # If zero vector, keep it unchanged 
            allImagesCellFeatureDict[imageName]["l2NormalizedFrequencyVector"] = l2_normalized_vector
            
            # Step 2: Compute df-idf scaling
            N = len(allImagesCellFeatureDict) 
            frequencyMatrix = np.stack([imageData["l2NormalizedFrequencyVector"] for imageData in allImagesCellFeatureDict.values()])
            df = np.sum(frequencyMatrix > 0, axis=0)
            idf = np.log(N / (df + np.finfo(float).eps)) 
        
        # Step 3: Apply df-idf normalization after L2 normalization
        for imageName, imageData in allImagesCellFeatureDict.items():
            dfidf_vector = imageData["l2NormalizedFrequencyVector"] * idf 
            allImagesCellFeatureDict[imageName]["normalizedFrequencyVector"] = dfidf_vector 


    if userInputList.normalizeMethod.lower() == "l2 exclude tfidf":     
        N = len(allImagesCellFeatureDict)
        frequencyMatrix = np.stack([imageData["frequencyVector"] for imageData in allImagesCellFeatureDict.values()])
        df = np.sum(frequencyMatrix > 0, axis=0)
        idf = np.log(N / (df + np.finfo(float).eps))  
        for imageName, imageData in allImagesCellFeatureDict.items():
            freqVector = imageData["frequencyVector"]
            # Step 1: Identify Outliers using IQR
            Q1 = np.percentile(freqVector, 25)
            print(f"q1 was {Q1}")
            Q3 = np.percentile(freqVector, 98)
            print(f"q3 is {Q3}")
            IQR = Q3 - Q1
            print(f"threshold is {Q3 + 1.5 * IQR}")
            # Define threshold for extreme values
            upper_threshold = Q3 + 1.5 * IQR
            # Create a mask for non-outlier values
            valid_mask = freqVector <= upper_threshold
            # Step 2: Apply L2 Normalization (only to non-outliers)
            print(freqVector)
            norm_factor = np.linalg.norm(freqVector[valid_mask]) #+ np.finfo(float).eps
            print(f"norm factor is {norm_factor}")
            l2NormalizedVector = np.zeros_like(freqVector)
            l2NormalizedVector[valid_mask] = freqVector[valid_mask] / norm_factor  
            dfidfVector = l2NormalizedVector * idf
            allImagesCellFeatureDict[imageName]["normalizedFrequencyVector"] = dfidfVector

    if userInputList.normalizeMethod.lower() == "tfidf l2 threshold":
        frequencyMatrix = np.stack([imageData["frequencyVector"] for imageData in allImagesCellFeatureDict.values()])
        tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
        tfidfMatrix = tfidf_transformer.fit_transform(frequencyMatrix).toarray() # type: ignore
        ##threshold out only top 90 percentile values
        threshold = np.percentile(tfidfMatrix, 90)
        thresholdedMatrix = np.where(tfidfMatrix >= threshold, tfidfMatrix, 0)

        for idx, (imageName, imageData) in enumerate(allImagesCellFeatureDict.items()):
            allImagesCellFeatureDict[imageName]["normalizedFrequencyVector"] = thresholdedMatrix[idx]

    
    return allImagesCellFeatureDict, tfidf_transformer



def appendChromatinData(saveFolder, allImagesCellFeatureDict, userInputList):
    import pandas as pd
    import numpy as np
    print("adding chromatin data....")
    chormatinFileLoc = list(saveFolder.parent.glob("*metadata.xlsx"))[0]
    chormatinPD = pd.read_excel(chormatinFileLoc)
    chStrMatch = f"_ch{userInputList.chToClassify}"
    chormatinPD["image_name"] = chormatinPD["names"].str.replace("_hyperstack", chStrMatch, regex=False)
    chromatin_lookup = chormatinPD.set_index("image_name").to_dict("index")

    # Loop over all images in the feature dictionary
    for imageName, imageData in allImagesCellFeatureDict.items():
        if imageName not in chromatin_lookup:
            print(f"Warning: {imageName} not found in chromatin data. Appending zeros.")
            lowSig = 0.0
            highSig = 0.0
        else:
            lowSig = chromatin_lookup[imageName]["lowSig_Vol_Ratio"]
            highSig = chromatin_lookup[imageName]["highSig_Vol_Ratio"]

        # Append to the normalized frequency vector
        freqVec = imageData["normalizedFrequencyVector"]
        extendedVec = np.concatenate([freqVec, np.array([lowSig, highSig])])
        imageData["extendedFrequencyVector"] = extendedVec


def appendWordLocalizationData(allImagesCellFeatureDict):
    import pandas as pd
    import numpy as np
    print("adding word localization data....")

    # Loop over all images in the feature dictionary
    for imageName, imageData in allImagesCellFeatureDict.items():
        lowSigVector = imageData["normalizedLowSignalRatio"]
        highSigVector = imageData["normalizedHighSignalRatio"]
        # Append to the normalized frequency vector
        freqVec = imageData["normalizedFrequencyVector"]
        extendedVec = np.concatenate([freqVec, lowSigVector, highSigVector])
        imageData["extendedFrequencyVector"] = extendedVec

        
    
    return allImagesCellFeatureDict

 
# def calcCoOccMatrix(allImagesCellFeatureDict,saveFolder,vocabSize):
#   import numpy as np
#   from scipy.spatial import cKDTree
#   import pandas as pd
#   import networkx as nx
#   import matplotlib.pyplot as plt
#   import seaborn as sns

#   windowSize = 1
#   for image in allImagesCellFeatureDict.keys():

#     imageData = allImagesCellFeatureDict[image]
#     words = np.array(imageData["word"])
#     locations = np.array(imageData["location"])
#     normFreqVector = np.array(imageData["normalizedFrequencyVector"])
#     # Determine the threshold for top 15% vectors
#     freqThreshold = np.percentile(normFreqVector, 85)    
#     # Find words with normalized frequency in top 15%
#     topWordsIndices = np.where(normFreqVector >= freqThreshold)[0]

#     tree = cKDTree(locations)
#     co_matrix = np.zeros((vocabSize, vocabSize))

#     for i, word_i in enumerate(words):    
#         neighbors_idx = tree.query_ball_point(locations[i], windowSize) 
#         # print(f"Point {i} (Word {word_i}, location {locations[i]} has {len(neighbors_idx)} neighbors")       
#         for j in neighbors_idx:
#             if word_i != words[j]:  # Avoid self-counting
#                 word_j = words[j]
#                 co_matrix[word_i, word_j] += 1 

#     coMatrixDF = pd.DataFrame(co_matrix)
#     saveName = image+f"_{vocabSize}k_coMatrix.xlsx"
#     coMatrixDF.to_excel(saveFolder.joinpath(saveName))
#     allImagesCellFeatureDict[image]["coMatrix"] = co_matrix.flatten()
    
#     G = nx.Graph()

#     # Add nodes explicitly for each unique word
#     for word_idx in range(vocabSize):
#         G.add_node(word_idx)

#     # Add edges clearly with weights from co_matrix
#     for word_i in range(vocabSize):
#         for word_j in range(word_i + 1, vocabSize):  # explicitly avoid duplicates and self loops
#             weight = co_matrix[word_i, word_j] + co_matrix[word_j, word_i]
#             if weight > 0:
#                 G.add_edge(word_i, word_j, weight=weight)

#     # Visualization explicitly and clearly
#     plt.figure(figsize=(10, 10))
#     pos = nx.spring_layout(G, seed=42, k=0.3)

#     # nodes
#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')

#     # edges explicitly weighted by co-occurrence strength
#     edges = G.edges(data=True)
#     weights = [edata['weight'] for _, _, edata in edges]
#     nx.draw_networkx_edges(G, pos, width=[(w/np.max(weights))*5 for w in weights], alpha=0.7)

#     # clearly labeled nodes
#     nx.draw_networkx_labels(G, pos, font_size=12)

#     # optional: add edge labels clearly to show exact co-occurrences
#     edge_labels = {(u, v): int(d['weight']) for u, v, d in edges}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

#     plt.title(f"{image} Visual Word Co-occurrence Graph")
#     plt.axis('off')

#     # save explicitly as jpg image
#     graph_image_saveName = saveFolder.joinpath(f"{image}_word_cooccurrence_graph.jpg")
#     plt.savefig(graph_image_saveName, bbox_inches='tight', dpi=300)
#     plt.close()
#     # clean up explicitly
#     del co_matrix, G


# def thresholdCoOccMatrix(allImagesCellFeatureDict, saveFolder, vocabSize):
#   import numpy as np
#   from scipy.spatial import cKDTree
#   import pandas as pd
#   import networkx as nx
#   import matplotlib
#   matplotlib.use("Agg")  # Set the backend BEFORE importing pyplot
#   import matplotlib.pyplot as plt
#   import seaborn as sns
#   print("thresholding co occ matrix now...")
#   matrixSaveFolder = saveFolder.joinpath("matrixFiles")
#   matrixSaveFolder.mkdir(exist_ok = True)
#   spatialConnectionSaveFolder = saveFolder.joinpath("spatialConnection")
#   spatialConnectionSaveFolder.mkdir(exist_ok = True)

#   windowSize = 1  # spatial window as defined previously
#   all_freq_vectors = np.stack([
#     imageData["normalizedFrequencyVector"]
#     for imageData in allImagesCellFeatureDict.values()])
#   all_freq_values = all_freq_vectors.flatten()

#   # Compute the global 95th percentile threshold
#   global_percentile = 95
#   global_threshold = np.percentile(all_freq_values, global_percentile)
#   topWordIndices_global = np.where(all_freq_vectors.max(axis=0) >= global_threshold)[0]

#   for image in allImagesCellFeatureDict.keys():
#       imageData = allImagesCellFeatureDict[image]
#       words = np.array(imageData["word"])
#       locations = np.array(imageData["location"])
#       normFreqVector = np.array(imageData["normalizedFrequencyVector"])

#       # Explicitly determine threshold for top 10 percentile clearly
#       percentileThreshold = 95
#       freqThreshold = np.percentile(normFreqVector, percentileThreshold)

#       # Clearly find words above threshold explicitly
#       #topWordsIndices = np.where(normFreqVector >= freqThreshold)[0]

#       topWordsIndices = topWordIndices_global
#       if len(topWordsIndices) == 0:
#           print(f"No words found above frequency threshold for {image}. Skipping graph creation.")
#           continue

#       # Explicitly filter locations and words to include ONLY top words
#       top_word_mask = np.isin(words, topWordsIndices)
#       filtered_locations = locations[top_word_mask]
#       filtered_words = words[top_word_mask]

#       if len(filtered_locations) == 0:
#           print(f"No locations matched top words for {image}. Skipping graph creation.")
#           continue

#       # Compute co-occurrence matrix clearly for the filtered subset
#       tree = cKDTree(filtered_locations)
#       co_matrix_filtered = np.zeros((vocabSize, vocabSize))

#       for i, word_i in enumerate(filtered_words):
#           neighbors_idx = tree.query_ball_point(filtered_locations[i], windowSize)
#           for j in neighbors_idx:
#              word_j = filtered_words[j]
#              co_matrix_filtered[word_i, word_j] += 1
#               # if i != j:
#               #     word_j = filtered_words[j]
#               #     if word_i != word_j:
#               #         co_matrix_filtered[word_i, word_j] += 1

#       # Explicitly save filtered co-occurrence matrix
#       coMatrixDF_filtered = pd.DataFrame(co_matrix_filtered)
#       topPercent = 100 - percentileThreshold
#       saveName_filtered = image + f"_{vocabSize}k_coMatrix_top{topPercent}percent.xlsx"
#       coMatrixDF_filtered.to_excel(saveFolder.joinpath(saveName_filtered))
            
#       # Limit heatmap to topWordsIndices if you want a cleaner plot
#       co_matrix_viz = co_matrix_filtered[np.ix_(topWordsIndices, topWordsIndices)]            
#       # Convert to DataFrame with word indices as labels
#       word_labels = [str(i) for i in topWordsIndices]
#       co_matrix_viz_df = pd.DataFrame(co_matrix_viz, index=word_labels, columns=word_labels)
#       co_matrix_viz_df.to_excel(matrixSaveFolder.joinpath(f"{image}_coMatrix_top{topPercent}percent.xlsx"))

#       # Create heatmap
#       plt.figure(figsize=(10, 8))
#       sns.heatmap(co_matrix_viz_df, cmap='viridis', square=True, cbar_kws={'label': 'Co-occurrence Count'})
#       plt.title(f"Co-occurrence Heatmap: {image}")
#       plt.xlabel("Word Index")
#       plt.ylabel("Word Index")
#       plt.xticks(rotation=90)  # Rotate for readability
#       plt.yticks(rotation=0)

#       # Save heatmap image
#       heatmap_path = matrixSaveFolder.joinpath(image + f"_coMatrix_top{topPercent}percent_heatmap.png")
#       plt.tight_layout()
#       plt.savefig(heatmap_path, dpi=300)
#       plt.close()

#       allImagesCellFeatureDict[image]["coMatrix"] = co_matrix_filtered.flatten()
#       # Now explicitly build the graph ONLY using top frequency words
#       G_filtered = nx.Graph()

#       # Add nodes explicitly only for topWordsIndices
#       for word_idx in topWordsIndices:
#           G_filtered.add_node(word_idx)

#       # Add edges clearly and explicitly based on co-occurrence strength
#       for word_i in topWordsIndices:
#           for word_j in topWordsIndices:
#               if word_i < word_j:  # avoid duplicates & self-loops explicitly
#                   weight = co_matrix_filtered[word_i, word_j] + co_matrix_filtered[word_j, word_i]
#                   if weight > 0:
#                       G_filtered.add_edge(word_i, word_j, weight=weight)

#       if G_filtered.number_of_edges() == 0:
#           print(f"No edges found between top words for {image}. Skipping graph visualization.")
#           continue
      
#       # Clearly define a colormap consistently mapping node indices
#       node_labels = list(G_filtered.nodes())
#       sorted_labels = sorted(node_labels)
#       cmap = plt.cm.get_cmap('tab20', vocabSize)  # adjust colormap as needed
#       # Explicitly assign colors to each node clearly by word index
#       node_colors = [cmap(word_idx % vocabSize) for word_idx in sorted_labels]

#       # Visualization explicitly for filtered graph
#       plt.figure(figsize=(10, 10))
#       pos = nx.spring_layout(G_filtered, seed=42, k=0.3)

#       # Nodes
#       nx.draw_networkx_nodes(G_filtered, 
#                             pos, 
#                             node_size=500, 
#                             node_color=node_colors)
#                             # cmap=cmap)

#       # Edges explicitly weighted by co-occurrence strength
#       edges = G_filtered.edges(data=True)
#       weights = [edata['weight'] for _, _, edata in edges]
#       nx.draw_networkx_edges(G_filtered, pos, width=[(w/np.max(weights))*5 for w in weights], alpha=0.7)

#       # Labels explicitly for each node
#       nx.draw_networkx_labels(G_filtered, pos, font_size=12)

#       # Optional explicit edge labels
#       edge_labels = {(u, v): int(d['weight']) for u, v, d in edges}
#       # nx.draw_networkx_edge_labels(G_filtered, pos, edge_labels=edge_labels, font_size=9)

#       plt.title(f"{image} Top 10% Word Spatial Co-occurrence Graph")
#       plt.axis('off')

#       # Save explicitly as JPG
#       graph_image_saveName = spatialConnectionSaveFolder.joinpath(f"{image}_word_cooccurrence_graph.jpg")
#       plt.savefig(graph_image_saveName, bbox_inches='tight', dpi=300)
#       plt.close()
#       # Explicit clean-up
#       del co_matrix_filtered, G_filtered


# def computePairwiseMantelFromCoMatrix(allImagesCellFeatureDict, vocabSize, savePath, userInputList):
#     import numpy as np
#     from scipy.spatial.distance import squareform
#     from skbio.stats.distance import mantel # type: ignore
#     import pandas as pd
#     import itertools
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     image_names = list(allImagesCellFeatureDict.keys())
#     co_matrix_dict = {}

#     # Step 1: Reconstruct and normalize each co-occurrence matrix
#     for image in image_names:
#         flat = allImagesCellFeatureDict[image]["coMatrix"]
#         matrix = flat.reshape((vocabSize, vocabSize))

#         # Optional normalization
#         norm = matrix / np.max(matrix) if np.max(matrix) > 0 else matrix

#         # Convert to distance matrix (1 - similarity)
#         dist = 1 - norm
#         np.fill_diagonal(dist, 0)  # force diagonal to 0
#         dist = (dist + dist.T) / 2  # ensure symmetry
#         co_matrix_dict[image] = dist

#     # Step 2: Pairwise Mantel tests
#     results = []
#     mantel_matrix = pd.DataFrame(index=image_names, columns=image_names, dtype=float)

#     for img1, img2 in itertools.combinations(image_names, 2):
#         distA = co_matrix_dict[img1]
#         distB = co_matrix_dict[img2]

#         mantel_stat, p_value, _ = mantel(squareform(distA), squareform(distB),
#                                          method='pearson', permutations=999)

#         results.append((img1, img2, mantel_stat))

#         # Fill both halves of the matrix
#         mantel_matrix.loc[img1, img2] = mantel_stat
#         mantel_matrix.loc[img2, img1] = mantel_stat

#     # Set diagonal to 1s and 0s
#     np.fill_diagonal(mantel_matrix.values, 1.0)

#     # Step 3: Print summary table
#     # print("Pairwise Mantel Test Results:")
#     # for r in results:
#     #     print(f"{r[0]} vs {r[1]}: Mantel r = {r[2]:.3f}")

#     # Step 5: Save detailed pairwise results as a CSV or Excel
#     mantel_df = pd.DataFrame(results, columns=["Image1", "Image2", "Mantel_r"])
#     mantelSaveName = f"pairwise_mantel_{vocabSize}k_{userInputList.cellSize}cellsize.xlsx"
#     excel_path = savePath.joinpath(mantelSaveName)
#     mantel_df.to_excel(excel_path, index=False)
#     print(f"Saved pairwise Mantel results to: {excel_path}")

#     # Step 4: Optional heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(mantel_matrix.astype(float), annot=True, cmap="coolwarm", vmin=0, vmax=1)
#     plt.title("Pairwise Mantel Correlation Matrix")
#     plt.tight_layout()
#     # plt.show()

#     # return mantel_matrix, pval_matrix, results



# def calcWordSignalOccupancy(allImagesCellFeatureDict, saveFolder, userInputList):
#   import pandas as pd
#   from skimage.io import imread
#   import numpy as np
#   from collections import defaultdict

#   segmentationFolder = saveFolder.joinpath(f"ch{userInputList.chToClassify}Crops")
#   if len(list(segmentationFolder.glob(f"*ch{userInputList.chToClassify}signalSeg.tif"))) == 0:
#     print("segmentation is missing, running now...")
#     thresholdChannel(segmentationFolder, userInputList)

#   result_rows = []
#   wordSignalDF = pd.DataFrame()

#   for imageName, imageData in allImagesCellFeatureDict.items():
#     if "ch0" in imageName:
#         tmpName = imageName.replace("ch0.tif",f"ch{userInputList.chToClassify}signalSeg.tif")
#     if "ch1" in imageName:
#         tmpName = imageName.replace("ch1.tif",f"ch{userInputList.chToClassify}signalSeg.tif")
#     tmpSegLoc = segmentationFolder / tmpName
#     tmpSegmentation = imread(tmpSegLoc)

#     listOfWords = imageData["word"]
#     print(f"length of word list was: {len(listOfWords)}")
#     listOfLocations = imageData["location"]
#     listOfNormFreq = imageData["normalizedFrequencyVector"]
    
#     word_counts = defaultdict(lambda: {"low": 0, "high": 0, "total": 0})

#     for word, loc in zip(listOfWords,listOfLocations):
#         z, y, x = int(loc[0]), int(loc[1]), int(loc[2])
#         region = tmpSegmentation[z,y,x]
#         if region == 1:
#             word_counts[word]["high"] += 1
#         elif region == 2:
#             word_counts[word]["low"] += 1
#         word_counts[word]["total"] += 1

#     for word, counts in word_counts.items():
#             total = counts["total"]
#             low_ratio = counts["low"] / total if total > 0 else 0
#             high_ratio = counts["high"] / total if total > 0 else 0
#             # print(f"word {word} in {imageName} had occured {total} and low ratio was {low_ratio} and high ratio was {high_ratio}")
#             result_rows.append({
#                 "image": imageName,
#                 "word": word,
#                 "low_signal_ratio": round(low_ratio, 3),
#                 "high_signal_ratio": round(high_ratio, 3),
#                 "total_occurrences": total,
#                 "norm_frequency_word": listOfNormFreq[word]
#             })

#   # Convert to DataFrame and save
#   wordSignalDF = pd.DataFrame(result_rows)
#   wordSignalDF.to_excel(saveFolder.joinpath("word_signal_occupancy.xlsx"), index=False)
    
#   return wordSignalDF




# def thresholdChannel(saveFolder, userInputList):
#     from skimage.io import imread, imsave
#     import numpy as np
#     from scipy.ndimage import distance_transform_edt, binary_erosion, label, binary_fill_holes
#     import warnings
#     warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")

#     print("Now thresholding signal...")   
#     chToThresh = str(userInputList.chToClassify) 
#     imgFileList = list(saveFolder.rglob("*hyperstack.tif"))
#     maskFileList = list(saveFolder.rglob("*_CellposeMask.tif"))
#     for file in imgFileList:
#         print(f"Working on...{file.stem}")
#         fileName = file.stem.replace("hyperstack","")
#         tmpName = f"{fileName}ch{chToThresh}signalSeg.tif"
#         for mask in maskFileList:
#             fileParentName = file.name.split("_cell_crop_", 1)[0]
#             maskParentName = mask.name.split("_cell_crop_", 1)[0]
#             sameImageBool = fileParentName == maskParentName
#             sameCropNumberBool = file.stem.split("_")[-2] == mask.stem.split("_")[-2]
#             if sameImageBool and sameCropNumberBool:
#                 tmpMask = imread(mask)

#                 edtMask = distance_transform_edt(tmpMask)
#                 tmpEDTName = file.parent.joinpath(mask.stem+"EDT.tif")
#                 imsave(tmpEDTName, edtMask)
 
#                 tmpImg = imread(file)[userInputList.chToClassify,:,:,:]
#                 tmpThresChImg = file.parent.joinpath(str(file.name).replace("hyperstack","ch"+chToThresh))
#                 imsave(tmpThresChImg, tmpImg)
                    
#                 bgOnlyPixels = tmpImg[tmpMask == 0]
#                 tmpImg[tmpMask == 0] = 0
#                 tmpZeros = np.zeros((tmpImg.shape))
#                 tmpImgStd = np.std(tmpImg[tmpImg > 0], axis=0)
#                 blownOutPixels = tmpImgStd*25
#                 # print(f"blown out pixel intensity was {blownOutPixels}")
#                 tmpImgBGPerc = np.nanpercentile(bgOnlyPixels, 50, axis=None)
                
#                 bgThreshold = tmpImgBGPerc * 6.5
                
#                 tmpImgPerc = np.percentile(tmpImg[(tmpImg > 0) & (tmpImg<blownOutPixels)], 50)
#                 tmpZeros[tmpImg >= tmpImgPerc] = 1
#                 tmpZeros[(tmpImg > tmpImgBGPerc) & (tmpImg < tmpImgPerc)] = 2
#                 tmpZeros[(tmpImg <= bgThreshold) & (edtMask < 4) & (tmpMask > 0)] = 0

#                 #now clean up masks by removing single pixel outliers
#                 min_size = 10
#                 cleanedMask = np.zeros_like(tmpZeros)
#                 for cls in [1, 2]:
#                     # Create binary mask for current class
#                     binaryMask = (tmpZeros == cls)
#                     labeled_mask, num_features = label(binaryMask)
#                     cleaned_binary = np.zeros_like(binaryMask)
#                     # Iterate through each labeled region
#                     for region_label in range(1, num_features + 1):
#                         region_pixels = (labeled_mask == region_label)
#                         region_size = np.sum(labeled_mask == region_label)
#                         regionEDTValue = np.mean(edtMask[region_pixels])
#                         # Retain regions above the size threshold
#                         if region_size >= min_size or regionEDTValue > 5:
#                             cleaned_binary[labeled_mask == region_label] = True

#                     # Update the cleaned mask with retained regions
#                     cleanedMask[cleaned_binary] = cls
#                 tmpNameClean = fileName + "ch" + chToThresh +"signalSegCleaned.tif"
#                 imsave(file.parent.joinpath(tmpName), cleanedMask)









# def detectSIFTKeypointsTimelapses(saveFolder, userInputList):
#     from skimage.io import imread
#     from collections import defaultdict
#     import numpy as np
#     print("computing SIFT keypoints and saving annotated images now...")
#     cropFolder = saveFolder.joinpath("keypointImgs")
#     cropFolder.mkdir(exist_ok = True)

#     listOfImages = saveFolder.glob("*filtered.tif")
#     allImagesCellFeatureDict = defaultdict(dict)
#     sigma_list = [1.0, 1.6, 2.2, 3.0]  # scale space
#     for imageName in listOfImages:
#         tmpImg = imread(imageName)
#         low, high = np.percentile(tmpImg, (5, 95))  # remove outliers
#         tmpImgScaled = np.clip((tmpImg - low) / (high - low), 0, 1)

#         dog_pyramid = build_DoG_pyramid(tmpImgScaled, sigma_list, num_octaves=3)
#         keypoints = detect_DoG_keypoints(dog_pyramid, sigma_list, tmpImg, threshold=0.015, num_octaves=3)
#         save_keypoint_visualizations(imageName, keypoints, sigma_list, cropFolder, userInputList)
        
        
#         allImagesCellFeatureDict[imageName.name]["keypoints"] = keypoints
#     exit()
#     return allImagesCellFeatureDict



