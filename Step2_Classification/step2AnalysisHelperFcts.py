


def train_with_wandb(project, allImagesCellFeatureDict, savePath, userInputList):
    import wandb
    from pathlib import Path
    import step2ClassificationHelperFcts as classFct
    """
    1) normalizeVectors
    2) linearRegressHistos
    3) log metrics, plots, confusion matrices, and artifacts to W&B
    """

    created_run = False
    if wandb.run is None:
        # No active run (e.g., manual single run) → create one
        run = wandb.init(
            project=project,
            job_type="train",
            config=dict(
                normalizeMethod=userInputList.normalizeMethod,
                smooth_idf=getattr(userInputList, "smooth_idf", True),
                sublinear_tf=getattr(userInputList, "sublinear_tf", False),
                penalty=getattr(userInputList, "penalty", "l2"),
                C=float(getattr(userInputList, "C", 1.0)),
                n_splits=int(getattr(userInputList, "n_splits", 5)),
                solver="liblinear" if getattr(userInputList, "penalty", "l2") in ("l1","l2") else "saga",
                ch=getattr(userInputList, "chToClassify", None),
            ),
            tags=[getattr(userInputList, "normalizeMethod", "unknown")]
        )
        created_run = True
    else:
        # Reuse the sweep's active run
        run = wandb.run

    run = wandb.init(
        project=project,
        job_type="train",
        config=dict(
            normalizeMethod=userInputList.normalizeMethod,
            smooth_idf=getattr(userInputList, "normSmooth", True),
            sublinear_tf=getattr(userInputList, "normSubLinear", False),
            penalty=getattr(userInputList, "lrPenalty", "l2"),
            C=float(getattr(userInputList, "lrC", 1.0)),
            n_splits=int(getattr(userInputList, "n_splits", 5)),
            solver="liblinear" if getattr(userInputList, "penalty", "l2") in ("l1","l2") else "saga",
            ch=getattr(userInputList, "chToClassify", None),
        ),
        tags=[getattr(userInputList, "normalizeMethod", "unknown")]
    )

    # 1) Normalize
    bovwSavePath = savePath.parent
    allImagesCellFeatureDict, tfidf_obj = classFct.normalizeVectors(bovwSavePath, userInputList, allImagesCellFeatureDict)

    # 2) Train + evaluate (this call creates logreg_eval/* including metrics JSON & oof_probs.tsv)
    clf, names, labels, X, report, classMapDict = linearRegressMultiClasses(
        allImgDictionary=allImagesCellFeatureDict,
        savePath=bovwSavePath,
        userInputList=userInputList,
        tfidfObject=tfidf_obj)

    # 3) Log scalar metrics
    wandb.log({
        "cv/auc_mean": report["cv_auc_mean"],
        "cv/auc_std": report["cv_auc_std"],
        "fit/auc": report["fit_auc_all"],
        "fit/acc": report["fit_acc_all"],
        "fit/f1": report["fit_f1_all"],
        "data/V_original": report["V_original"],
        "data/V_used": report["V_used"],
        "data/n_samples": report["n_samples"],
    })

    # 4) Log plots/images saved by your code (word-effects, ROC, PR, conf PNG, etc.)
    eval_dir = Path(savePath) / "logreg_eval"
    img_candidates = list(eval_dir.glob("*.png")) + list((savePath/"attention_maps").glob("*.png"))
    for p in img_candidates:
        wandb.log({p.stem: wandb.Image(str(p))})

    # 5) Build & log W&B confusion matrices from saved OOF probabilities + thresholds
    #    (Your helper writes both: metrics_ch{ch}_lr_{model}.json and oof_probs.tsv)
    try:
        import json, numpy as np, pandas as pd

        # Find the metrics JSON (prefer the channel-specific filename)
        ch = getattr(userInputList, "chToClassify", "NA")
        model_name = getattr(userInputList, "linearRegressModel", "default")
        metrics_path = eval_dir / f"metrics_ch{ch}_lr_{model_name}.json"
        if not metrics_path.exists():
            # fallback: pick any metrics_*.json in the folder
            candidates = sorted(eval_dir.glob("metrics_ch*_lr_*.json"))
            metrics_path = candidates[0] if candidates else None

        oof_path = eval_dir / "oof_probs.tsv"

        if metrics_path is not None and metrics_path.exists() and oof_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            df = pd.read_csv(oof_path, sep="\t")  # cols: name, label, prob_control
            ctl = int(metrics["label_encoding"]["control_label"])
            lof = int(metrics["label_encoding"]["lof_label"])
            order = ["LOF", "Control"]
            to_idx = {lof: 0, ctl: 1}

            y_true_lbl = df["label"].astype(int).to_numpy()
            p_ctrl = df["prob_control"].to_numpy()

            # Youden threshold
            t_youden = float(metrics["threshold_youden_control"])
            y_pred_lbl = np.where(p_ctrl >= t_youden, ctl, lof)
            y_true_idx = [to_idx[v] for v in y_true_lbl]
            y_pred_idx = [to_idx[v] for v in y_pred_lbl]
            wandb.log({
                "confusion_youden": wandb.plot.confusion_matrix(
                    y_true=y_true_idx, preds=y_pred_idx, class_names=order
                ),
                "thresholds/youden_control": t_youden
            })

            # Macro-F1 threshold (if present)
            if "threshold_macroF1" in metrics:
                t_macro = float(metrics["threshold_macroF1"])
                y_pred_lbl_m = np.where(p_ctrl >= t_macro, ctl, lof)
                y_pred_idx_m = [to_idx[v] for v in y_pred_lbl_m]
                wandb.log({
                    "confusion_macroF1": wandb.plot.confusion_matrix(
                        y_true=y_true_idx, preds=y_pred_idx_m, class_names=order
                    ),
                    "thresholds/macroF1": t_macro
                })
    except Exception as e:
        # Don't fail the run if files are missing; just note it.
        wandb.log({"_warn/confusion_logging_error": str(e)})

    # 6) Attach artifacts (versioned files)
    art = wandb.Artifact("run_artifacts", type="results")
    for p in list(eval_dir.glob("*")):
        art.add_file(str(p))
    (savePath/"savepoints").mkdir(exist_ok=True, parents=True)
    for p in (savePath/"savepoints").glob("*"):
        art.add_file(str(p))
    run.log_artifact(art)

    # Summaries for easy sorting
    wandb.run.summary["AUC_CV_mean"] = report["cv_auc_mean"]
    wandb.run.summary["AUC_fit"] = report["fit_auc_all"]
    wandb.finish()

    if created_run:
        run.finish()










def linearRegressMultiClasses(allImgDictionary, savePath, userInputList, tfidfObject):
    """
    Multiclass (e.g., 4-class) Logistic Regression on BoVW histograms.

    Expects:
      - userInputList.class_names : list/tuple of class name strings (len=4)
      - (optional) userInputList.class_patterns : list len=4 of patterns; each pattern can be:
            * str  -> tokenized substring match
            * list/tuple[str] -> all substrings must appear
      - allImgDictionary[name]["normalizedFrequencyVector"] (or [ablateLevel]) holds the image-level feature vector
      - allImgDictionary[name]["sparseCodes"] and ["location"] exist if you want heatmaps

    Returns:
      clf, names, labels, H, report, class_map
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import re
    import json
    import numpy as np
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
        roc_auc_score, confusion_matrix, classification_report)

    # ------------------------ helpers ------------------------
    def _tokenize(s: str):
        return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

    def _pattern_to_requirements(pat):
        """
        Convert a pattern into a list of required substrings.
        - If pat is a list/tuple, treat as required substrings.
        - If pat is a string, split into tokens and require each token as substring.
        """
        if isinstance(pat, (list, tuple)):
            req = [str(x).lower() for x in pat if str(x)]
            if not req:
                raise ValueError(f"Empty class pattern list/tuple: {pat}")
            return req
        if isinstance(pat, str):
            toks = _tokenize(pat)
            if not toks:
                raise ValueError(f"Empty/invalid class pattern string: {pat!r}")
            return toks
        raise TypeError(f"class pattern must be str or list/tuple[str], got: {type(pat)}")

    def infer_labels_from_names(names, class_names, class_patterns=None):
        """
        Infer multiclass label for each name by matching patterns in filename.
        Exactly one class must match per name.
        """
        if class_patterns is None:
            class_patterns = class_names

        reqs_per_class = [_pattern_to_requirements(p) for p in class_patterns]
        labels = []

        for nm in names:
            s = nm.lower()
            matches = []
            for ci, reqs in enumerate(reqs_per_class):
                if all(r in s for r in reqs):
                    matches.append(ci)

            if len(matches) != 1:
                raise ValueError(
                    f"Ambiguous/unmatched class for file: {nm}\n"
                    f"  matches={matches}\n"
                    f"  class_names={class_names}\n"
                    f"  class_patterns={class_patterns}\n"
                    f"Tip: pass userInputList.class_patterns as explicit token-lists, e.g.\n"
                    f"  class_patterns=[['control','pre'],['control','post'],['par3oe','pre'],['par3oe','post']]"
                )
            labels.append(matches[0])

        return np.array(labels, dtype=int)

    def save_multiclass_weights(clf, out_dir, vocab_names=None, class_names=None, k=30):
        """
        Save coef/intercepts and a CSV of top words per class.
        """
        import pandas as pd

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        W = np.asarray(clf.coef_)          # (C, K) for multiclass, (1,K) for binary
        b = np.asarray(clf.intercept_)     # (C,) or (1,)
        np.save(out_dir / "logreg_weights.npy", W)
        np.save(out_dir / "logreg_intercepts.npy", b)

        if class_names is None:
            class_names = [f"class_{c}" for c in clf.classes_]

        if vocab_names is None:
            vocab_names = [f"w{i}" for i in range(W.shape[1])]

        rows = []
        for ci in range(W.shape[0]):
            w = W[ci]
            order_pos = np.argsort(-w)[:k]
            order_neg = np.argsort(w)[:k]
            for rank, idx in enumerate(order_pos, 1):
                rows.append({
                    "class": class_names[ci],
                    "direction": "positive",
                    "rank": rank,
                    "word_idx": int(idx),
                    "word": vocab_names[int(idx)],
                    "weight": float(w[idx]),
                })
            for rank, idx in enumerate(order_neg, 1):
                rows.append({
                    "class": class_names[ci],
                    "direction": "negative",
                    "rank": rank,
                    "word_idx": int(idx),
                    "word": vocab_names[int(idx)],
                    "weight": float(w[idx]),
                })

        pd.DataFrame(rows).to_csv(out_dir / "top_words_per_class.csv", index=False)

    # ------------------------ paths ------------------------
    savepointsFolder = Path(savePath) / "logreg_eval"
    savepointsFolder.mkdir(parents=True, exist_ok=True)


    # ------------------------ build X ------------------------
    names = list(allImgDictionary.keys())

    histos = []
    for name in names:
        vec = np.asarray(allImgDictionary[name]["normalizedFrequencyVector"], dtype=float)
        histos.append(vec)

    H = np.vstack(histos)
    n_samples, n_features = H.shape

    # ------------------------ labels (multiclass) ------------------------
    class_names = getattr(userInputList, "classNames", None)
    if class_names is None:
        raise ValueError("Provide userInputList.class_names as a list/tuple of 4 class name strings.")

    class_patterns = getattr(userInputList, "classPatterns", None)
    y = infer_labels_from_names(names, class_names, class_patterns=class_patterns)

    n_classes_present = int(np.unique(y).size)
    # map class name <-> label id
    classMapDict = {str(class_names[i]): int(i) for i in range(len(class_names))}
    idToClass = {int(i): str(class_names[i]) for i in range(len(class_names))}

    # ------------------------ LR hyperparams ------------------------
    penalty  = getattr(userInputList, "lrPenalty", "l2")
    C        = float(getattr(userInputList, "lrC", 1.0))
    n_splits = int(getattr(userInputList, "nSplits", 5))
    multi    = getattr(userInputList, "lrMultiClass", "multinomial")  # "multinomial" or "ovr"
    class_weight = getattr(userInputList, "classWeight", "balanced")  # "balanced" or None

    if n_classes_present == 2:
        multi = "ovr"                 # match binary-style fit
        if penalty in ("l1", "l2"):
            solver = "liblinear"      # match old default
        else:
            solver = "saga"
    
    if penalty in ("l1", "elasticnet"):
        solver = "saga"
    else:
        solver = "lbfgs" if multi == "multinomial" else "liblinear"

    lr_kwargs = dict(
        penalty=penalty,
        C=C,
        solver=solver,
        class_weight=class_weight,
        max_iter=5000,
        multi_class=multi,
    )
    if penalty == "elasticnet":
        lr_kwargs["l1_ratio"] = float(getattr(userInputList, "l1_ratio", 0.2))

    clf = LogisticRegression(**lr_kwargs) # type: ignore

    # ------------------------ CV metrics ------------------------
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_scores = cross_val_score(clf, H, y, cv=cv, scoring="accuracy")
    f1m_scores = cross_val_score(clf, H, y, cv=cv, scoring="f1_macro")
    bacc_scores = cross_val_score(clf, H, y, cv=cv, scoring="balanced_accuracy")

    # multiclass AUC (may fail if folds miss a class)
    auc_ovr_scores = None
    try:
        auc_ovr_scores = cross_val_score(clf, H, y, cv=cv, scoring="roc_auc_ovr")
    except Exception:
        auc_ovr_scores = None

    # also grab OOF preds for a confusion matrix / report
    y_oof = cross_val_predict(clf, H, y, cv=cv, method="predict")

    if n_classes_present == 2:
        # OOF probabilities for ROC curve
        proba_oof = cross_val_predict(clf, H, y, cv=cv, method="predict_proba")
        # proba_oof columns align to clf.classes_ *for that fold*, but in binary it’s stable.
        # To be extra robust, refit once to establish global class ordering:
        clf_tmp = LogisticRegression(**lr_kwargs) # type: ignore
        clf_tmp.fit(H, y)
        global_classes = clf_tmp.classes_

        auc_dict = save_binary_roc_oof(
        y_true=y,
        proba_oof=proba_oof,
        classes=global_classes,
        out_path=savepointsFolder / "roc_curve_oof.png",
        dpi=300
    )

    # ------------------------ fit final model ------------------------
    clf.fit(H, y)

    # ------------------------ save weights + report ------------------------
    vocab_names = getattr(userInputList, "vocab_names", None)
    save_multiclass_weights(
        clf=clf,
        out_dir=savepointsFolder,
        vocab_names=vocab_names,
        class_names=[idToClass[int(c)] for c in clf.classes_],
        k=30
    )

    save_weights_pretty_multiclass(
        clf=clf,
        out_dir=savepointsFolder,
        vocab_names=vocab_names,
        class_id_to_name=idToClass,   # {0:"...",1:"...",2:"...",3:"..."}
        top_k=50,
        save_pairwise=True,
        pairwise_k=30,
    )

    save_multiclass_weight_plot_jpg(
        clf=clf,
        out_dir=savepointsFolder,
        vocab_names=vocab_names,
        class_id_to_name=idToClass,     # {0: "...", 1: "...", ...}
        top_n=30,
        plot_mode="vs_mean_others",     # matches your heatmap “contrast” spirit
        filename="logreg_weights_top_words.jpg",
        dpi=300,
    )

    cm = confusion_matrix(y, clf.predict(H), labels=np.unique(y))

    cm_oof = confusion_matrix(y, y_oof, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_oof, display_labels=class_names)
    disp.plot(values_format="d", xticks_rotation=45)   # raw counts
    plt.tight_layout()
    plt.savefig(savepointsFolder / "confusion_matrix_oof.png", dpi=300)
    plt.close()    

    report = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_classes": int(len(class_names)),
        "class_names": list(class_names),
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "multi_class": multi,
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std": float(acc_scores.std()),
        "cv_f1_macro_mean": float(f1m_scores.mean()),
        "cv_f1_macro_std": float(f1m_scores.std()),
        "cv_balanced_acc_mean": float(bacc_scores.mean()),
        "cv_balanced_acc_std": float(bacc_scores.std()),
        "cv_auc_ovr_mean": float(auc_ovr_scores.mean()) if auc_ovr_scores is not None else None,
        "cv_auc_ovr_std": float(auc_ovr_scores.std()) if auc_ovr_scores is not None else None,
        "train_accuracy": float(accuracy_score(y, clf.predict(H))),
        "train_f1_macro": float(f1_score(y, clf.predict(H), average="macro")),
        "train_balanced_accuracy": float(balanced_accuracy_score(y, clf.predict(H))),
        "train_confusion_matrix": cm.tolist(),
        "oof_confusion_matrix": confusion_matrix(y, y_oof, labels=np.unique(y)).tolist(),
    }

    if n_classes_present == 2:
        report.update(auc_dict)

    with open(savepointsFolder / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(savepointsFolder / "classification_report_oof.txt", "w") as f:
        f.write(classification_report(y, y_oof, target_names=list(class_names))) # type: ignore

    with open(savepointsFolder / "names_and_labels.tsv", "w") as f:
        f.write("name\tlabel\tclass\n")
        for nm, yy in zip(names, y):
            f.write(f"{nm}\t{int(yy)}\t{idToClass[int(yy)]}\n")

    # ------------------------ optional: generate heatmaps ------------------------
    if tfidfObject is not None:
        idf_vec = getattr(tfidfObject, "idf_", None)
        if idf_vec is None:
            raise ValueError("tfidfObject provided but missing .idf_")

        contrast_mode = getattr(userInputList, "heatmap_contrast_mode", "runner_up")
        fixed_contrast = getattr(userInputList, "heatmap_contrast_class", None)

        attn_dir = Path(savePath) / "attention_maps"
        attn_dir.mkdir(parents=True, exist_ok=True)

        # precompute probabilities once (fast)
        proba_all = clf.predict_proba(H)  # (N, C)

        for i, imageName in enumerate(names):
            true_class = idToClass[int(y[i])]

            # pick a contrast class
            contrast_class = None
            if contrast_mode == "fixed":
                contrast_class = fixed_contrast
            elif contrast_mode == "runner_up":
                p = proba_all[i].copy()
                true_row = int(np.where(clf.classes_ == int(y[i]))[0][0])
                p[true_row] = -np.inf
                contrast_row = int(np.argmax(p))
                contrast_class = idToClass[int(clf.classes_[contrast_row])]
            elif contrast_mode == "mean_others":
                contrast_class = None
            else:
                # fallback
                contrast_class = None

            make_tfidf_heatmap_for_image_multi_classes(
                userInputList=userInputList,
                image_name=imageName,
                allImgDictionary=allImgDictionary,
                classMap=classMapDict,      # name->id
                clf=clf,
                idf=idf_vec,
                save_dir=attn_dir,
                ch_crop_folder=Path(savePath) / f"ch{userInputList.chToClassify}Crops",
                mode="window",
                coverage_norm=True,
                smooth_sigma=(userInputList.cellSize/2,)*3 if getattr(userInputList, "cellSize", None) else None,
                make_mips=False,
                user_cell_size=getattr(userInputList, "cellSize", None),
                prefix=true_class,
                tfidf_object=tfidfObject,
                window_paint="fill",
                target_class=true_class,
                contrast_class=contrast_class,
            )

    return clf, names, y, H, report, classMapDict



def save_binary_roc_oof(y_true, proba_oof, classes, out_path, dpi=300):
    """
    Save a standard binary ROC curve using out-of-fold probabilities.
    - y_true: shape (N,)
    - proba_oof: shape (N, 2) aligned to `classes`
    - classes: array-like of the two class ids in the same order as proba_oof columns
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    y_true = np.asarray(y_true)
    proba_oof = np.asarray(proba_oof, dtype=float)
    classes = np.asarray(classes)

    # Use the "positive" class as classes[1] (matches sklearn convention)
    pos_id = classes[1]
    pos_col = int(np.where(classes == pos_id)[0][0])

    y_bin = (y_true == pos_id).astype(int)
    p = proba_oof[:, pos_col]

    auc_val = float(roc_auc_score(y_bin, p))
    fpr, tpr, _ = roc_curve(y_bin, p)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (binary) — out-of-fold")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {"auc_oof": auc_val, "positive_class_id": int(pos_id)}


def make_tfidf_heatmap_for_image_multi_classes(
    userInputList,
    image_name: str,
    allImgDictionary: dict,
    classMap: dict,             # class name -> numeric id (0..3)
    clf,                        # fitted sklearn LogisticRegression (multiclass OK)
    idf=None,                   # IDF vector of shape (K,)
    save_dir=None,
    *,
    ch_crop_folder=None,
    vol_shape=None,
    mode="window",              # "center" or "window"
    coverage_norm=True,
    smooth_sigma=None,
    make_mips=False,
    user_cell_size=None,
    dtype_out="float32",
    prefix="",
    tfidf_object=None,
    window_paint="fill",
    target_class=None,          # string from userInputList.class_names
    contrast_class=None,        # string or None ("mean others")
):
	"""
	Multiclass-capable attribution heatmap.

	Interprets the "evidence" as a *contrast*:
		- if contrast_class is provided:
			coef = w[target] - w[contrast]
			=> positive voxels support target over contrast
		- if contrast_class is None:
			coef = w[target] - mean(w[others])
			=> positive voxels support target vs average of others

	The heatmap is made by:
		(1) building an image-level TF-IDF vector from patch masses
		(2) computing per-word contribution: coef[word] * x_tfidf[word]
		(3) apportioning each word’s contribution to patches by fractional ownership
		(4) painting patch scores back into the 3D volume
	"""
	import numpy as np
	from pathlib import Path
	from skimage.io import imread, imsave
	import pandas as pd

    # ---------------- helpers ----------------
	def patch_bbox(center_zyx, patch_size, vol_shape_zyx):
		z, y, x = map(int, center_zyx)
		Z, Y, X = map(int, vol_shape_zyx)
		r = int(patch_size) // 2
		z0 = max(0, z - r); z1 = min(Z, z + r + 1)
		y0 = max(0, y - r); y1 = min(Y, y + r + 1)
		x0 = max(0, x - r); x1 = min(X, x + r + 1)
		return z0, z1, y0, y1, x0, x1

	def _row_for_class_id(class_id: int):
		"""Map numeric class_id to row index in clf.coef_ via clf.classes_."""
		return int(np.where(clf.classes_ == class_id)[0][0])

	# ---------------- inputs & basic checks ----------------
	assert image_name in allImgDictionary, f"{image_name} not in dictionary"
	data = allImgDictionary[image_name]

	patch_codes_abs = np.abs(np.asarray(data["sparseCodes"], dtype=np.float64))  # [P, K]
	patch_centers_zyx = np.asarray(data["location"], dtype=int)                  # [P, 3]
	num_patches, vocab_size = patch_codes_abs.shape

	if idf is None:
		idf_by_word = np.ones((vocab_size,), dtype=np.float64)
	else:
		idf_by_word = np.asarray(idf, dtype=np.float64)
		if idf_by_word.shape != (vocab_size,):
			raise ValueError(f"IDF length {idf_by_word.shape[0]} != vocab_size={vocab_size}")

    # ---------------- choose target/contrast ----------------
	if target_class is None:
		# default: infer from prefix if it matches a key, else raise
		if prefix in classMap:
			target_class = prefix
		else:
			raise ValueError("Provide target_class (string) or set prefix to a valid class name.")

	if target_class not in classMap:
		raise ValueError(f"target_class={target_class!r} not in classMap keys={list(classMap.keys())}")

	target_id = int(classMap[target_class])

	classes = np.asarray(clf.classes_)
	W = np.asarray(clf.coef_)  # multiclass: (C,K), binary: (1,K)

	# Resolve contrast_id (if any)
	contrast_id = None
	if contrast_class is not None:
		if contrast_class not in classMap:
			raise ValueError(f"contrast_class={contrast_class!r} not in classMap keys={list(classMap.keys())}")
		contrast_id = int(classMap[contrast_class])

	# --- Build coef_by_word robustly for binary OR multiclass ---
	if classes.size == 2:
		# Binary LR stores only one row: weights for classes_[1] vs classes_[0]
		if W.shape[0] != 1:
			# Extremely unusual, but handle anyway
			w = W[0].ravel()
		else:
			w = W.ravel()

		pos_id = int(classes[1])
		neg_id = int(classes[0])

		# If contrast not specified, default to "the other class"
		if contrast_id is None:
			contrast_id = neg_id if target_id == pos_id else pos_id

		if contrast_id == target_id:
			coef_by_word = np.zeros_like(w)
		else:
			# Evidence for target over contrast is +w if target is pos_id else -w
			coef_by_word = w if target_id == pos_id else -w

	else:
		# Multiclass (>=3): rows align with clf.classes_
		if W.shape[0] != classes.size:
			raise ValueError(f"Inconsistent shapes: classes_ has {classes.size} classes but coef_ has {W.shape[0]} rows.")

		def _row_for_class_id(class_id: int):
			return int(np.where(classes == class_id)[0][0])

		rt = _row_for_class_id(target_id)

		if contrast_id is None:
			# target vs mean(other classes)
			other_rows = [i for i in range(W.shape[0]) if i != rt]
			coef_by_word = W[rt] - np.mean(W[other_rows], axis=0)
		else:
			rc = _row_for_class_id(contrast_id)
			coef_by_word = W[rt] - W[rc]

	coef_by_word = np.asarray(coef_by_word, dtype=np.float64).ravel()


    # ---------------- per-patch scores ----------------
	tiny = 1e-12

	# image-level word "counts" from patch masses
	count_by_word = patch_codes_abs.sum(axis=0)  # (K,)

	# match your TF-IDF configuration (log1p by default)
	use_sublinear = True if tfidf_object is None else getattr(tfidf_object, "sublinear_tf", True)
	tf_by_word = np.log1p(count_by_word) if use_sublinear else count_by_word

	# tf-idf and L2 normalize (image-level)
	tfidf_by_word = idf_by_word * tf_by_word
	tfidf_L2 = np.linalg.norm(tfidf_by_word) + tiny
	x_unit = tfidf_by_word / tfidf_L2

	# per-word contribution for the chosen contrast
	contribution_by_word = coef_by_word * x_unit  # (K,)

	# apportion each word's image-level contribution to patches by ownership
	patch_share_of_word = patch_codes_abs / (count_by_word[None, :] + tiny)  # (P,K)
	patch_scores = patch_share_of_word @ contribution_by_word                # (P,)

	# optional sanity check (can comment out later)
	img_score = float(coef_by_word @ x_unit)
	if not np.allclose(patch_scores.sum(), img_score, atol=1e-6):
        # don't hard-fail; just warn via a soft assert
		pass

    # ---------------- derive volume shape ----------------
	if vol_shape is None:
		if ch_crop_folder is None:
			raise ValueError("Provide either vol_shape=(Z,Y,X) or ch_crop_folder to read the crop.")
		vol_path = Path(ch_crop_folder) / image_name
		vol = imread(vol_path)
		vol_shape = vol.shape
		del vol

	Z, Y, X = map(int, vol_shape)

    # ---------------- patch sizes ----------------
	ps_field = data.get("patch_size", None)
	if isinstance(ps_field, (list, tuple, np.ndarray)):
		patch_sizes = np.asarray(ps_field)
		if patch_sizes.ndim == 0:
			patch_sizes = np.full((num_patches,), int(patch_sizes), dtype=int)
		elif patch_sizes.ndim == 1 and len(patch_sizes) == num_patches:
			patch_sizes = patch_sizes.astype(int)
		else:
			raise ValueError("patch_size present but not length=num_patches or scalar.")
	else:
		if ps_field is None and user_cell_size is None:
			raise ValueError("Need per-patch 'patch_size' list or a fallback user_cell_size.")
		scalar_size = int(ps_field if ps_field is not None else user_cell_size) # type: ignore
		patch_sizes = np.full((num_patches,), scalar_size, dtype=int)

	# persist some patch-level info (optional)
	data["patch_attention_scores"] = patch_scores.astype(np.float32)
	data["patch_attention_target_class"] = str(target_class)
	data["patch_attention_contrast_class"] = str(contrast_class) if contrast_class is not None else "mean_others"
	data["patch_centers_zyx"] = patch_centers_zyx.astype(np.int32)
	data["patch_sizes"] = patch_sizes.astype(np.int32)

	# ---------------- paint volume ----------------
	attention_volume = np.zeros((Z, Y, X), dtype=np.float64)
	coverage_volume = np.zeros_like(attention_volume) if coverage_norm else None

	if mode == "center":
		for (z, y, x), score in zip(patch_centers_zyx, patch_scores):
			if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
				attention_volume[z, y, x] += score
				if coverage_norm:
					coverage_volume[z, y, x] += 1.0 # type: ignore

	elif mode == "window":
		for (z, y, x), score, psize in zip(patch_centers_zyx, patch_scores, patch_sizes):
			z0, z1, y0, y1, x0, x1 = patch_bbox((z, y, x), int(psize), (Z, Y, X))
			voxels_in_window = max(1, (z1 - z0) * (y1 - y0) * (x1 - x0))

			if window_paint == "density":
				attention_volume[z0:z1, y0:y1, x0:x1] += score / voxels_in_window
			else:
				# "fill" / "mean" both treated as additive fill here
				attention_volume[z0:z1, y0:y1, x0:x1] += score

			if coverage_norm:
				coverage_volume[z0:z1, y0:y1, x0:x1] += 1.0 # pyright: ignore[reportOptionalSubscript]
	else:
		raise ValueError("mode must be 'center' or 'window'")

	# ---------------- masked display normalization ----------------
	clip_percentile = 99.0
	roi_mask = (coverage_volume > 0) if coverage_norm else np.isfinite(attention_volume) # pyright: ignore[reportOptionalOperand]
	display_volume = attention_volume.copy()

	if np.any(roi_mask):
		vals = display_volume[roi_mask]
		abs_bound = np.percentile(np.abs(vals), clip_percentile)
		if abs_bound > 0:
			display_volume = np.clip(display_volume, -abs_bound, abs_bound)

	if smooth_sigma is not None:
		from scipy.ndimage import gaussian_filter
		m = roi_mask.astype(float)
		num = gaussian_filter(display_volume * m, sigma=smooth_sigma)
		den = gaussian_filter(m, sigma=smooth_sigma) + tiny
		display_volume = num / den
		display_volume[~roi_mask] = 0.0

	if np.any(roi_mask):
		vals = display_volume[roi_mask]
		abs_bound = np.percentile(np.abs(vals), clip_percentile)
		scale = abs_bound if abs_bound > 0 else (np.max(np.abs(vals)) + tiny)
		out = np.zeros_like(display_volume, dtype=np.dtype(dtype_out))
		out[roi_mask] = np.clip(display_volume[roi_mask] / (scale + tiny), -1.0, 1.0).astype(dtype_out, copy=False)
		display_volume = out
	else:
		display_volume = np.zeros_like(display_volume, dtype=np.dtype(dtype_out))

	# ---------------- save outputs ----------------
	if save_dir is None:
		save_dir = Path(ch_crop_folder) if ch_crop_folder is not None else Path(".")
	save_dir = Path(save_dir)
	save_dir.mkdir(parents=True, exist_ok=True)

	base = Path(image_name).name.replace(".tif", "")
	# include target/contrast in filenames so you don’t overwrite
	contrast_tag = str(contrast_class) if contrast_class is not None else "meanOthers"
	safe_prefix = str(prefix) if prefix else str(target_class)

	tif_path = save_dir / f"{base}_attn_{safe_prefix}_vs_{contrast_tag}.tif"
	tif_raw_path = save_dir / f"{base}_attn_{safe_prefix}_vs_{contrast_tag}_raw.tif"

	imsave(tif_path, display_volume, check_contrast=False)
	imsave(tif_raw_path, attention_volume.astype(np.dtype(dtype_out), copy=False), check_contrast=False)

	# Save top positive AND top negative patches
	k_show = min(200, num_patches)
	order_pos = np.argsort(-patch_scores)[:k_show]
	order_neg = np.argsort(patch_scores)[:k_show]

	def _patch_df(order, tag):
		return pd.DataFrame({
            "tag": tag,
            "rank": np.arange(1, len(order) + 1),
            "score": patch_scores[order],
            "z": patch_centers_zyx[order, 0],
            "y": patch_centers_zyx[order, 1],
            "x": patch_centers_zyx[order, 2],
            "patch_size": patch_sizes[order].astype(int),
        })

	df = pd.concat([_patch_df(order_pos, "top_positive"), _patch_df(order_neg, "top_negative")], ignore_index=True)
	top_path = save_dir / f"{base}_attn_{safe_prefix}_vs_{contrast_tag}_topPatches.csv"
	df.to_csv(top_path, index=False)

	# MIPs (NOTE: max() will ignore strong negative evidence; enable if you like)
	mip_paths = {}
	if make_mips:
		from skimage.io import imsave

		def _to_uint8(img01):
			arr = np.nan_to_num(img01, nan=0.0)
			arr = np.clip(arr, 0.0, 1.0)
			return (arr * 255.0).round().astype(np.uint8)

		mip_xy = np.max(display_volume, axis=0)
		mip_xz = np.max(display_volume, axis=1)
		mip_yz = np.max(display_volume, axis=2)

		for name, arr in [("XY", mip_xy), ("XZ", mip_xz), ("YZ", mip_yz)]:
			out_png = save_dir / f"{base}_attn_{safe_prefix}_vs_{contrast_tag}_MIP_{name}.png"
			imsave(out_png, _to_uint8(arr), check_contrast=False)
			mip_paths[name] = str(out_png)

	return {
        "tif": str(tif_path),
        "tif_raw": str(tif_raw_path),
        "top_patches": str(top_path),
        "mips": mip_paths,
        "n_patches": int(num_patches),
        "vol_shape": (int(Z), int(Y), int(X)),
        "mode": mode,
        "coverage_norm": bool(coverage_norm),
        "target_class": str(target_class),
        "contrast_class": str(contrast_class) if contrast_class is not None else "mean_others",
        "img_score_no_intercept": float(img_score),
    }

def save_weights_pretty_multiclass(
    clf,
    out_dir,
    vocab_names=None,
    class_id_to_name=None,   # dict like {0:"Control_Pre", 1:"Control_Post", ...}
    top_k=50,
    save_pairwise=True,
    pairwise_k=30,
):
    """
    Multiclass pretty weight export for sklearn LogisticRegression.

    Writes:
      - weights_by_word_long.csv              (rows: class, word, coef)
      - weights_by_word_multiclass.json       (top + and - words per class)
      - pairwise_contrasts_top_words.csv      (optional; contrasts using W[a]-W[b])

    Notes on interpretation:
      - In multinomial softmax LR, each row of coef_ is a per-class logit weight.
      - For “odds ratio between classes A vs B”, use exp(W[A]-W[B]).
    """
    import json, math, csv
    import numpy as np
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = np.asarray(clf.classes_)
    W = np.asarray(clf.coef_, dtype=float)
    b = np.asarray(clf.intercept_, dtype=float)

    # --- Handle binary special case (sklearn stores (1,K) for 2-class) ---
    if classes.size == 2 and W.shape[0] == 1:
        # fall back to your original binary pretty saver if you want
        # (requires class_map_dict / name mapping for pushes_toward)
        # save_weights_pretty(clf, out_dir, vocab_names=vocab_names, class_map_dict=...)
        # For now, write a simple long CSV as well:
        w = W.ravel()
        if vocab_names is None:
            vocab_names = [f"word_{j}" for j in range(w.size)]
        with open(out_dir / "weights_by_word_long.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "intercept", "word_idx", "word", "coef", "abs_coef", "exp_coef"])
            # classes_[1] is the “positive” class for the stored row
            pos_id = classes[1]
            pos_name = str(pos_id) if class_id_to_name is None else class_id_to_name.get(int(pos_id), str(pos_id))
            for j, coef in enumerate(w):
                writer.writerow([int(pos_id), pos_name, float(b[0]), int(j), vocab_names[j],
                                 float(coef), float(abs(coef)), float(math.exp(coef))])
        return

    # --- Multiclass ---
    n_classes, n_features = W.shape
    if vocab_names is None:
        vocab_names = [f"word_{j}" for j in range(n_features)]
    else:
        vocab_names = list(vocab_names)
        if len(vocab_names) < n_features:
            vocab_names = vocab_names + [f"word_{j}" for j in range(len(vocab_names), n_features)]

    def cname(cid):
        if class_id_to_name is None:
            return f"class_{cid}"
        # class ids are often ints 0..C-1, but keep it safe:
        try:
            return str(class_id_to_name[int(cid)])
        except Exception:
            return str(class_id_to_name.get(cid, f"class_{cid}"))

    # 1) Long CSV (easy to open/filter/plot)
    long_csv = out_dir / "weights_by_word_long.csv"
    with open(long_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "intercept", "word_idx", "word", "coef", "abs_coef", "exp_coef"])
        for ri, cid in enumerate(classes):
            for j in range(n_features):
                coef = float(W[ri, j])
                writer.writerow([
                    int(cid) if str(cid).isdigit() else str(cid),
                    cname(cid),
                    float(b[ri]) if b.size == n_classes else float(b.ravel()[0]),
                    int(j),
                    vocab_names[j],
                    coef,
                    abs(coef),
                    math.exp(coef),
                ])

    # 2) JSON summary: top + and - per class
    payload = {
        "classes_": [int(c) if str(c).isdigit() else str(c) for c in classes],
        "class_names": [cname(c) for c in classes],
        "n_features": int(n_features),
        "top_k": int(top_k),
        "per_class": []
    }

    for ri, cid in enumerate(classes):
        w = W[ri].ravel()
        order_pos = np.argsort(-w)[:min(top_k, n_features)]
        order_neg = np.argsort(w)[:min(top_k, n_features)]

        def pack(idx_list):
            out = []
            for idx in idx_list:
                coef = float(w[idx])
                out.append({
                    "word_idx": int(idx),
                    "word": vocab_names[int(idx)],
                    "coef": coef,
                    "abs_coef": float(abs(coef)),
                    "exp_coef": float(math.exp(coef)),
                })
            return out

        payload["per_class"].append({
            "class_id": int(cid) if str(cid).isdigit() else str(cid),
            "class_name": cname(cid),
            "intercept": float(b[ri]) if b.size == n_classes else float(b.ravel()[0]),
            "top_positive": pack(order_pos),
            "top_negative": pack(order_neg),
        })

    with open(out_dir / "weights_by_word_multiclass.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 3) Optional: pairwise contrasts (most interpretable)
    if save_pairwise and n_classes >= 3:
        pair_csv = out_dir / "pairwise_contrasts_top_words.csv"
        with open(pair_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["contrast", "rank", "direction", "word_idx", "word", "coef_diff", "abs_coef_diff", "exp_coef_diff"])
            for ra, ca in enumerate(classes):
                for rb, cb in enumerate(classes):
                    if ra == rb:
                        continue
                    diff = (W[ra] - W[rb]).ravel()  # evidence for A vs B
                    order = np.argsort(-np.abs(diff))[:min(pairwise_k, n_features)]
                    for rank, idx in enumerate(order, 1):
                        d = float(diff[idx])
                        writer.writerow([
                            f"{cname(ca)}_vs_{cname(cb)}",
                            rank,
                            "supports_A" if d > 0 else "supports_B",
                            int(idx),
                            vocab_names[int(idx)],
                            d,
                            abs(d),
                            math.exp(d),
                        ])

def save_multiclass_weight_plot_jpg(
    clf,
    out_dir,
    vocab_names=None,
    class_id_to_name=None,   # e.g. {0:"Control_Pre", 1:"Control_Post", ...}
    top_n=30,
    plot_mode="vs_mean_others",  # "raw" or "vs_mean_others"
    filename="logreg_weights_top_words.jpg",
    dpi=300,
):
    """
    Saves a JPG bar plot of multiclass LR weights, color-coded by class.

    - plot_mode="raw": uses clf.coef_ rows directly (per-class logit weights)
    - plot_mode="vs_mean_others": uses W[c] - mean(W[others]) (more contrast-like / interpretable)
    """
    import numpy as np
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = np.asarray(clf.classes_)
    W = np.asarray(clf.coef_, dtype=float)

    # --- Handle binary special case (sklearn stores (1,K)) ---
    if classes.size == 2 and W.shape[0] == 1:
        w = W.ravel()
        K = w.size
        if vocab_names is None:
            vocab_names = [f"word_{j}" for j in range(K)]
        idx = np.argsort(-np.abs(w))[:min(top_n, K)]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(idx))))
        ax.barh(np.arange(len(idx)), w[idx])
        ax.axvline(0, linewidth=1)
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels([vocab_names[i] for i in idx])
        ax.invert_yaxis()
        ax.set_xlabel("Weight")
        ax.set_title("Top Logistic Regression Weights (binary)")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=dpi)
        plt.close(fig)
        return

    C, K = W.shape

    # vocab names
    if vocab_names is None:
        vocab_names = [f"word_{j}" for j in range(K)]
    else:
        vocab_names = list(vocab_names)
        if len(vocab_names) < K:
            vocab_names += [f"word_{j}" for j in range(len(vocab_names), K)]

    def cname(cid):
        if class_id_to_name is None:
            return f"class_{cid}"
        try:
            return str(class_id_to_name[int(cid)])
        except Exception:
            return str(class_id_to_name.get(cid, f"class_{cid}"))

    class_names = [cname(c) for c in classes]

    # choose weights to plot
    if plot_mode == "raw":
        W_plot = W
        title = "Top LR Weights (raw per-class logits)"
    elif plot_mode == "vs_mean_others":
        W_plot = np.zeros_like(W)
        for i in range(C):
            others = [j for j in range(C) if j != i]
            W_plot[i] = W[i] - np.mean(W[others], axis=0)
        title = "Top LR Weights (class vs mean of others)"
    else:
        raise ValueError("plot_mode must be 'raw' or 'vs_mean_others'")

    # pick global top words by max |weight| across classes
    score = np.max(np.abs(W_plot), axis=0)         # (K,)
    idx = np.argsort(-score)[:min(top_n, K)]       # word indices

    # grouped horizontal bars
    y = np.arange(len(idx))
    bar_h = 0.8 / C
    offsets = (np.arange(C) - (C - 1) / 2.0) * bar_h

    fig_h = max(4, 0.28 * len(idx))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    for ci in range(C):
        ax.barh(
            y + offsets[ci],
            W_plot[ci, idx],
            height=bar_h,
            label=class_names[ci],
        )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([vocab_names[i] for i in idx])
    ax.invert_yaxis()
    ax.set_xlabel("Weight")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=dpi)
    plt.close(fig)



def runHaralickMultiClasses(allImagesCellFeatureDict, saveFolder, userInputList):
    """
    Go through image dictionary, extract each image, go patch by patch and calculate the Harralick features. 
    Takes in:
    allImageDictionary: overall image dictionary
    saveFolder: folder that contains images/graphs/etc
    userInputList: user parameters
    Each patch will return a list of Harlick numbers, add back to original dictionary
    
    """

    import numpy as np
    import pandas as pd
    from skimage.io import imread, imsave
    import tifffile as tiff
    print("Looking for Haralick Features now")
    ##---Set up file locations--##
    chFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
    heatmapFolder = saveFolder / "attention_maps"
    harFeatureFolder = saveFolder / "haralickFeatureMaps"
    harFeatureFolder.mkdir(exist_ok = True)
    nLevels = 32    
    attnThreshold = 0.05 #0.50
    # which Haralick features you want full 3D maps for
    features_for_maps = [0,1,2,3,4, 5, 6,7,8,9,10,11,12]
    # this will hold all maps for all images
    har_maps_by_image = {}
    
    ##--Main loop through dictionary keys--##
    for imageName, imageData in allImagesCellFeatureDict.items():
        # print(f"working on {imageName}")
        tmpImg = imread(chFolder / imageName).astype(np.float32, copy=False)
        imgShape = tmpImg.shape
        # condition = "Control" if "control" in imageName.lower() else userInputList.conditionStr
        tmpAttnMapName = imageName.replace(".tif","") + "_attn_" #+ condition #+ "*raw.tif"
        # finalAttnName = list(heatmapFolder.glob(tmpAttnMapName))[0]
        badStrings = ["raw", "top"]
        finalAttnName = next(
            (p for p in heatmapFolder.glob(f"{tmpAttnMapName}*.tif")
            if all(b not in p.name.lower() for b in badStrings)), None)

        tmpAttnMap = imread(finalAttnName)
        listOfFeatures = []
            
        vals = tmpImg
        # robust range (you can tweak these percentiles)
        lo, hi = np.percentile(vals, [1, 99])
        vol_clipped = np.clip(tmpImg, lo, hi)
        vol_norm = (vol_clipped - lo) / (hi - lo)
        # map to 1..(n_levels-1) for foreground
        tmpImgQuan = np.zeros_like(tmpImg, dtype=np.uint8)
        tmpImgQuan = np.floor(vol_norm * (nLevels - 1e-3)).astype(np.uint8) + 1

        # --- allocate per-feature maps for THIS image ---
        har_maps = {
            f_idx: np.zeros_like(tmpImg, dtype=np.float32)
            for f_idx in features_for_maps
        }
        
        for patchLocation, patchSize in zip(imageData["location"], imageData["patch_size"]):
            z0, z1, y0, y1, x0, x1 = patch_bbox(patchLocation, patchSize, imgShape)
            tmpPatch = tmpImgQuan[z0:z1, y0:y1, x0:x1]
            tmpAttn = tmpAttnMap[z0:z1, y0:y1, x0:x1]
            if np.mean(tmpAttn) > attnThreshold:
                harFeature = haralick_3d_mahotas(tmpPatch, distance=1, return_mean=True)
                listOfFeatures.append(harFeature)
                for f_idx in features_for_maps:
                    har_maps[f_idx][z0:z1, y0:y1, x0:x1] = harFeature[f_idx]
            else:
                harFeature = np.nan
                listOfFeatures.append(harFeature)
        #--- Save out selected haralick feature maps ---##
        harMapsToSave = [4, 10, 11]
        gray01_3d = robust_norm(tmpImg, 1, 99)
        for harMapIndex in harMapsToSave:
            tmpMap = har_maps[harMapIndex]
            tmpSaveName = f"{imageName.replace('.tif','')}_haralickFeature_{harMapIndex}.tif"
            imsave(harFeatureFolder / tmpSaveName, tmpMap, check_contrast=False)       
            r01_3d = robust_norm(np.abs(tmpMap), 1, 99)
            out_path = harFeatureFolder / f"{imageName.replace('.tif','')}_haralick{harMapIndex}_stack.tif"
            save_two_channel_stack(out_path, gray01_3d, r01_3d)

        allImagesCellFeatureDict[imageName]["haralickFeatures"] = listOfFeatures
        har_maps_by_image[imageName] = har_maps
    
    ##--Build plots to show difference between conditions ---###
    har_df = build_haralick_image_summary(allImagesCellFeatureDict)
    har_df.to_excel(saveFolder / "dataframes" / f"haralickFeaturesInfo_{attnThreshold}.xlsx")
    plot_haralick_violin_combined(har_df, saveFolder, attnThreshold)
    haralickFeatureList = [(0,"Angular second moment (ASM / “energy”)"),(1,"Contrast"),(2,"Correlation"),(3,"Sum of squares (variance)"),(4,"Inverse difference moment (homogeneity)"),(5,"Sum average"),(6,"Sum variance"),(7,"Sum entropy"),(8,"Entropy"),(9, "Difference variance"),(10,"Difference entropy"),(11,"Information measure of correlation 1"),(12,"Information measure of correlation 2")]
    for featureIndex, featureName in haralickFeatureList:
        plot_haralick_feature_kde_multiClass(allImagesCellFeatureDict, saveFolder, feature_idx=featureIndex, feature_name=featureName,  classNames=userInputList.classNames, classPatterns= userInputList.classPatterns, min_patches_for_kde=2, kde_points=512)
        

def robust_norm(arr, p_lo=1, p_hi=99):
    import numpy as np
    """Normalize to 0..1 using nanpercentiles; NaNs -> 0."""
    arr = arr.astype(np.float32, copy=False)
    # m = np.isfinite(arr)
    # if not np.any(m):
    #     return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.nanpercentile(arr, [p_lo, p_hi])
    # if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
    #     out = np.zeros_like(arr, dtype=np.float32)
    #     out[~m] = 0
    #     return out

    out = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    # out[~m] = 0
    return out

def make_rgb_overlay(gray01, r01, g01, b01, gain=1.0):
    import numpy as np
    """gray01, r01, g01, b01 are 0..1 arrays of same shape (2D or 3D)."""
    rgb = np.stack([gray01, gray01, gray01], axis=-1)  # grayscale base
    rgb[..., 0] = np.clip(rgb[..., 0] + gain * r01, 0, 1)
    rgb[..., 1] = np.clip(rgb[..., 1] + gain * g01, 0, 1)
    rgb[..., 2] = np.clip(rgb[..., 2] + gain * b01, 0, 1)
    return rgb

def save_two_channel_stack(out_path, img3d, feat3d):
    import numpy as np
    import tifffile as tiff
    """
    Saves a 2-channel 3D stack for Fiji/ImageJ:
      Channel 1: img3d
      Channel 2: feat3d
    Both should be (Z, Y, X).
    """
    img3d  = np.asarray(img3d, dtype=np.float32)
    feat3d = np.asarray(feat3d, dtype=np.float32)
    assert img3d.shape == feat3d.shape, "img3d and feat3d must match shape (Z,Y,X)"

    Z, Y, X = img3d.shape
    stack = np.stack([img3d, feat3d], axis=1)      # (Z, C, Y, X)
    stack = stack[None, ...]                       # (T=1, Z, C, Y, X)

    tiff.imwrite(
            out_path,
            stack,
            imagej=True,
            metadata={"axes": "TZCYX"},
            compression="zlib",
        )


def patch_bbox(center_zyx, patch_size, vol_shape_zyx):
        """Axis-aligned integer bbox around center with ~patch_size support."""
        z, y, x = map(int, center_zyx)
        Z, Y, X = map(int, vol_shape_zyx)
        r = int(patch_size) // 2
        z0 = max(0, z - r); z1 = min(Z, z + r + 1)
        y0 = max(0, y - r); y1 = min(Y, y + r + 1)
        x0 = max(0, x - r); x1 = min(X, x + r + 1)
        return z0, z1, y0, y1, x0, x1




def haralick_3d_mahotas(imagePatch, distance=1,
                         return_mean=True):
    """
    Compute 3D Haralick features for a (Z,Y,X) patch using mahotas.
    Returns a 1D feature vector if return_mean=True,
    otherwise a (13, n_features) array (one row per direction).
    """    
    import mahotas as mh # type: ignore

    feats = mh.features.haralick(
        imagePatch,
        distance=distance,
        ignore_zeros=True,        # treat 0 (outside mask) as background
        return_mean=return_mean   # collapse across directions
        # you can also set compute_14th_feature=True if you want all 14
    )
    # feats is float64 array
    return feats


def build_haralick_image_summary(allImageDictionary):
    """
    Returns a DataFrame with one row per (image, feature_idx):
      columns: image_name, condition, feature_idx, feature_sum, n_patches
    """    
    import numpy as np
    import pandas as pd

    rows = []
    belowThresCount = 0 
    allCount = 0

    for imageName, imageData in allImageDictionary.items():
        # print(f"summarizing now: {imageName}")
        allCount += 1
        condition = "control" if "control" in imageName.lower() else "lof"

        feat_list = imageData.get("haralickFeatures", None)
        if feat_list is None or len(feat_list) == 0:
            print(f"  [WARN] {imageName}: no haralickFeatures list")
            continue

        valid_vecs = []
        for f in feat_list:
            # Skip patches that weren't analyzed (scalar NaN sentinel)
            if np.isscalar(f) and (isinstance(f, float) and np.isnan(f)):
                continue

            f_arr = np.asarray(f, dtype=np.float64)
            # optionally skip rows that are all-NaN
            if np.isnan(f_arr).all():
                continue

            valid_vecs.append(f_arr)

        if not valid_vecs:
            print(f"  [WARN] {imageName}: no patches above attention threshold")
            belowThresCount += 1
            continue

        feats = np.vstack(valid_vecs)  # shape: (P_valid, F)
        n_patches, n_features = feats.shape

        feature_avg = feats.mean(axis=0)

        for f_idx in range(n_features):
            rows.append({
                "image_name": imageName,
                "condition": condition,
                "feature_idx": f_idx,
                "feature_avg": feature_avg[f_idx],
                "n_patches": n_patches,
            })
    print(f"Out of {allCount} images {belowThresCount} images weren't analyzed")
    return pd.DataFrame(rows)




def plot_haralick_violin(har_df, saveFolder, conditionStr):
    """
    Make a violin plot for control images only.
    x-axis: Haralick feature index
    y-axis: feature_sum (per image)
    """        
    import matplotlib.pyplot as plt
    import numpy as np
    # Focus on control cells
    df_ctrl = har_df[har_df["condition"] == conditionStr].copy()
    if df_ctrl.empty:
        print("No control rows found in Haralick summary.")
        return

    # Sort feature indices so they plot in order
    feature_ids = sorted(df_ctrl["feature_idx"].unique())

    # Prepare data for violinplot: list of 1D arrays, one per feature
    violin_data = [
        df_ctrl.loc[df_ctrl["feature_idx"] == f_idx, "feature_avg"].values
        for f_idx in feature_ids
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    parts = ax.violinplot(
        violin_data,
        positions=feature_ids,
        showmeans=True,
        showextrema=True,
        showmedians=False,
    )

    ax.set_xlabel("Haralick feature index")
    ax.set_ylabel(f"Mean across high-attention patches ({conditionStr} images)")
    ax.set_title(f"Distribution of Haralick feature means across {conditionStr} cells")

    ax.set_xticks(feature_ids)
    ax.set_xticklabels([str(i) for i in feature_ids], rotation=45)

    fig.tight_layout()

    out_dir = saveFolder / "outputGraphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"haralick_violin_{conditionStr}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved Haralick violin plot to: {out_path}")




def plot_haralick_violin_combined(har_df, saveFolder, threshold):
    """
    Combined violin plot for control vs LOF.

    x-axis: Haralick feature index
    y-axis: mean Haralick value across high-attention patches (per image)
    Two violins per feature: control (left) and lof (right).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Only keep the two conditions we care about
    conditions = ["control", "lof"]
    df = har_df[har_df["condition"].isin(conditions)].copy()
    if df.empty:
        print("No control/lof rows found in Haralick summary.")
        return

    feature_ids = sorted(df["feature_idx"].unique())
    if len(feature_ids) == 0:
        print("No feature indices found in Haralick summary.")
        return

    # Build data per feature & condition
    data_control = [
        df.loc[(df["condition"] == "control") &
               (df["feature_idx"] == f_idx), "feature_avg"].values
        for f_idx in feature_ids
    ]
    data_lof = [
        df.loc[(df["condition"] == "lof") &
               (df["feature_idx"] == f_idx), "feature_avg"].values
        for f_idx in feature_ids
    ]

    # Base positions: one x per feature
    base_x = np.array(feature_ids, dtype=float)
    offset = 0.15  # half-spacing between control & lof violins

    pos_control = base_x - offset
    pos_lof     = base_x + offset

    fig, ax = plt.subplots(figsize=(10, 5))

    # Control violins
    v_control = ax.violinplot(
        data_control,
        positions=pos_control,
        showmeans=True,
        showextrema=False,
        showmedians=False,
    )
    for body in v_control["bodies"]: # type: ignore
        body.set_facecolor("tab:blue")
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    if "cmeans" in v_control:
        v_control["cmeans"].set_color("black")

    # LOF violins
    v_lof = ax.violinplot(
        data_lof,
        positions=pos_lof,
        showmeans=True,
        showextrema=False,
        showmedians=False,
    )
    for body in v_lof["bodies"]: # type: ignore
        body.set_facecolor("tab:orange")
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    if "cmeans" in v_lof:
        v_lof["cmeans"].set_color("black")

    # Axes / ticks / labels
    ax.set_xlabel("Haralick feature index")
    ax.set_ylabel("Mean across high-attention patches (per image)")
    ax.set_title(f"Haralick feature distributions: control vs LOF, attention threshold: {threshold}")

    ax.set_xticks(base_x)
    ax.set_xticklabels([str(i) for i in feature_ids], rotation=45)

    # Legend
    legend_patches = [
        Patch(facecolor="tab:blue",  edgecolor="black", alpha=0.6, label="control"),
        Patch(facecolor="tab:orange", edgecolor="black", alpha=0.6, label="lof"),
    ]
    ax.legend(handles=legend_patches, title="Condition", loc="best")

    fig.tight_layout()

    out_dir = saveFolder / "outputGraphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"haralick_violin_control_vs_lof_{threshold}threshold.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved combined Haralick violin plot to: {out_path}")

    
def plot_haralick_feature_kde_multiClass(
    allImageDictionary,
    saveFolder,
    feature_idx=6,
    feature_name="Haralick 6",
    classNames=None,
    classPatterns=None,
    min_patches_for_kde=2,
    kde_points=512,
):
    """
    Plot KDEs of a single Haralick feature across *patches*, separated into classes.

    Saves:
      - PNG plot to:   saveFolder/outputGraphs/haralick_feature{idx}_patchKDE.png
      - Excel (x + densities) to: saveFolder/dataframes/haralick_feature{idx}_patchKDE_data.xlsx
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import gaussian_kde
    import pandas as pd
    import re

    # collect values per class
    vals_by_class = collect_patch_haralick_feature_multiClass(
        allImageDictionary,
        feature_idx=feature_idx,
        classNames=classNames,
        classPatterns=classPatterns,
        strict=False,
    )

    # keep only classes with enough samples for KDE
    valid = {k: v for k, v in vals_by_class.items() if v is not None and len(v) >= min_patches_for_kde}

    # print counts for all classes (even if insufficient)
    counts_str = ", ".join([f"{k}={len(v)}" for k, v in vals_by_class.items()])
    print(f"[KDE] Feature {feature_idx} patch counts: {counts_str}")

    if len(valid) == 0:
        print(f"[WARN] No classes have >= {min_patches_for_kde} patches; skipping KDE.")
        return

    # Common x-range across valid classes
    all_vals = np.concatenate(list(valid.values()))
    x_min = float(np.min(all_vals))
    x_max = float(np.max(all_vals))
    pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    xs = np.linspace(x_min - pad, x_max + pad, int(kde_points))

    # compute KDE curves
    density_cols = {}
    kde_curves = {}  # class -> ys
    for cls, arr in valid.items():
        kde = gaussian_kde(arr)
        ys = kde(xs)
        kde_curves[cls] = ys

        # make an excel-safe column name
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(cls)).strip("_")
        density_cols[f"density_{safe}"] = ys

    # ---------- SAVE x,y VALUES TO EXCEL ----------
    dataframeFolder = Path(saveFolder) / "dataframes"
    dataframeFolder.mkdir(parents=True, exist_ok=True)

    df_kde = pd.DataFrame({"x": xs, **density_cols})
    excel_path = dataframeFolder / f"haralick_feature{feature_idx}_patchKDE_data.xlsx"
    df_kde.to_excel(excel_path, index=False)
    # ------------------------------------------------

    # ---------- PLOT ----------
    fig, ax = plt.subplots(figsize=(9, 4))

    for cls, ys in kde_curves.items():
        n = len(valid[cls])
        ax.plot(xs, ys, lw=2, alpha=0.9, label=f"{cls} (n={n})")
        ax.fill_between(xs, ys, alpha=0.18)

    ax.set_xlabel(f"{feature_name} value (feature index {feature_idx})")
    ax.set_ylabel("Density (KDE over patches)")
    ax.set_title(f"KDE of Haralick feature {feature_idx} across patches (multiclass)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_dir = Path(saveFolder) / "outputGraphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"haralick_feature{feature_idx}_patchKDE.svg"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved patch-level KDE for feature {feature_idx} to {out_path}")
    print(f"[OK] Saved KDE data to {excel_path}")



def collect_patch_haralick_feature_multiClass(
    allImageDictionary,
    feature_idx=6,
    classNames=None,
    classPatterns=None,
    strict=False,
):
    """
    Collect patch-level Haralick values for a single feature index, split into classes.

    Args:
        allImageDictionary: dict keyed by imageName, each entry contains "haralickFeatures" list
        feature_idx: which Haralick feature index to collect
        classNames: list of class name strings
        classPatterns: list of patterns parallel to classNames. Each pattern can be:
                       - list/tuple of required substrings, e.g. ["control","pre"]
                       - OR a string (will be tokenized)
        strict: if True, raise on unmatched/ambiguous class; if False, skip with warning

    Returns:
        vals_by_class: dict {className: np.array([...])} (1D float arrays)
    """
    import numpy as np
    import re


    # ----------------- pattern helpers -----------------
    def _tokenize(s: str):
        return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

    def _pattern_to_requirements(pat):
        if isinstance(pat, (list, tuple)):
            req = [str(x).lower() for x in pat if str(x)]
            if not req:
                raise ValueError(f"Empty class pattern list/tuple: {pat}")
            return req
        if isinstance(pat, str):
            toks = _tokenize(pat)
            if not toks:
                raise ValueError(f"Empty/invalid class pattern string: {pat!r}")
            return toks
        raise TypeError(f"class pattern must be str or list/tuple[str], got: {type(pat)}")

    reqs_per_class = [_pattern_to_requirements(p) for p in classPatterns] # pyright: ignore[reportOptionalIterable]

    def _infer_class(image_name: str):
        s = image_name.lower()
        matches = [ci for ci, reqs in enumerate(reqs_per_class) if all(r in s for r in reqs)]
        if len(matches) != 1:
            msg = (
                f"Ambiguous/unmatched class for file: {image_name}\n"
                f"  matches={matches}\n"
                f"  classNames={classNames}\n"
                f"  classPatterns={classPatterns}\n"
            )
            if strict:
                raise ValueError(msg)
            else:
                print("[WARN]", msg.strip())
                return None
        return classNames[matches[0]] # pyright: ignore[reportOptionalSubscript]

    # ----------------- collect values -----------------
    vals_by_class = {c: [] for c in classNames} # pyright: ignore[reportOptionalIterable]

    for imageName, imageData in allImageDictionary.items():
        cls = _infer_class(imageName)
        if cls is None:
            continue

        feat_list = imageData.get("haralickFeatures", None)
        if not feat_list:
            continue

        for f in feat_list:
            # skip non-analyzed patches (scalar NaN)
            if np.isscalar(f) and isinstance(f, float) and np.isnan(f):
                continue

            arr = np.asarray(f, dtype=np.float64)
            if arr.ndim == 0:
                continue
            if feature_idx >= arr.shape[0]:
                continue

            val = arr[feature_idx]
            if np.isnan(val):
                continue

            vals_by_class[cls].append(val)

    # cast to arrays
    for k in list(vals_by_class.keys()):
        vals_by_class[k] = np.asarray(vals_by_class[k], dtype=np.float64) # pyright: ignore[reportArgumentType]

    return vals_by_class



def collect_patch_haralick_feature(allImageDictionary, feature_idx=6):
    """
    Collect patch-level Haralick values for a single feature index.

    Returns:
        control_vals: 1D np.array of feature_idx values for all control patches
        lof_vals:     1D np.array of feature_idx values for all LOF patches
    """
    import numpy as np

    control_vals = []
    lof_vals = []

    for imageName, imageData in allImageDictionary.items():
        condition = "control" if "control" in imageName.lower() else "lof"

        feat_list = imageData.get("haralickFeatures", None)
        if not feat_list:
            continue

        for f in feat_list:
            # skip non-analyzed patches (scalar NaN)
            if np.isscalar(f) and isinstance(f, float) and np.isnan(f):
                continue

            arr = np.asarray(f, dtype=np.float64)
            if arr.ndim == 0:
                # just in case something weird slipped in
                continue
            if feature_idx >= arr.shape[0]:
                # safety guard if feature_idx is out of range
                continue

            val = arr[feature_idx]
            if np.isnan(val):
                continue

            if condition == "control":
                control_vals.append(val)
            else:
                lof_vals.append(val)

    return np.asarray(control_vals), np.asarray(lof_vals)





def calcAttnMaskOverlap(allImagesCellFeatureDict, saveFolder, userInputList):    
    import numpy as np
    import pandas as pd
    from skimage.io import imread, imsave
    import tifffile as tiff
    import plotly.express as px

    ##---Set up file locations--##
    chFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
    heatmapFolder = saveFolder / "attention_maps"
    graphFolder = saveFolder / "outputGraphs"
    graphFolder.mkdir(exist_ok = True)
    wiggleRoom = 0.1
    dataList = []

    ##--Main loop through dictionary keys--##
    for imageName, imageData in allImagesCellFeatureDict.items():
        print(f"working on:{imageName}")
        tmpMaskName = imageName.replace("_ch0", "_rgbmask")
        tmpMask = tiff.imread(chFolder / tmpMaskName)

        beginningName = imageName.split("_attn_")[0]
        conditionStr = "Control" if "control" in beginningName.lower() else userInputList.conditionStr
        groupStr = "preNetrin" if "prenetrin" in beginningName.lower() else "postNetrin"

        rgbMask = ensure_channels_last(tmpMask)      
        R_mask = rgbMask[..., 0]
        G_mask = rgbMask[..., 1]
        B_mask = rgbMask[..., 2]
        tmpAttnMapName = imageName.replace(".tif","") + "_attn_"
        badStrings = ["raw", "top"]
        finalAttnName = next((p for p in heatmapFolder.glob(f"{tmpAttnMapName}*.tif")
            if all(b not in p.name.lower() for b in badStrings)), None)
        tmpAttnMap = imread(finalAttnName)

        #make max projections of attention map
        attnMIP_positive = np.max(np.where(tmpAttnMap > 0, tmpAttnMap, 0), axis=0)
        attnMIP_negative = np.max(np.where(tmpAttnMap < 0, -tmpAttnMap, 0), axis=0)
        attnMIP_all      = np.max(np.abs(tmpAttnMap), axis=0)   # "all" as magnitude

        #threshold masks, blue = clusters, green = cell, blue + green = total cell coverage
        total_cell_mask = (G_mask > wiggleRoom) | (B_mask > wiggleRoom)
        cluster_mask = (B_mask > wiggleRoom)
        red_only_bg = (R_mask > wiggleRoom) & (G_mask <= wiggleRoom) & (B_mask <= wiggleRoom)
        total_cell_mask = total_cell_mask & (~red_only_bg)
        cluster_mask = cluster_mask & (~red_only_bg)

        #calculate overlap
        positive_attn_total_cell_overlap = np.sum(attnMIP_positive[total_cell_mask]) / (np.sum(attnMIP_positive) + 1e-8)
        positive_attn_cluster_overlap = np.sum(attnMIP_positive[cluster_mask]) / (np.sum(attnMIP_positive) + 1e-8)
        negative_attn_total_cell_overlap = np.sum(attnMIP_negative[total_cell_mask]) / (np.sum(attnMIP_negative) + 1e-8)
        negative_attn_cluster_overlap = np.sum(attnMIP_negative[cluster_mask]) / (np.sum(attnMIP_negative) + 1e-8)
        all_attn_total_cell_overlap = np.sum(attnMIP_all[total_cell_mask]) / (np.sum(attnMIP_all) + 1e-8)
        all_attn_cluster_overlap = np.sum(attnMIP_all[cluster_mask]) / (np.sum(attnMIP_all) + 1e-8)

        #output excel file and graph of those results
        data_entry = {
            "image_name": imageName,
            "group": f"{conditionStr}_{groupStr}",
            "positive_attn_total_cell_overlap": positive_attn_total_cell_overlap,
            "positive_attn_cluster_overlap": positive_attn_cluster_overlap,
            "negative_attn_total_cell_overlap": negative_attn_total_cell_overlap,
            "negative_attn_cluster_overlap": negative_attn_cluster_overlap,
            "all_attn_total_cell_overlap": all_attn_total_cell_overlap,
            "all_attn_cluster_overlap": all_attn_cluster_overlap
        }
        dataList.append(data_entry)
    
    

    overlapDataFrame = pd.DataFrame(dataList)
    overlapDataFramePath = saveFolder / "dataframes" / "attentionMaskOverlap.xlsx"
    overlapDataFrame.to_excel(overlapDataFramePath, index = False)
    print(f"Saved attention mask overlap data to: {overlapDataFramePath}")

    dataToGraphX = ["positive_attn_cluster_overlap", "negative_attn_cluster_overlap", "positive_attn_total_cell_overlap", "negative_attn_total_cell_overlap", "all_attn_total_cell_overlap", "all_attn_cluster_overlap"]
    plotTitleList = ["Positive Attention Mask Cluster Overlap", "Negative Attention Mask Cluster Overlap", "Positive Attention Mask Total Cell Overlap", "Negative Attention Mask Total Cell Overlap", "All Attention Mask Total Cell Overlap", "All Attention Mask Cluster Overlap"]
    fileSaveNameList = ["positive_attn_cluster_overlap", "negative_attn_cluster_overlap", "positive_attn_total_cell_overlap", "negative_attn_total_cell_overlap", "all_attn_total_cell_overlap", "all_attn_cluster_overlap"]

    for dataX, plotTitle, fileSaveName in zip(dataToGraphX, plotTitleList, fileSaveNameList):
        fig = px.histogram(
            overlapDataFrame,
            x=dataX,
            color="group",
            opacity=0.6,
            title=plotTitle)
        fig.for_each_trace(lambda t: t.update(offsetgroup=None))
        fig.update_layout(template="plotly_white")
        save_fig(graphFolder, fig, f"histo_{fileSaveName}")



        
        

def ensure_channels_last(arr):
    """
    Return array with channels in the last axis.
    Supports shapes:
      (Y, X, 3), (Z, Y, X, 3), (3, Y, X), (3, Z, Y, X)
    """
    import numpy as np

    arr = np.asarray(arr)

    if arr.ndim < 3:
        raise ValueError(f"Expected at least 3D array for RGB, got shape {arr.shape}")

    # If channels already last
    if arr.shape[-1] == 3:
        return arr

    # If channels first
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)

    raise ValueError(
        f"Can't identify RGB channel axis (expected a dimension of size 3). Got shape {arr.shape}"
    )



def runPCAandUmapSparse(multiScaleVectors, saveFolder, userInputList, sampleNumber):
    from sklearn.decomposition import PCA
    import pandas as pd
    from umap import UMAP
    import numpy as np
    from pathlib import Path

    print("Starting PCA....") 
    dataFrameFolder = saveFolder.joinpath("dataframes")
    dataFrameFolder.mkdir(parents = True, exist_ok = True)

    outputGraphFolder = saveFolder.joinpath("outputGraphs")
    outputGraphFolder.mkdir(parents = True, exist_ok = True)

    freqVectorChoice = "normalizedFrequencyVector"
    normScaledFreqVectors = np.stack([multiScaleVectors[imageName][freqVectorChoice] for imageName in multiScaleVectors.keys()])

    # Perform PCA to reduce to 2D (two principal components)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(normScaledFreqVectors)
    exVariance = pca.explained_variance_ratio_

    print("Starting u-map....")
    print("scanning umap conditions now")

    uMAPConditions = [2, 3, 4, 5, 6, 10]
    data_list = []
    for condition in uMAPConditions:
        print(f"working on {condition} number of neighbors...")
        reducer = UMAP(n_neighbors=condition, n_components=2, n_epochs=1000, metric="euclidean", random_state=42)
        embedding = reducer.fit_transform(normScaledFreqVectors)
        for i, (imageName, imageData) in enumerate(multiScaleVectors.items()):
            multiScaleVectors[imageName]["umapx"] = embedding[i, 0] # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]
            multiScaleVectors[imageName]["umapy"] = embedding[i, 1] # type: ignore
        for image_name, image_data in multiScaleVectors.items():
            name_lower = image_name.lower()
            groupStr = userInputList.get_group_name(Path(image_name).stem)

            image_entry = {
                "image_name": image_name,
                "neighborNumber": condition,
                "umapx": image_data["umapx"],
                "umapy": image_data["umapy"],
                "group": groupStr,
            }      
            data_list.append(image_entry)

    # Convert list of dictionaries into a DataFrame
    umapScanMetaData = pd.DataFrame(data_list)
    for condition in uMAPConditions:
        filtered_df = umapScanMetaData[umapScanMetaData["neighborNumber"] == condition]
        save_umap_jpg(filtered_df, outputGraphFolder, userInputList, "group", condition)

    metaDataSplits = umapScanMetaData["image_name"].str.split("_", expand = True)
    columnNames = umapScanMetaData["image_name"].iloc[0].split('_')
    metaDataSplits.columns = columnNames
    umapScanMetaData = pd.concat([umapScanMetaData, metaDataSplits], axis = 1)
    # for col in umapScanMetaData.columns:
    #     if col == "control" or userInputList.conditionStr in col.lower():
    #         umapScanMetaData = umapScanMetaData.rename(columns={col: "condition"})

    finalMetaDataPath = dataFrameFolder.joinpath("umap_neighbor_scan.xlsx")
    umapScanMetaData.to_excel(finalMetaDataPath)


    reducer = UMAP(n_neighbors=userInputList.numberOfNeighbors, n_components=2, n_epochs=1000, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(normScaledFreqVectors)
        
    for i, imageName in enumerate(multiScaleVectors.keys()):
        multiScaleVectors[imageName] = {
            "PCAx": pca_result[i, 0],
            "PCAy": pca_result[i, 1],
            "umapx": embedding[i, 0], # type: ignore
            "umapy": embedding[i, 1], # type: ignore
            freqVectorChoice: multiScaleVectors[imageName][freqVectorChoice]  
        }
          
    data_list = []
    for i, (image_name, image_data) in enumerate(multiScaleVectors.items()):
        name_lower = image_name.lower()
        splits = name_lower.split("_")
        groupStr = userInputList.get_group_name(Path(image_name).stem)
        image_entry = {
            "image_name": image_name,
            "PCAx": pca_result[i, 0],
            "PCAy": pca_result[i, 1],
            "umapx": embedding[i, 0], # type: ignore
            "umapy": embedding[i, 1], # type: ignore
            "group": groupStr,
        }    
        # Add normalized frequency vector as separate columns
        if freqVectorChoice not in image_data:
            print(f"Missing key '{freqVectorChoice}' for image: {image_name}")
            print(f"Available keys: {image_data.keys()}")
            raise KeyError(f"{freqVectorChoice} not found in image_data for {image_name}")

        for idx, value in enumerate(image_data[freqVectorChoice]):
            image_entry[f"word_{idx}"] = value
        
        data_list.append(image_entry)

    updatedMetaData = pd.DataFrame(data_list)
    metaDataSplits = updatedMetaData["image_name"].str.split("_", expand = True)
    columnNames = updatedMetaData["image_name"].iloc[0].split('_')
    metaDataSplits.columns = columnNames
    updatedMetaData = pd.concat([updatedMetaData, metaDataSplits], axis = 1)

    # for col in updatedMetaData.columns:
    #     if col.lower() == "control" or col.lower() == userInputList.conditionStr.lower():
    #         updatedMetaData = updatedMetaData.rename(columns={col: "condition"})

    updatedMetaData.to_csv(dataFrameFolder.joinpath("PCA_Frequency_Vector_Data_sparse.csv"), index=False)
    save_umap_jpg(updatedMetaData, outputGraphFolder, userInputList, "group", userInputList.numberOfNeighbors)

    finalMetaDataPath = saveFolder.joinpath(f"{sampleNumber}files_{userInputList.dictionarySize}_words_ch{userInputList.chToClassify}_{userInputList.numberOfNeighbors}_n_sparse.xlsx") 

    updatedMetaData.to_excel(finalMetaDataPath)

    computeClusterPurity(multiScaleVectors, saveFolder, "normalizedFrequencyVector", userInputList)

    return finalMetaDataPath




def calcVolumeFromList(nameList, chCellCropLocation, userInputList):
    import numpy as np
    from skimage.io import imread
    nuclearVolumeList = []
    maskString = userInputList.classMaskChoice
    maskList = list(chCellCropLocation.rglob(f"*{maskString}.tif"))

    for name in nameList:
        tmpCellString = name.rsplit("_",1)[0]
        for maskName in maskList:
            tmpMaskString = maskName.stem.rsplit("_",1)[0]
            if tmpCellString == tmpMaskString:
                cellMask = maskName
                break
        maskCrop = imread(cellMask)
        nuclearVol = np.count_nonzero(maskCrop)
        nuclearVolumeList.append(nuclearVol)

    return nuclearVolumeList




def computeClusterPurity(multiScaleVectors, saveFolder, variableToCluster, userInputList):
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  import numpy as np
  from collections import Counter, defaultdict 
  import pandas as pd
  from pathlib import Path

  print("Computing cluster purity...")
  dataFrameFolder = saveFolder.joinpath("dataframes")
  dataFrameFolder.mkdir(parents = True, exist_ok = True)
  

  image_names = list(multiScaleVectors.keys())
  feature_vectors = np.stack([multiScaleVectors[img][variableToCluster] for img in image_names])

  num_clusters = [2, 3, 4, 5, 6]
  all_results = []
  for clusterNumber in num_clusters:
    kmeans = KMeans(n_clusters=clusterNumber, random_state=42, n_init="auto")    
    cluster_labels = kmeans.fit_predict(feature_vectors) 
    score = silhouette_score(feature_vectors, cluster_labels)
# Group images by their cluster assignment
    cluster_groups = defaultdict(list)
    for img_name, cluster_label in zip(image_names, cluster_labels):
        cluster_groups[cluster_label].append(img_name)

    # For each cluster, compute purity
    cluster_results = []
    for cluster_id, img_list in cluster_groups.items():
        # Extract conditions from filenames
      label_list = []
      for name in img_list:
          name_lower = name.lower()
          groupStr = userInputList.get_group_name(Path(name_lower).stem)
          label_list.append(groupStr)

      label_counts = Counter(label_list)
      total = sum(label_counts.values())
      dominant_label = label_counts.most_common(1)[0][0]
      purity = label_counts[dominant_label] / total

      # Break down individual counts per group
      breakdown = {label: count for label, count in label_counts.items()}

      cluster_results.append({
                "num_clusters": clusterNumber,
                "cluster_id": cluster_id,
                "purity": purity,
                "dominant_label": dominant_label,
                "total_images": total,
                **breakdown,  # unpack label counts as individual columns
                "image_list": img_list,
                "silhouetteScore": score
            })

    # Save result for this clustering
    all_results.extend(cluster_results)
      
  # Save to CSV (summary table, no image lists)
  df = pd.DataFrame(all_results)

  image_list_idx = df.columns.get_loc("image_list")
  condition_cols = df.columns[5:image_list_idx]
  df[condition_cols] = df[condition_cols].fillna(0).astype(int)
  

  resultsSaveName = dataFrameFolder.joinpath(f"cluster_summary_{variableToCluster}.xlsx")
  df.to_excel(resultsSaveName, index=False)


def save_umap_jpg(filtered_df, outpath, userInputList, colorStr, nNeighbors):
    from pathlib import Path
    import plotly.express as px

    colorMap = {"Control_cgn": "#3b82f6", 
                "lof_cgn": "#ef4444"}

    fig = px.scatter(
      filtered_df,
      x="umapx",
      y="umapy",
      color=colorStr,
      symbol=colorStr,
      color_discrete_map = colorMap,
      hover_data=["image_name"],
      custom_data=["image_name"])

    fig.update_traces(
        marker=dict(size=8))

    fig.update_layout( 
     title=dict(
        text=f"UMAP with {nNeighbors} neighbors for channel {userInputList.chToClassify} with {userInputList.normalizeMethod} normalization",
        x=0.5,              
        xanchor="center"),
      xaxis=dict(showgrid=True),
      yaxis=dict(showgrid=True),
      height=500,
      width=700
  )
    normalizeStr = userInputList.normalizeMethod.replace(" ", "_")
    fileName = f"umap_{nNeighbors}_neighbors_ch{userInputList.chToClassify}_{normalizeStr}_normalization.jpg"
    fig.write_image(outpath / fileName, format="jpg")




def runHarralick3D(allImageDictionary, saveFolder, userInputList):
    """
    Go through image dictionary, extract each image, go patch by patch and calculate the Harralick features. 
    Takes in:
    allImageDictionary: overall image dictionary
    saveFolder: folder that contains images/graphs/etc
    userInputList: user parameters
    Each patch will return a list of Harlick numbers, add back to original dictionary
    
    """

    import numpy as np
    import pandas as pd
    from skimage.io import imread, imsave
    print("Looking for Haralick Features now")
    ##---Set up file locations--##
    chFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
    heatmapFolder = saveFolder / "attention_maps"
    harFeatureFolder = saveFolder / "haralickFeatureMaps"
    harFeatureFolder.mkdir(exist_ok = True)
    nLevels = 32    
    attnThreshold = 0.50
    # which Haralick features you want full 3D maps for
    features_for_maps = [0,1,2,3,4, 5, 6,7,8,9,10,11,12]
    # this will hold all maps for all images
    har_maps_by_image = {}
    
    ##--Main loop through dictionary keys--##
    for imageName, imageData in allImageDictionary.items():
        # print(f"working on {imageName}")
        tmpImg = imread(chFolder / imageName).astype(np.float32, copy=False)
        imgShape = tmpImg.shape
        tmpMaskName = imageName.replace("ch1", "CellposeMask")
        tmpMask = imread(chFolder / tmpMaskName)
        condition = "control" if "control" in imageName.lower() else "lof"
        tmpAttnMapName = imageName.replace(".tif","") + "_attn_" + condition + ".tif"
        tmpHarImgName6 = imageName.replace(".tif","") + "_harFeature6.tif"
        tmpHarImgName3 = imageName.replace(".tif","") + "_harFeature3.tif"
        tmpHarImgName5 = imageName.replace(".tif","") + "_harFeature5.tif"

        tmpAttnMap = imread(heatmapFolder / tmpAttnMapName)
        listOfFeatures = []
        harlickMapImg6 = np.zeros_like(tmpImg)
        harlickMapImg3 = np.zeros_like(tmpImg)
        harlickMapImg5 = np.zeros_like(tmpImg) 

        valid = tmpMask.astype(bool)            
        vals = tmpImg[valid]
        # robust range (you can tweak these percentiles)
        lo, hi = np.percentile(vals, [1, 99])
        vol_clipped = np.clip(tmpImg, lo, hi)
        vol_norm = (vol_clipped - lo) / (hi - lo)
        # map to 1..(n_levels-1) for foreground
        tmpImgQuan = np.zeros_like(tmpImg, dtype=np.uint8)
        tmpImgQuan[valid] = np.floor(vol_norm[valid] * (nLevels - 1e-3)).astype(np.uint8) + 1

        # --- allocate per-feature maps for THIS image ---
        har_maps = {
            f_idx: np.zeros_like(tmpImg, dtype=np.float32)
            for f_idx in features_for_maps
        }
        
        for patchLocation, patchSize in zip(imageData["location"], imageData["patch_size"]):
            z0, z1, y0, y1, x0, x1 = patch_bbox(patchLocation, patchSize, imgShape)
            tmpPatch = tmpImgQuan[z0:z1, y0:y1, x0:x1]
            tmpAttn = tmpAttnMap[z0:z1, y0:y1, x0:x1]
            if np.mean(tmpAttn) > attnThreshold:
                harFeature = haralick_3d_mahotas(tmpPatch, distance=1, return_mean=True)
                listOfFeatures.append(harFeature)
                for f_idx in features_for_maps:
                    har_maps[f_idx][z0:z1, y0:y1, x0:x1] = harFeature[f_idx]
            else:
                harFeature = np.nan
                listOfFeatures.append(harFeature)
            
        allImageDictionary[imageName]["haralickFeatures"] = listOfFeatures
        har_maps_by_image[imageName] = har_maps
        # imsave(harFeatureFolder / tmpHarImgName6, harlickMapImg6)
        # imsave(harFeatureFolder / tmpHarImgName3, harlickMapImg3)
        # imsave(harFeatureFolder / tmpHarImgName5, harlickMapImg5)
    
    ##--Build plots to show difference between conditions ---###
    har_df = build_haralick_image_summary(allImageDictionary)
    har_df.to_excel(saveFolder / "dataframes" / f"haralickFeaturesInfo_{attnThreshold}.xlsx")
    plot_haralick_violin_combined(har_df, saveFolder, attnThreshold)
    analyzeHarlickEDT(har_maps_by_image, saveFolder, userInputList)
    haralickFeatureList = [(0,"Angular second moment (ASM / “energy”)"),(1,"Contrast"),(2,"Correlation"),(3,"Sum of squares (variance)"),(4,"Inverse difference moment (homogeneity)"),(5,"Sum average"),(6,"Sum variance"),(7,"Sum entropy"),(8,"Entropy"),(9, "Difference variance"),(10,"Difference entropy"),(11,"Information measure of correlation 1"),(12,"Information measure of correlation 2")]
    for featureIndex, featureName in haralickFeatureList:
        plot_haralick_feature_kde(allImageDictionary, saveFolder, feature_idx=featureIndex, feature_name=featureName)



def plot_haralick_feature_kde(allImageDictionary, saveFolder,
                              feature_idx=6, feature_name="Haralick 6"):
    """
    Plot KDEs of a single Haralick feature across *patches*,
    separated by condition (control vs LOF).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import gaussian_kde
    import pandas as pd

    colors = {"control": "#3b82f6", "lof": "#ef4444"}

    ctrl_vals, lof_vals = collect_patch_haralick_feature(allImageDictionary, feature_idx)

    print(f"[KDE] Feature {feature_idx}: {len(ctrl_vals)} control patches, {len(lof_vals)} LOF patches")

    if len(ctrl_vals) < 2 or len(lof_vals) < 2:
        print("[WARN] Not enough patches in one or both conditions for KDE (need >=2 each).")
        return

    # Common x-range with a little padding
    x_min = float(min(ctrl_vals.min(), lof_vals.min()))
    x_max = float(max(ctrl_vals.max(), lof_vals.max()))
    pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    xs = np.linspace(x_min - pad, x_max + pad, 512)

    kde_ctrl = gaussian_kde(ctrl_vals)
    kde_lof  = gaussian_kde(lof_vals)

    ys_ctrl = kde_ctrl(xs)
    ys_lof  = kde_lof(xs)

    # ---------- SAVE x,y VALUES TO EXCEL ----------
    dataframeFolder = Path(saveFolder) / "dataframes"

    df_kde = pd.DataFrame({
        "x": xs,
        "density_control": ys_ctrl,
        "density_lof": ys_lof
    })

    excel_path = dataframeFolder / f"haralick_feature{feature_idx}_patchKDE_data.xlsx"
    df_kde.to_excel(excel_path, index=False)
    # ------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(xs, ys_ctrl, label=f"control (n={len(ctrl_vals)})", lw=2, alpha=0.9, color = colors["control"])
    ax.plot(xs, ys_ctrl, lw=2, alpha=0.9, color = colors["control"])
    ax.fill_between(xs, ys_ctrl, alpha=0.2, color = colors["control"])

    ax.plot(xs, ys_lof, label=f"lof (n={len(lof_vals)})", lw=2, alpha=0.9, color = colors["lof"])
    ax.plot(xs, ys_lof, lw=2, alpha=0.9, color = colors["lof"])
    ax.fill_between(xs, ys_lof, alpha=0.2, color = colors["lof"])

    ax.set_xlabel(f"{feature_name} value (feature index {feature_idx})")
    ax.set_ylabel("Density (KDE over patches)")
    ax.set_title(f"KDE of Haralick feature {feature_idx} across patches")
    ax.legend()
    ax.grid(alpha=0.3)

    out_dir = Path(saveFolder) / "outputGraphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"haralick_feature{feature_idx}_patchKDE.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved patch-level KDE for feature {feature_idx} to {out_path}")



def analyzeHarlickEDT(har_maps_by_image, saveFolder, userInputList):
    import numpy as np
    import pandas as pd
    from skimage.io import imread

    edtImgFolder     = saveFolder / "attention_maps"
    outputGraphFolder = saveFolder / "outputGraphs"
    dataframeFolder   = saveFolder / "dataframes"

    harlickFeatureList = [0,1,2,3,4, 5, 6,7,8,9,10,11,12]
    edtLevelsList      = [1, 2, 3, 4, 5]

    for harFeature in harlickFeatureList:
        rows = []

        # loop over images we actually computed maps for
        for imageName, feature_maps in har_maps_by_image.items():
            if harFeature not in feature_maps:
                continue  # in case you only built maps for a subset

            tmpHarImg  = feature_maps[harFeature]
            tmpImgName = imageName.replace(".tif", "")
            tmpEDTLevelsNameList = [f"{tmpImgName}_1_levels.tif", f"{tmpImgName}_2_levels.tif"]
            for edtLevel in tmpEDTLevelsNameList:
                tmpEDTImg = imread(edtImgFolder / edtLevel)
                common_meta = {
                    "image_name": tmpImgName,
                    "condition": "control" if "control" in tmpImgName.lower() else "lof",
                    "chromatinSpecies": "heterochromatin" if "_1_levels" in edtLevel else "euchromatin",
                    "haralickFeature": harFeature}
                
                for level in edtLevelsList:
                    validMask = (tmpEDTImg == level)
                    numVoxelsinLevel = np.count_nonzero(tmpEDTImg[validMask])
                    sumHarFeatureInLevel = np.sum(tmpHarImg[validMask])
                    avgHarFeature = sumHarFeatureInLevel / numVoxelsinLevel

                    rows.append({**common_meta,
                        "edtValue": f"level_{level}",
                        "numVoxels": numVoxelsinLevel,
                        "sumHarValue": sumHarFeatureInLevel,
                        "avgHarValue": avgHarFeature
                    })
        
        fullData = pd.DataFrame(rows) 
        fullData["edtLevel"] = fullData["edtValue"].str.split("_").str[-1].astype("Int64")
        fullData["edtLevelSigned"] = fullData["edtLevel"].where(fullData["chromatinSpecies"].eq("heterochromatin"), -fullData["edtLevel"])  
        fullData.to_excel(dataframeFolder / f"harlickFeatureEDTLevels_feature{harFeature}.xlsx")
        
        is_control = fullData[fullData["condition"].astype(str).str.contains("control", case=False, na=False)]
        is_lof     = fullData[fullData["condition"].astype(str).str.contains("lof", case=False, na=False)]

        plot_grouped_violins_by_level(
        is_control,
        is_lof,
        level_col = "edtLevelSigned",
        value_col = "avgHarValue",
        outfile = outputGraphFolder / f"avgHarValue_vs_edtZone_{harFeature}Feature.jpg",
        control_color="#FF007F",
        lof_color="#008080",
        dpi = 300,
        plotTitle= f"avg value for harlick feature {harFeature} by signed EDT level (grouped violins)")




def analyzeAttention(saveFolder, userInputList, allImagesCellFeatureDict, codebook):

    analyzeAttentionMaps(saveFolder, userInputList)
    compareAttentionIntensity(userInputList, saveFolder)
    # pearsonCorrelationIntAttn(userInputList, saveFolder)
    # calcErrorPerPatchAndAnalyze(allImagesCellFeatureDict, codebook, saveFolder, userInputList)



def pearsonCorrelationIntAttn(userInputList, saveFolder):   
    from skimage.io import imread, imsave
    import numpy as np
    import pandas as pd
    import plotly.io as pio
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    print("Running Pearson's Correlation now...")
   
    ##-- File locations ---##
    heatmapFolder = saveFolder / "attention_maps"
    chCropFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
   
    imageSaveFolder = saveFolder.joinpath("outputGraphs")
    imageSaveFolder.mkdir(parents = True, exist_ok = True)
    dataFrameFolder = saveFolder.joinpath("dataframes")
    dataFrameFolder.mkdir(parents = True, exist_ok = True)
    correFolder = saveFolder / "correlationMaps"
    correFolder.mkdir(parents = True, exist_ok = True)

    heatmapNames = heatmapFolder.glob("*.tif")
    badStrList = ["blob", "levels", "raw", "patches", "edt"]
    refinedNames = [namePath for namePath in heatmapNames if not any(badStr in namePath.name.lower() for badStr in badStrList)]
    all_img_rows = []

    for heatmapName in refinedNames:
        tmpAttentionMap = imread(heatmapName)
        tmpImgName = heatmapName.stem.rsplit("_attn_")[0] + ".tif"
        tmpCellposeName = tmpImgName.rsplit(f"ch{userInputList.chToClassify}")[0] + "CellposeMask.tif"
        tmpCorreName = heatmapName.stem.rsplit("_attn_")[0] +"_correlation.tif"
        tmpCellposeMask = imread(chCropFolder / tmpCellposeName)
        tmpImg = imread(chCropFolder / tmpImgName)

        label = "control" if "control" in tmpImgName.lower() else "lof"

        locCor = local_corr_zyx(tmpAttentionMap, tmpImg, win=(3, 9, 9), shift=(0, 0, 0),
                   nucleus_mask=tmpCellposeMask, mode='reflect', ddof=0,
                   return_masked=True, min_support_frac=0.125)
        imsave(correFolder / tmpCorreName, locCor)

    compareCorrelationEDTZones(userInputList, saveFolder)


def local_corr_zyx(attn, inten, win=(3, 9, 9), shift=(0, 0, 0),
                   nucleus_mask=None, mode='reflect', ddof=0,
                   return_masked=True, min_support_frac=0.125):
    """
    Local Pearson correlation between attention and intensity volumes.

    Parameters
    ----------
    attn : (Z,Y,X) array-like, float
        Attention in [-1, 1]. Convention: attn == 0 marks voxels OUTSIDE the nucleus
        (unless you provide nucleus_mask).
    inten : (Z,Y,X) array-like, float
        Raw intensity (any positive range). May be non-zero outside the nucleus.
    win : (wz, wy, wx) int tuple
        3D sliding window size along (Z, Y, X). Use odd sizes for symmetry, e.g., (3,9,9).
    shift : (dz, dy, dx) int tuple
        Spatial offset applied to the intensity before correlation (e.g., (0,0,0) for none).
    nucleus_mask : (Z,Y,X) bool, optional
        If provided, True=inside-nucleus voxels to include. If None, uses (attn != 0).
        Prefer passing your explicit mask if zeros can appear inside the nucleus.
    mode : str
        Padding mode for uniform_filter. 'reflect' avoids edge bias; avoid 'constant' unless needed.
    ddof : int
        0 for population variance/covariance; 1 for sample versions.
    return_masked : bool
        If True, set correlation outside the nucleus to np.nan in the output.
    min_support_frac : float in (0,1]
        Minimum fraction of the full window that must be inside-nucleus to report a value.
        Voxels with fewer valid samples are set to NaN.

    Returns
    -------
    corr : (Z,Y,X) float32
        Local correlation in [-1, 1]. NaN outside nucleus (if return_masked) or where support is too small.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    A = np.asarray(attn, dtype=np.float32)
    I = np.asarray(inten, dtype=np.float32)

    if A.shape != I.shape or A.ndim != 3:
        raise ValueError("attn and inten must be 3D with the same shape (Z,Y,X).")

    # Inside-nucleus mask
    if nucleus_mask is None:
        M = (A != 0)
    else:
        M = np.asarray(nucleus_mask, dtype=bool)
        if M.shape != A.shape:
            raise ValueError("nucleus_mask must match (Z,Y,X).")

    # Apply spatial shift to intensity (correlate A(z,y,x) with I(z+dz, y+dy, x+dx))
    dz, dy, dx = shift
    if dz or dy or dx:
        I = np.roll(np.roll(np.roll(I, dz, axis=0), dy, axis=1), dx, axis=2)

    # Zero-out invalid voxels before filtering; count valid voxels per window separately
    A0 = np.where(M, A, 0.0)
    I0 = np.where(M, I, 0.0)

    wz, wy, wx = win
    size = (wz, wy, wx)
    window_vol = int(np.prod(size))
    eps = 1e-7

    # Valid counts per window (corrects edge effects and masking)
    cnt = uniform_filter(M.astype(np.float32), size=size, mode=mode) * window_vol # pyright: ignore[reportArgumentType]

    # Local sums
    sumA  = uniform_filter(A0,        size=size, mode=mode) * window_vol # type: ignore
    sumI  = uniform_filter(I0,        size=size, mode=mode) * window_vol # type: ignore
    sumAA = uniform_filter(A0 * A0,   size=size, mode=mode) * window_vol # type: ignore
    sumII = uniform_filter(I0 * I0,   size=size, mode=mode) * window_vol # type: ignore
    sumAI = uniform_filter(A0 * I0,   size=size, mode=mode) * window_vol # type: ignore

    # Means
    safe_cnt = np.maximum(cnt, 1.0)
    meanA = sumA / safe_cnt
    meanI = sumI / safe_cnt

    # Variances and covariance (handles varying support per window)
    denom = np.maximum(cnt - ddof, 1.0)
    varA = (sumAA - (sumA * sumA) / safe_cnt) / denom
    varI = (sumII - (sumI * sumI) / safe_cnt) / denom
    covAI = (sumAI - (sumA * sumI) / safe_cnt) / denom

    # Pearson r
    corr = covAI / (np.sqrt(varA * varI) + eps)
    corr = np.clip(corr, -1.0, 1.0).astype(np.float32)

    # Suppress poorly supported windows
    min_support = max(1, int(round(min_support_frac * window_vol)))
    corr[cnt < min_support] = np.nan

    # Optional: hide results outside nucleus
    if return_masked:
        corr[~M] = np.nan

    return corr



def compareCorrelationEDTZones(userInputList, saveFolder):
    import pandas as pd
    import numpy as np  
    from skimage.io import imread, imsave

    ###-- File Locations ---###
    chToFind = userInputList.chToClassify
    imageFolder = saveFolder / f"ch{chToFind}Crops"
    heatmapFolder = saveFolder / "attention_maps"
    correFolder = saveFolder / "correlationMaps"
    dataFrameFolder = saveFolder / "dataframes"
    outputGraphFolder = saveFolder / "outputGraphs"
    
    rows = []
    strMatchLevel = ["*_1_levels.tif", "*_2_levels.tif"]
    fileList = correFolder.glob("*.tif")
    thresholdValue = 0.2

    ##-- Go through each file and find average correlation per level --###
    for file in fileList:
        tmpFile = imread(file)
        for strM in strMatchLevel:
            edtLevelName = file.stem.rsplit("_correlation")[0] + strM
            tmpEDTImage = imread(heatmapFolder / edtLevelName)
            for edtValue in np.unique(tmpEDTImage[tmpEDTImage != 0]):
                mask = (tmpEDTImage == edtValue)
                region = tmpFile[mask]        
                posRegionOnly = region[region > 0]  
                posAboveThreshold = posRegionOnly[posRegionOnly > thresholdValue]        
                numVoxels = int(mask.sum())
                avgIntensity = posAboveThreshold.sum() / numVoxels
                coverage = np.count_nonzero(posAboveThreshold) / numVoxels

                rows.append({
                    "image_name": file.stem,
                    "condition": "control" if "control" in file.stem.lower() else "lof",
                    "chromatinSpecies":"heterochromatin" if "1" in strM else "euchromatin",
                    "edtValue": f"level_{edtValue}",
                    "numVoxels": numVoxels,
                    "avgCorrelation": avgIntensity,
                    "coverageAboveThreshold": coverage,
                    "threshold": thresholdValue
                })

    fullData = pd.DataFrame(rows)
    fullData["edtLevel"] = fullData["edtValue"].str.split("_").str[-1].astype("Int64")
    fullData["edtLevelSigned"] = fullData["edtLevel"].where(
        fullData["chromatinSpecies"].eq("heterochromatin"), -fullData["edtLevel"])    
    fullData.to_excel(dataFrameFolder / "correlation_vs_edtZone.xlsx")

    is_control = fullData[fullData["condition"].astype(str).str.contains("control", case=False, na=False)]
    is_lof     = fullData[fullData["condition"].astype(str).str.contains("lof", case=False, na=False)]


    plot_grouped_violins_by_level(
    is_control,
    is_lof,
    level_col = "edtLevelSigned",
    value_col = "avgCorrelation",
    outfile = outputGraphFolder / "avgCorrelation_vs_edtZone.jpg",
    control_color="#FF007F",
    lof_color="#008080",
    dpi = 300,
    plotTitle= "avgCorrelation by signed EDT level (grouped violins)")

    plot_grouped_violins_by_level(
    is_control,
    is_lof,
    level_col = "edtLevelSigned",
    value_col = "coverageAboveThreshold",
    outfile = outputGraphFolder / "correlationCoverage_vs_edtZone.jpg",
    control_color="#FF007F",
    lof_color="#008080",
    dpi = 300,
    plotTitle= f"Coverage [above {thresholdValue}] by signed EDT level (grouped violins)")



def plot_grouped_violins_by_level(
    is_control,
    is_lof,
    level_col: str,
    value_col: str,
    plotTitle: str,
    outfile,
    control_color="#FF007F",
    lof_color="#008080",
    dpi: int = 300,    
):
    """
    Grouped violins per EDT level: one rose (Control) and one teal (LOF).
    Requires >=2 samples to draw a violin for a given condition/level.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd

    # Keep only needed columns and coerce to numeric
    def _clean(df):
        sub = df[[level_col, value_col]].copy()
        sub[level_col] = pd.to_numeric(sub[level_col], errors="coerce")
        sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
        return sub.dropna()

    c = _clean(is_control)
    l = _clean(is_lof)

    # All levels present in either condition
    levels = sorted(pd.unique(pd.concat([c[level_col], l[level_col]], ignore_index=True)))
    if len(levels) == 0:
        print("[WARN] No levels to plot.")
        return

    # Data arrays per level for each condition
    c_data = [c.loc[c[level_col] == lvl, value_col].values for lvl in levels]
    l_data = [l.loc[l[level_col] == lvl, value_col].values for lvl in levels]

    # Prepare positions (two offsets per level)
    base = np.arange(1, len(levels) + 1, dtype=float)
    offset = 0.15
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    dropped_info = {"control": [], "lof": []}

    # Helper to draw one side
    def _draw(data_list, positions, color, label_name, key):
        # Only draw violins for levels with >=2 samples
        kept_data, kept_pos, kept_lvls = [], [], []
        for lvl, arr, pos in zip(levels, data_list, positions):
            if len(arr) >= 2:
                kept_data.append(arr)
                kept_pos.append(pos)
                kept_lvls.append(lvl)
            else:
                dropped_info[key].append(lvl)
        if kept_data:
            parts = ax.violinplot(
                kept_data,
                positions=kept_pos,
                widths=width,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for b in parts["bodies"]: # pyright: ignore[reportGeneralTypeIssues]
                b.set_facecolor(color)
                b.set_edgecolor("black")
                b.set_alpha(0.7)
            if "cmedians" in parts and parts["cmedians"] is not None:
                parts["cmedians"].set_linewidth(2.0)
            # Legend proxy
            ax.scatter([], [], color=color, label=label_name)

    _draw(c_data, base - offset, control_color, "Control", "control")
    _draw(l_data, base + offset, lof_color, "LOF", "lof")

    # Axes / labels
    ax.set_xticks(base)
    ax.set_xticklabels([str(lvl) for lvl in levels])
    ax.set_xlabel("EDT level (signed)")
    ax.set_ylabel(value_col)
    ax.set_title(plotTitle)
    ax.legend(title="Condition")
    fig.tight_layout()

    # Save
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Logging of dropped levels (optional)
    if dropped_info["control"]:
        print(f"[INFO] Control: dropped levels (<2 samples): {sorted(set(dropped_info['control']))}")
    if dropped_info["lof"]:
        print(f"[INFO] LOF: dropped levels (<2 samples): {sorted(set(dropped_info['lof']))}")
    print(f"[OK] Saved {outfile}")




def compareAttentionIntensity(userInputList, saveFolder):
    ###---Compare scaled attention maps [from LR] to normalized intensity of probe--- ###

    from skimage.io import imread, imsave
    import numpy as np
    import pandas as pd
    import plotly.io as pio
    chrome_path = pio.get_chrome() # pyright: ignore[reportAttributeAccessIssue]
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
   
    ##-- File locations ---##
    heatmapFolder = saveFolder / "attention_maps"
    chCropFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
   
    imageSaveFolder = saveFolder.joinpath("outputGraphs")
    imageSaveFolder.mkdir(parents = True, exist_ok = True)
    dataFrameFolder = saveFolder.joinpath("dataframes")
    dataFrameFolder.mkdir(parents = True, exist_ok = True)

    heatmapNames = heatmapFolder.glob("*.tif")
    badStrList = ["blob", "levels", "raw", "patches", "edt"]
    refinedNames = [namePath for namePath in heatmapNames if not any(badStr in namePath.name.lower() for badStr in badStrList)]

    ##-- Parameters --##
    attnBins = 30
    attnRange = (-1,1)
    intBins = 30
    intRange = (0,1)
    # accumulator for combined histogram
    H_class = {
        "control": np.zeros((intBins, attnBins), dtype=np.float64),
        "lof":     np.zeros((intBins, attnBins), dtype=np.float64),
    }

    for heatmapName in refinedNames:
        tmpAttentionMap = imread(heatmapName)
        tmpImgName = heatmapName.stem.rsplit("_attn_")[0] + ".tif"
        label = "control" if "control" in tmpImgName.lower() else "lof"
        tmpImg = imread(chCropFolder / tmpImgName)
        low, high = np.percentile(tmpImg, (5, 95))
        tmpNormImage = np.clip((tmpImg - low) / (high - low), 0, 1)
        tmpNorm50 = np.percentile(tmpNormImage, 50)
        mask50p = (tmpNormImage > tmpNorm50)

        intensity1D = tmpNormImage[mask50p].ravel()  # intensity
        attention1D = tmpAttentionMap[mask50p].ravel()      # attention

        # 2D histogram
        H, intedges, attedges = np.histogram2d(
            intensity1D, attention1D,
            bins=(intBins, attnBins),
            range=(intRange, attnRange)
        )
        H_class[label] += H

        # --- Plot and save per class ---
    for label, H in H_class.items():

        # Heatmap (log counts)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[intRange[0], intRange[1], attnRange[0], attnRange[1]], # pyright: ignore[reportArgumentType]
            norm=mcolors.LogNorm(vmin=1, vmax=max(1, H.max()))
        )
        ax.set_xlabel("Normalized intensity (5–95% scaled)")
        ax.set_ylabel("Attention")
        ax.set_title(f"2D Histogram: Intensity vs Attention ({label}) for values > median")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Count (log scale)")
        fig.savefig(imageSaveFolder / f"{label}_intensity_vs_attention.png", bbox_inches="tight")
        plt.close(fig)

        # Save raw histogram and edges for downstream analysis
        np.savez_compressed(
            dataFrameFolder / f"{label}_intensity_vs_attention_2dhist.npz",
            H=H,
            xedges=np.linspace(intRange[0], intRange[1], intBins + 1),
            yedges=np.linspace(attnRange[0], attnRange[1], attnBins + 1)
        )



def calcErrorPerPatchAndAnalyze(allImagesCellFeatureDict, codebook, saveFolder, userInputList):
    
    allImagesCellFeatureDict = compute_patch_residuals(allImagesCellFeatureDict, codebook, calc_relative=True)
    allImagesCellFeatureDict = compute_patch_recon_errors(allImagesCellFeatureDict, codebook, metrics =  "rel_mse", eps=1e-8)
    make_reconstruction_error_map(allImagesCellFeatureDict, saveFolder, userInputList,
                                  key = "recon_rel_mse_per_patch", aggregator="max")
    summarize_residuals_vs_attention(saveFolder, userInputList)
    calcResidualsVsEDT(saveFolder, userInputList)


def compute_patch_residuals(allImagesCellFeatureDict, sparse_dictionary, calc_relative=True, eps=1e-8):
    """
    Adds per-patch residual arrays to each image dict:
      - imageData["recon_error_per_patch"]  : shape (n_patches,)
      - imageData["recon_error_per_patch_rel"] : relative residual (optional)
    Requires keys: "feature" (list/array NxF), "sparseCodes" (NxK).
    """
    import numpy as np

    D = np.asarray(sparse_dictionary)  # (K, F)
    for imageName, imageData in allImagesCellFeatureDict.items():
        X = np.asarray(imageData["feature"], dtype=np.float64)       # (N, F)
        C = np.asarray(imageData["sparseCodes"], dtype=np.float64)   # (N, K)
        X_hat = C @ D                                                # (N, F)
        resid = np.sum((X - X_hat)**2, axis=1)                       # (N,)
        imageData["recon_error_per_patch"] = resid.astype(np.float32)

        if calc_relative:
            denom = np.sum(X**2, axis=1) + eps
            rel = resid / denom
            imageData["recon_error_per_patch_rel"] = rel.astype(np.float32)
    return allImagesCellFeatureDict


def compute_patch_recon_errors(
    allImagesCellFeatureDict, 
    sparse_dictionary, 
    metrics=("sse", "mse", "rmse", "rel_mse", "rel_rmse"),
    eps=1e-8
):
    """
    Adds selected per-patch error arrays to each image dict.
    Keys written (if requested in `metrics`):
      - recon_sse_per_patch        : sum_j (x_ij - xhat_ij)^2
      - recon_mse_per_patch        : recon_sse / F
      - recon_rmse_per_patch       : sqrt(recon_mse)
      - recon_rel_mse_per_patch    : recon_sse / (||x_i||^2 + eps)
      - recon_rel_rmse_per_patch   : ||x_i - xhat_i|| / (||x_i|| + eps)
      - recon_mae_per_patch        : mean_j |x_ij - xhat_ij|
      - recon_cosine_dist_per_patch: 1 - cos_sim(x_i, xhat_i)
    """
    import numpy as np

    D = np.asarray(sparse_dictionary, dtype=np.float64)  # (K, F)
    for imageName, imageData in allImagesCellFeatureDict.items():
        X = np.asarray(imageData["feature"], dtype=np.float64)       # (N, F)
        C = np.asarray(imageData["sparseCodes"], dtype=np.float64)   # (N, K)
        X_hat = C @ D                                                # (N, F)
        N, F = X.shape

        R = X - X_hat
        sse = np.einsum('ij,ij->i', R, R)              # (N,)
        x_norm2 = np.einsum('ij,ij->i', X, X)          # (N,)
        r_norm = np.sqrt(sse)                          # (N,)
        x_norm = np.sqrt(x_norm2 + eps)

        if "sse" in metrics:
            imageData["recon_sse_per_patch"] = sse.astype(np.float32)
        if "mse" in metrics:
            imageData["recon_mse_per_patch"] = (sse / F).astype(np.float32)
        if "rmse" in metrics:
            imageData["recon_rmse_per_patch"] = (r_norm / np.sqrt(F)).astype(np.float32)
        if "rel_mse" in metrics:
            imageData["recon_rel_mse_per_patch"] = (sse / (x_norm2 + eps)).astype(np.float32)
        if "rel_rmse" in metrics:
            imageData["recon_rel_rmse_per_patch"] = (r_norm / x_norm).astype(np.float32)
        if "mae" in metrics:
            imageData["recon_mae_per_patch"] = (np.mean(np.abs(R), axis=1)).astype(np.float32)
        if "cosine" in metrics or "cosine_dist" in metrics:
            # cosine distance = 1 - cosine similarity
            xhat_norm = np.sqrt(np.einsum('ij,ij->i', X_hat, X_hat) + eps)
            cos_sim = (np.einsum('ij,ij->i', X, X_hat) / (x_norm * xhat_norm + eps))
            imageData["recon_cosine_dist_per_patch"] = (1.0 - cos_sim).astype(np.float32)

    return allImagesCellFeatureDict



def make_reconstruction_error_map(allImagesCellFeatureDict, saveFolder, userInputList,
                                  key = "recon_rel_mse_per_patch", aggregator="max",
                                  write_float32=True, folder_name="recon_error_maps"):
    """
    For each image:
      - creates a voxel map of reconstruction error (same shape as crop)
      - saves float32 TIFF (and a 0..1 rescaled version)
    Overlaps are combined by aggregator: "mean" | "max" | "median".
    """
    from skimage.io import imread, imsave
    import numpy as np
    from pathlib import Path

    out_dir = saveFolder / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for imageName, imageData in allImagesCellFeatureDict.items():
        # locate volume and mask to get shape; reuse your crop path logic
        chCropFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
        vol_path = chCropFolder / imageName
        vol = imread(vol_path)  # (Z,Y,X)
        Z, Y, X = vol.shape

        # nucleus mask (Cellpose); fall back to >0 of vol if not found
        mask_name = imageName.replace(f"_ch{userInputList.chToClassify}.tif", "_CellposeMask.tif")
        mask_path = chCropFolder / mask_name
        nucleus_mask = imread(mask_path) if mask_path.exists() else (vol > 0).astype(np.uint8)

        # choose residual source
        # key = "recon_error_per_patch_rel" if use_relative else "recon_error_per_patch"
        if key not in imageData:
            raise KeyError(f"Missing {key} for {imageName}. Run compute_patch_residuals first.")

        residuals = imageData[key]                             # (Npatch,)
        locs = np.asarray(imageData["location"]).astype(int)   # (Npatch, 3) z,y,x
        sizes = np.asarray(imageData["patch_size"]).astype(int)

        # accumulators
        accum = np.zeros((Z, Y, X), dtype=np.float64)
        count = np.zeros((Z, Y, X), dtype=np.float32)

        # use your helper to crop the patch cuboid bounds
        from step2AnalysisHelperFcts import patch_bbox  # you already have this
        for r, loc, sz in zip(residuals, locs, sizes):
            z0,z1,y0,y1,x0,x1 = patch_bbox(loc, sz, (Z,Y,X)) 
            if aggregator == "max":
                block = accum[z0:z1, y0:y1, x0:x1]
                accum[z0:z1, y0:y1, x0:x1] = np.maximum(block, r)
            else:
                accum[z0:z1, y0:y1, x0:x1] += float(r)
                count[z0:z1, y0:y1, x0:x1] += 1.0

        if aggregator == "mean":
            with np.errstate(invalid="ignore", divide="ignore"):
                rem = accum / np.maximum(count, 1.0)
        elif aggregator == "max":
            rem = accum
        elif aggregator == "median":
            # (optional) implement via per-voxel lists if needed; mean is usually fine.
            raise NotImplementedError("median aggregator not implemented.")
        else:
            raise ValueError("aggregator must be 'mean' | 'max' | 'median'.")

        # outside nucleus → NaN
        rem = rem.astype(np.float32)
        rem[nucleus_mask == 0] = np.nan

        # save float map
        base = imageName.replace(f"_ch{userInputList.chToClassify}.tif", "")
        f32_path = out_dir / f"{base}reconError_{key}_{aggregator}.tif"
        if write_float32:
            imsave(f32_path, rem, check_contrast=False)

        # also save 0..1 rescaled for quick viz (robust to outliers)
        valid = np.isfinite(rem)
        if valid.any():
            lo, hi = np.nanpercentile(rem, (2, 98))
            scaled = np.clip((rem - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
            imsave(out_dir / f"{base}reconErrorScaled_{key}_{aggregator}.tif",
                   scaled, check_contrast=False)


def summarize_residuals_vs_attention(saveFolder, userInputList,
                                     recon_folder="recon_error_maps",
                                     attention_folder="attention_maps"):
    """
    Per image:
      - loads REM and attention map (float)
      - computes voxel-wise Pearson & Spearman (ignoring NaNs)
      - returns a small DataFrame you can plot later
    """
    import numpy as np, pandas as pd
    from skimage.io import imread
    from pathlib import Path
    from scipy.stats import pearsonr, spearmanr

    out_rows = []
    recon_dir = saveFolder / recon_folder
    attn_dir  = saveFolder / attention_folder
    dataframeFolder = saveFolder / "dataframes"

    for f in recon_dir.glob("*reconError*.tif"):
        # match attention by base
        base = f.name.split("reconError", 1)[0]
        attn_path = next(attn_dir.glob(base + "*.tif"), None)
        if attn_path is None:
            continue

        rem = imread(f).astype(np.float32)
        attn = imread(attn_path).astype(np.float32)

        # align shapes, mask NaNs
        m = np.isfinite(rem) & np.isfinite(attn)
        if not m.any():
            continue

        r_p = pearsonr(rem[m].ravel(), attn[m].ravel())[0]
        r_s = spearmanr(rem[m].ravel(), attn[m].ravel()).correlation # pyright: ignore[reportAttributeAccessIssue]
        out_rows.append({"image": base.rstrip("_"), "pearson": r_p, "spearman": r_s})

    finalDF = pd.DataFrame(out_rows)
    finalDF.to_excel(dataframeFolder / "perPatchResid_vs_attention.xlsx")

    

def calcResidualsVsEDT(saveFolder, userInputList):
    import numpy as np
    import pandas as pd
    from skimage.io import imread, imsave

    ###-- File Locations ---###
    chToFind = userInputList.chToClassify
    imageFolder = saveFolder / f"ch{chToFind}Crops"
    heatmapFolder = saveFolder / "attention_maps"
    residualFolder = saveFolder / "recon_error_maps"
    dataFrameFolder = saveFolder / "dataframes"
    outputGraphFolder = saveFolder / "outputGraphs"
    
    rows = []
    strMatchLevel = ["*_1_levels.tif", "*_2_levels.tif"]
    fileList = residualFolder.glob("*Error_*.tif")

    ##-- Go through each file and find average correlation per level --###
    for file in fileList:
        tmpErrorMap = imread(file)
        for strM in strMatchLevel:
            edtLevelName = file.stem.rsplit("reconError")[0] + strM
            tmpEDTImage = imread(heatmapFolder / edtLevelName)
            for edtValue in np.unique(tmpEDTImage[tmpEDTImage != 0]):
                mask = (tmpEDTImage == edtValue)
                region = tmpErrorMap[mask]             
                numVoxels = int(mask.sum())
                avgError = region.sum() / numVoxels
                # coverage = np.count_nonzero(posAboveThreshold) / numVoxels

                rows.append({
                    "image_name": file.stem,
                    "condition": "control" if "control" in file.stem.lower() else "lof",
                    "chromatinSpecies":"heterochromatin" if "1" in strM else "euchromatin",
                    "edtValue": f"level_{edtValue}",
                    "numVoxels": numVoxels,
                    "avgReconError": avgError})

    fullData = pd.DataFrame(rows)
    fullData["edtLevel"] = fullData["edtValue"].str.split("_").str[-1].astype("Int64")
    fullData["edtLevelSigned"] = fullData["edtLevel"].where(
        fullData["chromatinSpecies"].eq("heterochromatin"), -fullData["edtLevel"])    
    fullData.to_excel(dataFrameFolder / "reconstructionError_vs_edtZone.xlsx")

    plotResidualsVsEDT(fullData, saveFolder, userInputList)

    calcResidualsPerSpecies(saveFolder, userInputList)



def plotResidualsVsEDT(fullData, saveFolder, userInputList):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd

    ##-- User Inputs -- ##
    level_col = "edtLevelSigned"
    value_col = "avgReconError"
    plotTitle = f"Average Reconstruction Error per EDT Zone for ch{userInputList.chToClassify}"
    outputGraphFolder = saveFolder / "outputGraphs"
    outfile = outputGraphFolder / f"avgReconstructionError_edtZone.jpg"
    control_color = "#FF007F"
    lof_color = "#008080"

    ##-- Seperate out control and lof--##
    is_control = fullData[fullData["condition"].astype(str).str.contains("control", case=False, na=False)]
    is_lof     = fullData[fullData["condition"].astype(str).str.contains("lof", case=False, na=False)]

    levels = sorted(pd.unique(pd.concat([is_control[level_col], is_lof[level_col]], ignore_index=True)))
    if len(levels) == 0:
        print("[WARN] No levels to plot.")
        return


    # Data arrays per level for each condition
    c_data = [is_control.loc[is_control[level_col] == lvl, value_col].values for lvl in levels]
    l_data = [is_lof.loc[is_lof[level_col] == lvl, value_col].values for lvl in levels]

    # Prepare positions (two offsets per level)
    base = np.arange(1, len(levels) + 1, dtype=float)
    offset = 0.15
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    dropped_info = {"control": [], "lof": []}

    # Helper to draw one side
    def _draw(data_list, positions, color, label_name, key):
        # Only draw violins for levels with >=2 samples
        kept_data, kept_pos, kept_lvls = [], [], []
        for lvl, arr, pos in zip(levels, data_list, positions):
            if len(arr) >= 2:
                kept_data.append(arr)
                kept_pos.append(pos)
                kept_lvls.append(lvl)
            else:
                dropped_info[key].append(lvl)
        if kept_data:
            parts = ax.violinplot(
                kept_data,
                positions=kept_pos,
                widths=width,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for b in parts["bodies"]: # type: ignore
                b.set_facecolor(color)
                b.set_edgecolor("black")
                b.set_alpha(0.7)
            if "cmedians" in parts and parts["cmedians"] is not None:
                parts["cmedians"].set_linewidth(2.0)
            # Legend proxy
            ax.scatter([], [], color=color, label=label_name)

    _draw(c_data, base - offset, control_color, "Control", "control")
    _draw(l_data, base + offset, lof_color, "LOF", "lof")

    # Axes / labels
    ax.set_xticks(base)
    ax.set_xticklabels([str(lvl) for lvl in levels])
    ax.set_xlabel("EDT level (signed)")
    ax.set_ylabel(value_col)
    ax.set_title(plotTitle)
    ax.legend(title="Condition")
    fig.tight_layout()

    # Save
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def calcResidualsPerSpecies(saveFolder, userInputList):
    import numpy as np
    import pandas as pd
    from skimage.io import imread, imsave
    import plotly.graph_objects as go

    ###-- File Locations ---###
    chToFind = userInputList.chToClassify
    imageFolder = saveFolder / f"ch{chToFind}Crops"
    residualFolder = saveFolder / "recon_error_maps"
    dataFrameFolder = saveFolder / "dataframes"
    outputGraphFolder = saveFolder / "outputGraphs"
    
    resultList = []
    fileList = residualFolder.glob("*Error_*.tif")

    ##-- Go through each file and find average correlation per level --###
    for file in fileList:
        tmpErrorMap = imread(file)
        segName = file.stem.rsplit("reconError")[0] + "_ch0signalSeg.tif"
        tmpSegImage = imread(imageFolder / segName)
        conditionList = [("heterochromatin", 1), ("euchromatin", 2)]
        for (conditionLabel, conditionInt) in conditionList:        
            mask = (tmpSegImage == conditionInt)
            region = tmpErrorMap[mask]             
            numVoxels = int(mask.sum())
            avgError = region.sum() / numVoxels
            summedError = region.sum()

            resultList.append({
                "image_name": file.stem,
                "condition": "control" if "control" in file.stem.lower() else "lof",
                "chromatinSpecies": conditionLabel,
                "averageError": avgError,
                "summedError": summedError,
            })
    
    fullData = pd.DataFrame(resultList)
    # after building fullData in calcResidualsPerSpecies(...)
    # stats_avg = plotSpeciesAvgWithIQR(fullData, saveFolder, userInputList, value_col="averageError")

    is_control = fullData[fullData["condition"].astype(str).str.contains("control", case=False, na=False)]
    is_lof     = fullData[fullData["condition"].astype(str).str.contains("lof", case=False, na=False)]

    fig = go.Figure()
    title = f"Average Error per Chromatin Species for ch{userInputList.chToClassify}"
    fig.add_scatter(x=is_control["chromatinSpecies"], y=is_control["averageError"],
                        mode="lines+markers", name="Control", line=dict(width=3, color="#3b82f6"))
    fig.add_scatter(x=is_lof["chromatinSpecies"], y=is_lof["averageError"],
                        mode="lines+markers", name="LOF", line=dict(width=3, color="#ef4444"))

    fig.update_xaxes(title_text="Chromatin Species", tickmode="linear", dtick=1)
    fig.update_yaxes(title_text="Average Error per Chromatin Species")
    fig.update_layout(title=title, template="plotly_white", margin=dict(l=60, r=20, t=60, b=50))
    out_path = outputGraphFolder / f"averageError_chromatinSpecies_ch{userInputList.chToClassify}.jpg"
    fig.write_image(out_path, width=900, height=600, scale=2)


def plotSpeciesAvgWithIQR(fullData, saveFolder, userInputList, value_col="averageError"):
    """
    Plot per-class mean with IQR ribbon (between-image variability) for:
      x: chromatin species (euchromatin, heterochromatin)
      y: mean of `value_col` across images
      lines: Control vs LOF
      ribbon: IQR (25th–75th) across images for each species/condition

    value_col: "averageError" (default) or "summedError"
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import plotly.graph_objects as go

    saveFolder = Path(saveFolder)
    outputGraphFolder = saveFolder / "outputGraphs"
    outputGraphFolder.mkdir(parents=True, exist_ok=True)

    # ensure canonical labels and ordering
    species_order = ["euchromatin", "heterochromatin"]

    df = fullData.copy()
    # df["condition"] = df["condition"].astype(str).str.lower().map(
    #     lambda s: "control" if "control" in s else ("lof" if "lof" in s else s)
    # )
    # df["chromatinSpecies"] = df["chromatinSpecies"].astype(str).str.lower()

    # (1) per-image values (already 1 row/image/species in your calc, but keep robust)
    per_image = (df.groupby(["image_name", "condition", "chromatinSpecies"], as_index=False)
                   [value_col].mean())

    # (2) aggregate across images → mean + IQR + N
    stats = (per_image.groupby(["condition", "chromatinSpecies"])
                     [value_col]
                     .agg(mean="mean",
                          q1=lambda x: np.percentile(x, 25),
                          q3=lambda x: np.percentile(x, 75),
                          n="count")
                     .reset_index())

    # pivot to easy arrays per condition, keeping desired species order
    def series_for(cond, col):
        sub = (stats[stats["condition"] == cond]
               .set_index("chromatinSpecies")
               .reindex(species_order))
        return sub[col].astype(float).fillna(0.0).tolist()

    xs = species_order
    mean_control = series_for("control", "mean")
    q1_control   = series_for("control", "q1")
    q3_control   = series_for("control", "q3")

    mean_lof = series_for("lof", "mean")
    q1_lof   = series_for("lof", "q1")
    q3_lof   = series_for("lof", "q3")

    # build figure: for each condition, add IQR ribbon then mean line+markers
    fig = go.Figure()

    # Control ribbon
    fig.add_scatter(x=xs, y=q3_control, mode="lines", line=dict(width=0),
                    name="Control IQR (upper)", showlegend=False)
    fig.add_scatter(x=xs, y=q1_control, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(59,130,246,0.2)",
                    name="Control IQR", showlegend=True)

    # Control mean
    fig.add_scatter(x=xs, y=mean_control, mode="lines+markers",
                    name="Control mean", line=dict(width=3, color="#3b82f6"))

    # LOF ribbon
    fig.add_scatter(x=xs, y=q3_lof, mode="lines", line=dict(width=0),
                    name="LOF IQR (upper)", showlegend=False)
    fig.add_scatter(x=xs, y=q1_lof, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(239,68,68,0.2)",
                    name="LOF IQR", showlegend=True)

    # LOF mean
    fig.add_scatter(x=xs, y=mean_lof, mode="lines+markers",
                    name="LOF mean", line=dict(width=3, color="#ef4444"))

    y_label = "Average Error per Chromatin Species" if value_col == "averageError" else "Summed Error per Chromatin Species"
    title = f"{y_label} (mean ± IQR) for ch{userInputList.chToClassify}"

    fig.update_xaxes(title_text="Chromatin Species", categoryorder="array", categoryarray=xs)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title, template="plotly_white",
                      margin=dict(l=60, r=20, t=60, b=50),
                      legend_title_text="Class / Band")

    out_name = f"{value_col}_by_species_mean_IQR_ch{userInputList.chToClassify}.jpg"
    fig.write_image(outputGraphFolder / out_name, width=900, height=600, scale=2)

    return stats



def compareErrorWithAttention(saveFolder, userInputList):
    import numpy as np
    import pandas as pd
    from skimage.io import imread, imsave
    import plotly.graph_objects as go
    import plotly.io as pio
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    ###-- File Locations ---###
    chToFind = userInputList.chToClassify
    heatmapFolder = saveFolder / "attention_maps"
    residualFolder = saveFolder / "recon_error_maps"
    dataFrameFolder = saveFolder / "dataframes"
    outputGraphFolder = saveFolder / "outputGraphs"
    
    resultList = []
    fileList = residualFolder.glob("*Error_*.tif")

     ##-- Parameters --##
    attnBins = 50
    attnRange = (-1,1)
    errorBins = 50
    errorRange = (0,0.2)
    # accumulator for combined histogram
    H_class = {
        "control": np.zeros((errorBins, attnBins), dtype=np.float64),
        "lof":     np.zeros((errorBins, attnBins), dtype=np.float64),
    }

    for errorMapName in fileList:
        tmpErrorMap = imread(errorMapName)
        label = "control" if "control" in errorMapName.stem.lower() else "lof"

        heatmapName = errorMapName.stem.rsplit("reconError")[0]+f"_ch{userInputList.chToClassify}_attn_{label}.tif"        
        tmpHeatMap = imread(heatmapFolder / heatmapName)

        error1D = tmpErrorMap.ravel()  # patch errors
        attention1D = tmpHeatMap.ravel()      # attention

        # 2D histogram
        H, erredges, attedges = np.histogram2d(
            error1D, attention1D,
            bins=(errorBins, attnBins),
            range=(errorRange, attnRange)
        )
        H_class[label] += H

    for label, H in H_class.items():

        # Heatmap (log counts)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[errorRange[0], errorRange[1], attnRange[0], attnRange[1]],  # type: ignore
            norm=mcolors.LogNorm(vmin=1, vmax=max(1, H.max()))
        )
        ax.set_xlabel("Raw Error Values")
        ax.set_ylabel("Attention")
        ax.set_title(f"2D Histogram: Patch Error vs Attention ({label}) for ch{userInputList.chToClassify}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Count (log scale)")
        fig.savefig(outputGraphFolder / f"{label}_error_vs_attention_ch{userInputList.chToClassify}.png", bbox_inches="tight")
        plt.close(fig)

        # Save raw histogram and edges for downstream analysis
        np.savez_compressed(
            dataFrameFolder / f"{label}_error_vs_attention_2dhist.npz",
            H=H,
            xedges=np.linspace(errorRange[0], errorRange[1], errorBins + 1),
            yedges=np.linspace(attnRange[0], attnRange[1], attnBins + 1)
        )




def analyzeAttentionMaps(saveFolder, userInputList):
  import numpy as np
  import pandas as pd
  from skimage.io import imread, imsave
  from scipy.ndimage import label, generate_binary_structure

  print("Analyzing attention maps now...")

  heatmapFolder = saveFolder / "attention_maps"
  graphFolder = saveFolder / "outputGraphs"
  dataFrameFolder = saveFolder.joinpath("dataframes")
  dataFrameFolder.mkdir(parents = True, exist_ok = True)

  heatmapNames = heatmapFolder.glob("*.tif")
  badStrList = ["mip", "patches", "edt", "only", "blob", "level", "8bit", "raw"]
  refinedNames = [namePath for namePath in heatmapNames if not any(badStr in namePath.name.lower() for badStr in badStrList)]
  rows = []
  negativeRows = []
  thresholdValue = 0.05 #try 0.5?
  structure_6 = generate_binary_structure(3, 1)

  for mapPath in refinedNames:
    tmpHeatMap = imread(mapPath)
    tmpNameOnly = mapPath.stem.rsplit("_attn_")[0]
    conditionLabel = userInputList.get_group_name(tmpNameOnly)
    if conditionLabel is None:
      print(
        "[WARN] analyzeAttentionMaps: no classPatterns match for stem "
        f"'{tmpNameOnly}' (classNames={userInputList.classNames}). Using 'Unassigned'."
      )
      condition = "Unassigned"
    else:
      condition = conditionLabel
    thresholdHeatMap = tmpHeatMap > thresholdValue
    negativeThresholdHeatMap = tmpHeatMap < (-1 * thresholdValue)
    labeledMask, num_features = label(thresholdHeatMap, structure=structure_6)  # type: ignore
    negativeLabeledMask, negative_num_features = label(negativeThresholdHeatMap, structure=structure_6)  # type: ignore
    unique_labels, counts = np.unique(labeledMask, return_counts=True)
    negativeUnique_labels, negative_counts = np.unique(negativeLabeledMask, return_counts = True)
    blobVolumes = counts[unique_labels != 0]
    negativeBlobVolumes = negative_counts[negativeUnique_labels != 0]

    rows.append({
       "imgName" : tmpNameOnly,
       "condition": condition,
       "numberOfBlobs": num_features,
       "averageVolume": np.mean(blobVolumes),
       "stdDevVolume": np.std(blobVolumes)})
    
    negativeRows.append({
       "imgName" : tmpNameOnly,
       "condition": condition,
       "numberOfBlobs_neg": negative_num_features,
       "averageVolume_neg": np.mean(negativeBlobVolumes),
       "stdDevVolume_neg": np.std(negativeBlobVolumes)
    })
    blobSavePath = heatmapFolder / (tmpNameOnly + "_blob.tif")
    imsave(blobSavePath, labeledMask)

  summary_df = pd.DataFrame(rows).sort_values("imgName")
  negativeSummary_df = pd.DataFrame(negativeRows).sort_values("imgName")
  negativeSummary_df.to_excel(dataFrameFolder / "attentionMap_blob_analysis_negative.xlsx")
  savePath = dataFrameFolder / "attentionMap_blob_analysis.xlsx"
  summary_df.to_excel(savePath)

  plotAttentionMaps(graphFolder, summary_df, negativeSummary_df)


def plotAttentionMaps(imageSaveFolder, summary_df, negativeSummary_df):
    import pandas as pd
    import plotly.express as px

   

    # ---- Replicate your Streamlit histograms (POSITIVE) ----
    if not summary_df.empty:
        # Ensure numeric types and drop NaNs for plotting
        summary_df["numberOfBlobs"] = pd.to_numeric(summary_df["numberOfBlobs"], errors="coerce")
        summary_df["averageVolume"] = pd.to_numeric(summary_df["averageVolume"], errors="coerce")

        # Histogram: numberOfBlobs (overlay by condition)
        if summary_df["numberOfBlobs"].notna().any():
            nb_min = summary_df["numberOfBlobs"].min()
            nb_max = summary_df["numberOfBlobs"].max()
            xbins_count = dict(start=nb_min - 0.5, end=nb_max + 0.5, size=1)

            figHisto = px.histogram(
                summary_df.dropna(subset=["numberOfBlobs"]),
                x="numberOfBlobs",
                color="condition",
                opacity=0.6,
                title="Positive blobs: count per image"
            )
            figHisto.update_traces(xbins=xbins_count, bingroup="x")
            figHisto.for_each_trace(lambda t: t.update(offsetgroup=None))
            figHisto.update_layout(barmode="overlay", bargap=0.02, template="plotly_white")
            save_fig(imageSaveFolder, figHisto, "hist_positive_numberOfBlobs")

        # Histogram: averageVolume (overlay by condition)
        if summary_df["averageVolume"].notna().any():
            figSizeHisto = px.histogram(
                summary_df.dropna(subset=["averageVolume"]),
                x="averageVolume",
                color="condition",
                opacity=0.6,
                nbins=15,
                title="Positive blobs: average volume per image"
            )
            # Match your binning (0 to 160k in steps of 10k). Adjust if you want dynamic edges.
            figSizeHisto.update_traces(xbins=dict(start=0, end=160000, size=10000))
            figSizeHisto.for_each_trace(lambda t: t.update(offsetgroup=None))
            figSizeHisto.update_layout(barmode="overlay", bargap=0.02, template="plotly_white")
            save_fig(imageSaveFolder, figSizeHisto, "hist_positive_averageVolume")

    # ---- Same two histograms for NEGATIVE blobs ----
    if not negativeSummary_df.empty:
        negativeSummary_df["numberOfBlobs_neg"] = pd.to_numeric(negativeSummary_df["numberOfBlobs_neg"], errors="coerce")
        negativeSummary_df["averageVolume_neg"] = pd.to_numeric(negativeSummary_df["averageVolume_neg"], errors="coerce")

        # Histogram: numberOfBlobs_neg
        if negativeSummary_df["numberOfBlobs_neg"].notna().any():
            nb_min = negativeSummary_df["numberOfBlobs_neg"].min()
            nb_max = negativeSummary_df["numberOfBlobs_neg"].max()
            xbins_count_neg = dict(start=nb_min - 0.5, end=nb_max + 0.5, size=1)

            figNegHisto = px.histogram(
                negativeSummary_df.dropna(subset=["numberOfBlobs_neg"]),
                x="numberOfBlobs_neg",
                color="condition",
                opacity=0.6,
                title="Negative blobs: count per image"
            )
            figNegHisto.update_traces(xbins=xbins_count_neg, bingroup="x")
            figNegHisto.for_each_trace(lambda t: t.update(offsetgroup=None))
            figNegHisto.update_layout(barmode="overlay", bargap=0.02, template="plotly_white")
            save_fig(imageSaveFolder, figNegHisto, "hist_negative_numberOfBlobs")

        # Histogram: averageVolume_neg
        if negativeSummary_df["averageVolume_neg"].notna().any():
            figNegSizeHisto = px.histogram(
                negativeSummary_df.dropna(subset=["averageVolume_neg"]),
                x="averageVolume_neg",
                color="condition",
                opacity=0.6,
                nbins=15,
                title="Negative blobs: average volume per image"
            )
            figNegSizeHisto.update_traces(xbins=dict(start=0, end=160000, size=10000))
            figNegSizeHisto.for_each_trace(lambda t: t.update(offsetgroup=None))
            figNegSizeHisto.update_layout(barmode="overlay", bargap=0.02, template="plotly_white")
            save_fig(imageSaveFolder, figNegSizeHisto, "hist_negative_averageVolume")



 # ---- Helper for saving Plotly figs as PNG + HTML ----
def save_fig(imageSaveFolder, fig, stem: str):
    html_path = imageSaveFolder / f"{stem}.html"
    png_path = imageSaveFolder / f"{stem}.png"
    # fig.write_html(str(html_path))
    try:
        # Requires `kaleido` installed: pip install -U kaleido
        fig.write_image(str(png_path), scale=2)
    except Exception as e:
        print(f"[INFO] Static export failed (install kaleido). Saved HTML instead. Error: {e}")



def build_patch_table(allImgDictionary,
                      error_key="recon_rel_mse_per_patch",
                      include_images=None):
    """
    Returns a pandas DataFrame with one row per patch and columns:
      image_name, condition, attn_patch, err_patch, size_lin, z,y,x
    """
    import numpy as np, pandas as pd

    rows = []
    for image_name, d in allImgDictionary.items():
        if include_images is not None and image_name not in include_images:
            continue

        # required fields populated by your pipeline
        if "patch_attention_scores" not in d:
            # skip if attention hasn't been computed/saved for this image
            continue

        attn = np.asarray(d["patch_attention_scores"], dtype=np.float64)    # (P,)
        err  = np.asarray(d.get(error_key, []), dtype=np.float64)           # (P,)
        size = np.asarray(d.get("patch_sizes", []), dtype=np.float64)       # (P,)
        locs = np.asarray(d.get("patch_centers_zyx", []), dtype=np.int32)   # (P,3)

        # basic sanity: lengths should match
        P = len(attn)
        if len(err) != P:
            # If you generated error later or with filtering, align safely by min length
            P = min(P, len(err))
            attn, err = attn[:P], err[:P]
            if len(size) >= P: size = size[:P]
            if len(locs) >= P: locs = locs[:P]

        # crude condition inference—replace with your own metadata if you have it
        cond = "control" if "control" in image_name.lower() else "lof"

        df = pd.DataFrame({
            "image_name": image_name,
            "condition": cond,
            "attn_patch": attn[:P],
            "err_patch": err[:P],
            "size_lin": size[:P] if len(size) == P else np.nan,
            "z": locs[:P, 0] if len(locs) == P else np.nan,
            "y": locs[:P, 1] if len(locs) == P else np.nan,
            "x": locs[:P, 2] if len(locs) == P else np.nan,
        })
        rows.append(df)

    if not rows:
        raise ValueError("No patch rows found—did you run the heatmap step and save patch_attention_scores?")
    return pd.concat(rows, ignore_index=True)


def plot_attention_vs_error(df,
                            userInputList,
                            where="both",          # "pos", "neg", or "both"
                            control_for_size=True, # regress err on log(size) first
                            saveFolder=None,
                            title_suffix=""):
    """
    Show relationship between patch attention and reconstruction error.
    Prints Pearson/Spearman correlations (overall and per condition),
    and draws a hexbin + binned trend line.
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr

    data = df.copy()

    conditionList = ["control", "lof"]
    for condition in conditionList:
        save_path = saveFolder / f"attention_vs_error_{where}_ch{userInputList.chToClassify}_{condition}.jpg" # type: ignore
        dataCondition = data[data["condition"].astype(str).str.contains(condition, case=False, na=False)]
        # Filter by sign of attention if desired
        if where == "pos":
            dataCondition = dataCondition[dataCondition["attn_patch"] > 0]
        elif where == "neg":
            dataCondition = dataCondition[dataCondition["attn_patch"] < 0]
        # else: "both" → no sign filter

        # Handle size and residualize if requested
        x = dataCondition["attn_patch"].to_numpy(dtype=float)
        y = dataCondition["err_patch"].to_numpy(dtype=float)

        if control_for_size and "size_lin" in dataCondition:
            s = dataCondition["size_lin"].to_numpy(dtype=float)
            s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
            log_s = np.log(s)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(log_s)
            # simple linear residualization: y ~ a + b*log(size)
            if ok.sum() > 10:
                A = np.c_[np.ones(ok.sum()), log_s[ok]]
                coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
                y_resid = y.copy()
                y_resid[ok] = y[ok] - (coef[0] + coef[1]*log_s[ok])
                y = y_resid
                resid_note = " (size-controlled)"
            else:
                resid_note = ""
        else:
            resid_note = ""

        # Clean NaNs
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        conds = dataCondition.loc[ok, "condition"].astype(str).to_numpy()

        # Correlations
        def corr_report(xx, yy, label):
            if len(xx) < 3:
                return f"{label}: n={len(xx)}"
            r_pear, p_pear = pearsonr(xx, yy)
            r_spear, p_spear = spearmanr(xx, yy)
            return f"{label}: Pearson r={r_pear:.3f} (p={p_pear:.2e}), Spearman ρ={r_spear:.3f} (p={p_spear:.2e}), n={len(xx)}"

        lines = [corr_report(x, y, f"ALL{resid_note}")]
        for c in sorted(set(conds)):
            m = (conds == c)
            lines.append(corr_report(x[m], y[m], f"{c}{resid_note}"))

        print("\n".join(lines))

        # Plot: hexbin with binned trend
        fig, ax = plt.subplots(figsize=(7, 5))
        hb = ax.hexbin(x, y, gridsize=50, mincnt=1, bins='log')  # log color counts
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Count (log)")

        # binned median trend
        try:
            bins = np.quantile(x, np.linspace(0, 1, 21))
            bins = np.unique(bins)
            idx = np.digitize(x, bins) - 1
            mids, med = [], []
            for b in range(len(bins)-1):
                sel = (idx == b)
                if sel.sum() >= 10:
                    mids.append((bins[b] + bins[b+1]) / 2.0)
                    med.append(np.median(y[sel]))
            if len(mids) > 1:
                ax.plot(mids, med, lw=2)
        except Exception:
            pass

        where_str = {"pos":"(positive attention only)", "neg":"(negative attention only)", "both":"(both signs)"}[where]
        ax.set_title(f"Patch Attention vs Reconstruction Error {where_str}{resid_note} {title_suffix} for {condition}")
        ax.set_xlabel("Patch attention score")
        ax.set_ylabel("Reconstruction error" + resid_note)
        ax.grid(True, alpha=0.2)

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax



def analyzeError(saveFolder, userInputList, allImagesCellFeatureDict):
    # calcResidualsPerSpecies(saveFolder, userInputList)
    compareErrorWithAttention(saveFolder, userInputList)    

    patch_df = build_patch_table(allImagesCellFeatureDict,
                             error_key="recon_rel_mse_per_patch")

    posSavePath = saveFolder / "outputGraphs" 
    _ = plot_attention_vs_error(
            patch_df,
            userInputList,
            where="pos",                 # try "pos", then "neg", then "both"
            control_for_size=False,       # set False to see raw relationship
            saveFolder=posSavePath,              # or "patch_attn_vs_err_pos.png"
            title_suffix=""
        )
    negSavePath = saveFolder / "outputGraphs"
    _ = plot_attention_vs_error(
            patch_df,
            userInputList,
            where="neg",                 # try "pos", then "neg", then "both"
            control_for_size=False,       # set False to see raw relationship
            saveFolder=negSavePath,              # or "patch_attn_vs_err_pos.png"
            title_suffix=""
        )



def analyzeIntensityBlobs(saveFolder, userInputList):
    """
    Threshold intensity channels and binarize voxels > 50th percentile
    Look for connected particles and count number/size of coverage
    Return histogram of number/size of blobs
    
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from skimage.io import imread, imsave
    from scipy.ndimage import label, generate_binary_structure

    print("Analyzing intensity channels now...")

    imageFolder = saveFolder / f"ch{userInputList.chToClassify}Crops"
    graphFolder = saveFolder / "outputGraphs"
    dataFrameFolder = saveFolder.joinpath("dataframes")
    dataFrameFolder.mkdir(parents = True, exist_ok = True)

    imageNameList = imageFolder.glob(f"*ch{userInputList.chToClassify}.tif")
    resultList = []

    for imageName in imageNameList:
        tmpImg = imread(imageName)
        tmpName = imageName.stem
        tmpCellposeName = tmpName.split("_ch")[0] + "_CellposeMask.tif"
        tmpCellposeImg = imread(imageFolder / tmpCellposeName)
        insideNucleus = tmpCellposeImg != 0
        nuclearVolume = np.count_nonzero(tmpCellposeImg)

        low, high = np.percentile(tmpImg, (5, 95))
        tmpNormImage = np.clip((tmpImg - low) / (high - low), 0, 1)

        tmp50Per = np.percentile(tmpNormImage[insideNucleus], 50)
        tmpImgAbove50 = (tmpNormImage > tmp50Per) & insideNucleus
        percentNuclearVolume = np.count_nonzero(tmpImgAbove50) / nuclearVolume
        structure_6 = generate_binary_structure(3, 1)

        labeledMask, num_features = label(tmpImgAbove50, structure=structure_6) # type: ignore
        unique_labels, counts = np.unique(labeledMask, return_counts=True)
        blobVolumes = counts[unique_labels != 0]

        resultList.append({
            "imgName" : tmpName,
            "condition": "control" if "control" in tmpName.lower() else "lof",
            "numberOfBlobs": num_features,
            "percentCoverage": percentNuclearVolume,
            "averageVolume": np.mean(blobVolumes),
            "stdDevVolume": np.std(blobVolumes)})
        
        blobSavePath = imageFolder / (tmpName + "_blob.tif")
        imsave(blobSavePath, labeledMask)

    summary_df = pd.DataFrame(resultList).sort_values("imgName")

    savePath = dataFrameFolder / "intensity_blob_analysis.xlsx"
    summary_df.to_excel(savePath)

    # Histogram: numberOfBlobs (overlay by condition)
    if summary_df["numberOfBlobs"].notna().any():
        nb_min = summary_df["numberOfBlobs"].min()
        nb_max = summary_df["numberOfBlobs"].max()
        xbins_count = dict(start=nb_min - 0.5, end=nb_max + 0.5, size=1)

        figHisto = px.histogram(
            summary_df.dropna(subset=["numberOfBlobs"]),
            x="numberOfBlobs",
            color="condition",
            opacity=0.6,
            title="Positive blobs: count per image"
        )
        figHisto.update_traces(xbins=xbins_count, bingroup="x")
        figHisto.for_each_trace(lambda t: t.update(offsetgroup=None))
        figHisto.update_layout(barmode="overlay", bargap=0.02, template="plotly_white")
        figSavePath = graphFolder / "histogram_intensity_numberOfBlobs.jpg"
        figHisto.write_image(figSavePath, scale = 2)

    # Histogram: averageVolume (overlay by condition)
    if summary_df["averageVolume"].notna().any():
        figSizeHisto = px.histogram(
            summary_df.dropna(subset=["averageVolume"]),
            x="averageVolume",
            color="condition",
            opacity=0.6,
            nbins=15,
            title="Positive blobs: average volume per image"
        )
        # Match your binning (0 to 160k in steps of 10k). Adjust if you want dynamic edges.
        figSizeHisto.update_traces(xbins=dict(start=0, end=160000, size=10000))
        figSizeHisto.for_each_trace(lambda t: t.update(offsetgroup=None))
        figSizeHisto.update_layout(barmode="overlay", bargap=0.02, template="plotly_white")
        figVolSavePath = graphFolder / "histogram_intensity_blobAvgVolume.jpg"
        figSizeHisto.write_image(figVolSavePath, scale = 2)

    figCoverageHisto = px.histogram(
            summary_df.dropna(subset=["percentCoverage"]),
            x="percentCoverage",
            color="condition",
            opacity=0.6,
            nbins=15,
            title="Coverage of nuclear volume where intensity > 50th percentile")
    # Match your binning (0 to 160k in steps of 10k). Adjust if you want dynamic edges.
    # figCoverageHisto.update_traces(xbins=dict(start=0, end=160000, size=10000))
    figCoverageHisto.for_each_trace(lambda t: t.update(offsetgroup=None))
    figCoverageHisto.update_layout(barmode="overlay", bargap=0.02, template="plotly_white")
    figCoverageSavePath = graphFolder / "histogram_intensity_coverage.jpg"
    figCoverageHisto.write_image(figCoverageSavePath, scale = 2)



