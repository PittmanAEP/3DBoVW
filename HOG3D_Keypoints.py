
##################################################################################################################
###Implementation of 3D Histogram of Oriented Gradiants, adapted from the Matlab code from Rob Dupre, Vasileios Argyriou, D. Greenhill, Georgios Tzimiropoulos 3D Scene Analysis Framework and Descriptors for Risk Evaluation. DV 2015: 100-108
################################################################################################################
  
##Workflow:
##(1) Local Regions: break down 3D image into blocks (default value: 2 cells in one block) and decompose those further into cells (default value: 16 voxels in one cell), which are collections of voxels. The size of the blocks/cells can be adjusted and they can overlap to improve robustness of the algorithm

##(2) Gradient Analysis: compute the gradient in 3 dimensions to capture intensity changes

##(3) Histogram Representation: the gradients are then binned into histograms based off of their two descriptive angles (theta and phi) creating a 2D histogram per 'cell'. These histograms summerize the local gradient orientations and magnitudes 

##(4) Normalization: each block is then normalized to improve invariance to intensity

##(5) Feature Outputs: The final output contains the following information: block location, cell location, and features (a matrix that contains the normalized histogram data, representing local features for each block)



def HOG_3D_keypoints(imagePath, cellMaskFile, userInputList, keypointList):
    ###Dependancies 
    import numpy as np
    from scipy.ndimage import convolve
    from skimage.io import imread
    import tifffile as tf
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image*")

    # ##import user inputs
    blockOccupancy = userInputList.blockOccupancy
    thetaHistogramBins = userInputList.thetaHistogramBins
    phiHistogramBins = userInputList.phiHistogramBins
    
    origImage = tf.imread(imagePath).astype(np.float64)
    if userInputList.siftNormalization == "5-95":            
        low, high = np.percentile(origImage, (5, 95))
        tmpNormImage = np.clip((origImage - low) / (high - low), 0, 1)
        img = tmpNormImage
    if userInputList.siftNormalization == "asinh":            
        #clip the bottom 25 and the top 2 percentiles to reduce outlier influence, then apply asinh and rescale to [0,1]
        low, high = np.percentile(origImage, (25, 98))
        image = np.clip(origImage, low, high)
        image_asinh = np.arcsinh(image)
        image_scaled = (image_asinh - np.min(image_asinh)) / (np.max(image_asinh) - np.min(image_asinh))
        img = image_scaled
    if userInputList.siftNormalization == "none":
        img = origImage

    if cellMaskFile == None:
        cellMask = np.ones_like(img)
    else:
        cellMaskName = cellMaskFile.stem
        origCellMask = imread(cellMaskFile)
        cellMask = np.copy(origCellMask)
        cellMask[cellMask > 0] = 1
    # if extraErosion:
    #     tmpErodedName = image.parent.joinpath(cellMaskName+"Extra.tif")
    #     if not tmpErodedName.exists():
    #         erodedCellMask = binary_erosion(cellMask, structure=np.ones((2,2,2))).astype(cellMask.dtype)            
    #         imsave(tmpErodedName, erodedCellMask)
    #     else:
    #         erodedCellMask = imread(tmpErodedName)
    #     cellMask = erodedCellMask

    if img.shape != cellMask.shape:
        print(f"image shape was {img.shape}")
        print(f"image shape was {cellMask.shape}")
        print(f"image name was {imagePath.stem}")
        print(f"mask name was {cellMaskName}")
        raise ValueError("Image and mask must have the same shape.")

    featureDict = {"Location":[], "Scale":[],"Features":[], "Features_Flat":[]}
    
    imgSizeX = img.shape[2]
    imgSizeY = img.shape[1]
    imgSizeZ = img.shape[0]  
    
    ##Step Two: Compute gradient vectors
    x_filter = np.zeros((3, 3, 3))
    x_filter[0, 1, 1] = 1
    x_filter[2, 1, 1] = -1
    x_vector = convolve(img, x_filter, mode='constant', cval=0)
    
    # Y FILTER AND VECTOR COMPONENT
    y_filter = np.zeros((3, 3, 3))
    y_filter[1, 0, 1] = 1
    y_filter[1, 2, 1] = -1
    y_vector = convolve(img, y_filter, mode='constant', cval=0)
    
    # Z FILTER AND VECTOR COMPONENT
    z_filter = np.zeros((3, 3, 3))
    z_filter[1, 1, 0] = 1
    z_filter[1, 1, 2] = -1
    z_vector = convolve(img, z_filter, mode='constant', cval=0)
    
    # GET MAGNITUDE OF EACH VECTOR
    magnitudes = np.sqrt(x_vector**2 + y_vector**2 + z_vector**2)    

    # GET A WEIGHTING FOR EACH VOXEL BASED ON THE VOXELS AROUND IT
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size**3)
    weights = convolve(img, kernel, mode='constant', cval=0)
    weights += 1
    
    # BUILD THE 3D VECTORS 4D ARRAY
    grad_vector = np.stack((x_vector, y_vector, z_vector), axis=-1)
    # print(f"grad vector is are: {grad_vector[..., 2]}")
    # THETA ANGLE FROM THE Z AND RADIUS PLANE IN [0, π]
    theta = np.arccos(grad_vector[..., 2] / (magnitudes + np.finfo(float).eps))  # Add epsilon to avoid division by zero


    # PHI ANGLE FROM THE X-Y PLANE IN [0, 2π]
    phi = np.arctan2(grad_vector[..., 1], grad_vector[..., 0])
    phi = phi + np.pi  # Shift to [0, 2π]
    
    ##Step Three: Binning the 3D histograms
    # blockSizeVoxels = cellSize*blockSize    
    # blockSizeVoxelsZ = int(round(zCellSize*zBlockSize))

    #tHistBins = np.pi/thetaHistogramBins
    tHistBins = (2* np.pi) / thetaHistogramBins
    pHistBins = (2* np.pi) / phiHistogramBins
    outsideCount = 0
    outsideBlockCount = 0
    errorCount = 0

    ## Step 2: Define Neighbor Offsets (26 in 3D)
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    
    num_neighbors = len(neighbor_offsets)
    
    ## Step 3: Precompute Neighboring Information for the Entire Image
    relative_magnitudes = np.zeros((*img.shape, num_neighbors))
    relative_theta = np.zeros((*img.shape, num_neighbors))
    relative_phi = np.zeros((*img.shape, num_neighbors))

    for i, (dx, dy, dz) in enumerate(neighbor_offsets):
        kernel = np.zeros((3, 3, 3))
        kernel[1 + dz, 1 + dy, 1 + dx] = 1

        neighbor_magnitudes = convolve(magnitudes, kernel, mode='constant', cval=0)
        neighbor_theta = convolve(theta, kernel, mode='constant', cval=0)
        neighbor_phi = convolve(phi, kernel, mode='constant', cval=0)

        relative_magnitudes[..., i] = np.arctan(magnitudes / (neighbor_magnitudes + np.finfo(float).eps))
        relative_theta[..., i] = (theta - neighbor_theta + np.pi) % (2 * np.pi) - np.pi
        relative_phi[..., i] = (phi - neighbor_phi + np.pi) % (2 * np.pi) - np.pi

    fullCellDictionary = {
    "location": [],
    "scale": [],
    "patch_size": [],
    "feature": [],
    "featureL2": []
}

    Z, Y, X = img.shape
    sigma_list = [1.0, 1.6, 2.2, 3.0]
    patch_scale_factor = 5  # similar to SIFT (3)

    for (octave_idx, scale_idx), zc, yc, xc in keypointList:
        factor = 2 ** octave_idx
        z_full, y_full, x_full = zc * factor, yc * factor, xc * factor

        # get absolute scale at this keypoint
        sigma_base = sigma_list[scale_idx]
        sigma_abs = sigma_base * factor

        # determine patch size
        patch_size = int(round(sigma_abs * patch_scale_factor))
        # print(f"patch size: {patch_size}")
        half_patch = patch_size // 2

        z1, z2 = int(z_full - half_patch), int(z_full + half_patch + 1)
        y1, y2 = int(y_full - half_patch), int(y_full + half_patch + 1)
        x1, x2 = int(x_full - half_patch), int(x_full + half_patch + 1)


        # z1, z2 = zc - half_patch, zc + half_patch
        # y1, y2 = yc - half_patch, yc + half_patch
        # x1, x2 = xc - half_patch, xc + half_patch

        if z1 < 0 or y1 < 0 or x1 < 0 or z2 > Z or y2 > Y or x2 > X:
            continue

        patch_mask = cellMask[z1:z2, y1:y2, x1:x2]
        patch_weights = weights[z1:z2, y1:y2, x1:x2]
        patch_relative_magnitudes = relative_magnitudes[z1:z2, y1:y2, x1:x2]
        patch_relative_theta = relative_theta[z1:z2, y1:y2, x1:x2]
        patch_relative_phi = relative_phi[z1:z2, y1:y2, x1:x2]

        checkBlockCondition = np.sum(patch_mask) > blockOccupancy*(patch_mask.shape[0]*patch_mask.shape[1]*patch_mask.shape[2])
        if not checkBlockCondition:
            continue
        else:
            voxel_contributions = np.zeros((patch_size, patch_size, patch_size))
            voxelNumber = 0
            descriptor = np.zeros((thetaHistogramBins, phiHistogramBins))
            for l in range(patch_relative_theta.shape[0]):
                for m in range(patch_relative_theta.shape[1]):
                    for n in range(patch_relative_theta.shape[2]):
                        if patch_mask[l, m, n] != 0:
                            cell_pos_x = int(np.ceil((l + 1) / patch_size)) - 1
                            cell_pos_y = int(np.ceil((m + 1) / patch_size)) - 1
                            cell_pos_z = int(np.ceil((n + 1) / patch_size)) - 1
                            voxel_contributions[cell_pos_x, cell_pos_y, cell_pos_z] += 1
                            voxelNumber += 1

                            for neighborNumber, (offset_x, offset_y, offset_z) in enumerate(neighbor_offsets):
                                neighbor_z = l + offset_z
                                neighbor_y = m + offset_y
                                neighbor_x = n + offset_x

                                ## Ensure Neighbor is Within Bounds
                                if not (0 <= neighbor_z < img.shape[0] and
                                        0 <= neighbor_y < img.shape[1] and
                                        0 <= neighbor_x < img.shape[2]):
                                    continue


                                relative_magnitude_value = patch_relative_magnitudes[l, m, n, neighborNumber]
                                relative_theta_value = patch_relative_theta[l, m, n, neighborNumber]
                                relative_phi_value = patch_relative_phi[l, m, n, neighborNumber]

                                # Normalize orientations to valid range
                                if relative_theta_value > np.pi:
                                    relative_theta_value -= 2 * np.pi
                                elif relative_theta_value < -np.pi:
                                    relative_theta_value += 2 * np.pi

                                if relative_phi_value > np.pi:
                                    relative_phi_value -= 2 * np.pi
                                elif relative_phi_value < -np.pi:
                                    relative_phi_value += 2 * np.pi

                                #adjust angles so that it runs from 0 - 2 pi
                                relative_theta_value += np.pi
                                relative_phi_value += np.pi

                                # ASSIGN THE GRADIENT ANGLE TO THE CORRESPONDING BIN
                                hist_pos_theta = int((relative_theta_value) / tHistBins) # - 1
                                hist_pos_phi = int((relative_phi_value) / pHistBins)
                                                            
                                # CHECK THAT THE VALUES FALL INTO ONE OF THE DEFINED BINS
                                if 0 <= hist_pos_theta < thetaHistogramBins and 0 <= hist_pos_phi < phiHistogramBins:      
                                    descriptor[hist_pos_theta, hist_pos_phi] += (relative_magnitude_value * patch_weights[l, m, n])                                                            
                                
            flatDescript = descriptor.flatten()

            # flatDescriptNorm = flatDescript / (np.linalg.norm(flatDescript) + 1e-8)
            flatDescriptNorm = (flatDescript + np.finfo(float).eps) / np.sqrt(np.sum(flatDescript**2) + np.finfo(float).eps)
            flatDescriptNorm = flatDescriptNorm / (voxelNumber + np.finfo(float).eps)


            if np.sum(flatDescript) < 1e-5:
                continue
            # print(flatDescript)
            # exit()
            fullCellDictionary["location"].append((zc, yc, xc))
            fullCellDictionary["scale"].append(scale_idx)
            fullCellDictionary["patch_size"].append(patch_size)
            fullCellDictionary["feature"].append(flatDescript)
            fullCellDictionary["featureL2"].append(flatDescriptNorm)
            
    return fullCellDictionary



