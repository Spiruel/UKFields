#this script is for prototype purposes only, it is not optimized for speed or memory usage
#for large projects, consider using https://github.com/Prindle19/segment-everything

import os
import matplotlib.pyplot as plt
#https://github.com/aliaksandr960/segment-anything-eo/tree/main
from sameo import SamEO

import os
import numpy as np
import rasterio
from tqdm import tqdm
import glob

file_paths = glob.glob('ukfields/*.tif*') 

# Available SamEO arguments:
# checkpoint="sam_vit_h_4b8939.pth",
# model_type='vit_h',
# device='cpu',
# erosion_kernel=(3, 3),
# mask_multiplier=255,
# sam_kwargs=None

# Availble sam_kwargs:
# points_per_side: Optional[int] = 32,
# points_per_batch: int = 64,
# pred_iou_thresh: float = 0.88,
# stability_score_thresh: float = 0.95,
# stability_score_offset: float = 1.0,
# box_nms_thresh: float = 0.7,
# crop_n_layers: int = 0,
# crop_nms_thresh: float = 0.7,
# crop_overlap_ratio: float = 512 / 1500,
# crop_n_points_downscale_factor: int = 1,
# point_grids: Optional[List[np.ndarray]] = None,
# min_mask_region_area: int = 0,
# output_mode: str = "binary_mask",

device = 'cuda:0'

sam_kwargs = {
  "points_per_side": 64,
  "pred_iou_thresh": 0.86,
  "stability_score_thresh": 0.92,
  "crop_n_layers": 1,
  "crop_n_points_downscale_factor": 2,
  "min_mask_region_area": 100,
}
#{'points_per_side':64, 'min_mask_region_area':0, 'pred_iou_thresh': 0.86, 'stability_score_thresh':0.92}
sam_eo = SamEO(checkpoint="sam_vit_h_4b8939.pth",#"sam_vit_h_4b8939.pth", #"/extra/demo2D/sam_model_best.pth",
               model_type='vit_h',
               device=device,
               erosion_kernel=(3,3),
               mask_multiplier=255,
               sam_kwargs=sam_kwargs)

def predict(patch):
    # Perform prediction on the chunk, e.g. using sam_eo
    pred = sam_eo(patch)
    return pred

for file_path in tqdm(file_paths):
    #reorder so rasterio open doesnt bottleneck!
    patch_filename = f"./predictions/patch_{os.path.basename(file_path).split('.')[-2]}.tif"
    if not os.path.exists(patch_filename): 
        with rasterio.open(file_path, dtype=np.uint8) as dataset:
            height, width = dataset.shape
            # Read the entire raster data
            raster_data = dataset.read()
            raster_data = np.moveaxis(raster_data, 0, -1)[:, :, :3]
            if np.count_nonzero(raster_data) == 0:
                os.remove(file_path)
                print('empty patch')
                continue
                
            patch = raster_data.astype(np.uint8)

            #print(patch.shape, file_path)
            # Apply the prediction function to the patch
            try:
                prediction = predict(patch)
            except Exception as e:
                print(e, file_path, patch_filename)
                continue

            # Save the prediction as a geotiff
            with rasterio.open(patch_filename, 'w', driver='GTiff', count=1,
                               width=width, height=height,
                               dtype=prediction.dtype,
                               crs=dataset.crs, transform=dataset.transform) as dst:
                dst.write(prediction, 1)

            print(f"Processed patch and saved results to {patch_filename}.")
    else:
        print(f"Patch already exists, skipping.")