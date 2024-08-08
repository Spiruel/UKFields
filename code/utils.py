import numpy as np
import os
from lxml import etree
import copy
from typing import Tuple

import glob
import os
import rasterio
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm
import geopandas as gpd
import pandas as pd

#vectorisation code from https://github.com/sentinel-hub/field-delineation/blob/main/fd/vectorisation.py

def setup_environment(vrt_filename, pred_directory, weight_file, shape, buffer):
    """
    Sets up the environment by creating a weight file and writing the VRT file.

    Parameters:
    - vrt_filename: Path to the output VRT file.
    - pred_directory: Directory containing the prediction .tif files.
    - weight_file: Path to the weight file to be created.
    - shape: Tuple representing the shape (width, height) of the weight file.
    - buffer: Tuple representing the buffer (x_buffer, y_buffer) around the shape.
    """
    if os.path.exists(vrt_filename):
        os.remove(vrt_filename)

    tifs = sorted(glob.glob(pred_directory + '/*.tif'))#[:500]

    with rasterio.open(weight_file, 'w', driver='GTiff', width=shape[0], height=shape[1], count=1, dtype=np.float32) as dst:
        dst.write_band(1, get_weights(shape, buffer))

    write_vrt(tifs, weight_file, vrt_filename)


def get_vrt_metadata(vrt_file):
    """
    Gets metadata from the VRT file.

    Parameters:
    - vrt_file: Path to the VRT file.

    Returns:
    - meta: Metadata of the VRT file.
    - vrt_dim: Dimensions of the VRT file.
    - transform: Transform of the VRT file.
    """
    with rasterio.open(vrt_file) as src:
        meta = src.meta
        vrt_dim = meta['width'], meta['height']
        transform = meta['transform']
    return meta, vrt_dim, transform


def generate_contours(vrt_file, vrt_dim, buffer, contours_dir, multiprocessing_contour=True, pool_size=2):
    """
    Generates contours from the VRT file.

    Parameters:
    - vrt_file: Path to the VRT file.
    - vrt_dim: Dimensions of the VRT file.
    - buffer: Buffer size for contour generation.
    - contours_dir: Directory to save the contour files.
    - multiprocessing_contour: Boolean to enable or disable multiprocessing.
    - pool_size: Number of processes for multiprocessing.
    """
    if multiprocessing_contour:
        pool = multiprocessing.Pool(pool_size)
        run_contour_partial = partial(run_contour, size=1024, vrt_file=vrt_file, threshold=255 * 0.6, contours_dir=contours_dir)
        pool.map(run_contour_partial, [(i, j) for i in range(0, vrt_dim[0], 1024 - buffer) for j in range(0, vrt_dim[1], 1024 - buffer)])
        pool.close()
        pool.join()
    else:
        for i in tqdm(range(0, vrt_dim[0], 1024 - buffer), total=vrt_dim[0] // (1024 - buffer)):
            for j in tqdm(range(0, vrt_dim[1], 1024 - buffer), total=vrt_dim[1] // (1024 - buffer)):
                run_contour((i, j), 1024, vrt_file, threshold=255 * 0.6, contours_dir=contours_dir)


def merge_contours(vrt_dim, buffer, contours_dir, output_file):
    """
    Merges the contour files into a single GeoPackage file.

    Parameters:
    - vrt_dim: Dimensions of the VRT file.
    - buffer: Buffer size used for contour generation.
    - contours_dir: Directory containing the contour files.
    - output_file: Path to the output GeoPackage file.
    """
    for col_num in tqdm(range(0, vrt_dim[0], 1024 - buffer)):
        try:
            contours = glob.glob(contours_dir + f'/merged_{col_num}_*.gpkg')
            col_gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(contour).explode(index_parts=False) for contour in contours], ignore_index=True))

            col_gdf['geometry'] = col_gdf['geometry'].buffer(0)
            col_gdf = col_gdf.dissolve()

            if len(col_gdf) > 0:
                col_gdf.to_file(f'col_{col_num}.gpkg', driver='GPKG')
        except Exception as e:
            print(e)
            continue

    col_files = glob.glob('col_*.gpkg')
    col_gdfs = [gpd.read_file(col_file).explode(index_parts=False) for col_file in col_files]

    all_gdf = gpd.GeoDataFrame(pd.concat(col_gdfs, ignore_index=True))
    all_gdf['geometry'] = all_gdf['geometry'].buffer(0)
    all_gdf = all_gdf.dissolve().explode()
    all_gdf.to_file(output_file, driver='GPKG')

def run_contour(row_col: tuple, size: int, vrt_file: str, threshold: float = 0.6,
                contours_dir: str = '.', cleanup: bool = True, skip_existing: bool = True, suppress_output: bool = True) -> Tuple[str, bool, str]:
    """ Will create a (small) tiff file over a srcwin (row, col, size, size) and run gdal_contour on it. """

    row, col = row_col
    
    file = f'merged_{row}_{col}_{size}_{size}'
    if skip_existing and os.path.exists(f'{contours_dir}/{file}.gpkg'):
        return file, True, 'Loaded existing file ...'
    try:
        gdal_str = f'gdal_translate --config GDAL_VRT_ENABLE_PYTHON YES -srcwin {col} {row} {size} {size} {vrt_file} {file}.tiff'
        if suppress_output:
            gdal_str += ' > /dev/null'
        os.system(gdal_str)
        gdal_str = f'gdal_contour -of gpkg {file}.tiff {contours_dir}/{file}.gpkg -i {threshold} -amin amin -amax amax -p'
        if suppress_output:
            gdal_str += ' > /dev/null'
        os.system(gdal_str)
        if cleanup:
            os.remove(f'{file}.tiff')
        return f'{contours_dir}/{file}.gpkg', True, None
    except Exception as exc:
        return f'{contours_dir}/{file}.gpkg', False, exc
    
def write_vrt(files, weights_file, out_vrt, function = None):
    """ Write virtual raster

    Function that will first build a temp.vrt for the input files, and then modify it for purposes of spatial merging
    of overlaps using the provided function
    """

    if not function:
        function = average_function()

    # build a vrt from list of input files
    gdal_str = f'gdalbuildvrt temp.vrt -b 1 {" ".join(files)}'
    #save gdal_str to a tmp file then execute the file
    with open('gdalbuildvrt.sh', 'w') as f:
        f.write(gdal_str)
    os.system('bash gdalbuildvrt.sh')
    os.remove('gdalbuildvrt.sh')

    # fix the vrt
    root = etree.parse('temp.vrt').getroot()
    vrtrasterband = root.find('VRTRasterBand')
    rasterbandchildren = list(vrtrasterband)
    root.remove(vrtrasterband)

    dict_attr = {'dataType': 'Float32', 'band': '1', 'subClass': 'VRTDerivedRasterBand'}
    raster_band_tag = etree.SubElement(root, 'VRTRasterBand', dict_attr)

    # Add childern tags to derivedRasterBand tag
    pix_func_tag = etree.SubElement(raster_band_tag, 'PixelFunctionType')
    pix_func_tag.text = 'average'

    pix_func_tag2 = etree.SubElement(raster_band_tag, 'PixelFunctionLanguage')
    pix_func_tag2.text = 'Python'

    pix_func_code = etree.SubElement(raster_band_tag, 'PixelFunctionCode')
    pix_func_code.text = etree.CDATA(function)

    new_sources = []
    for child in rasterbandchildren:
        if child.tag == 'NoDataValue':
            pass
        else:
            raster_band_tag.append(child)
        if child.tag == 'SimpleSource':
            new_source = copy.deepcopy(child)
            new_source.find('SourceFilename').text = weights_file
            new_source.find('SourceProperties').attrib['DataType'] = 'Float32'
            for nodata in new_source.xpath('//NODATA'):
                nodata.getparent().remove(nodata)
            new_sources.append(new_source)

    for new_source in new_sources:
        raster_band_tag.append(new_source)

    os.remove('temp.vrt')

    with open(out_vrt, 'w') as out:
        out.writelines(etree.tounicode(root, pretty_print=True))

def get_weights(shape: Tuple[int, int], buffer: int, low: float = 0, high: float = 1) -> np.ndarray:
    """ Create weights array based on linear gradient from low to high from edges to 2*buffer, and 1 elsewhere. """
    weight = np.ones(shape, dtype=np.float32)
    weight[..., :2 * buffer] = np.tile(np.linspace(low, high, 2 * buffer), shape[0]).reshape((shape[0], 2 * buffer))
    weight[..., -2 * buffer:] = np.tile(np.linspace(high, low, 2 * buffer), shape[0]).reshape((shape[0], 2 * buffer))
    weight[:2 * buffer, ...] *= np.repeat(np.linspace(low, high, shape[1]), 2 * buffer).reshape((2 * buffer, shape[1]))
    weight[-2 * buffer:, ...] *= np.repeat(np.linspace(high, low, 2 * buffer), shape[1]).reshape((2 * buffer, shape[1]))
    return weight

def average_function(no_data = 0, round_output: bool =False) -> str:
    """ A Python function that will be added to VRT and used to calculate weighted average over overlaps

    :param no_data: no data pixel value (default = 0)
    :param round_output: flag to round the output (to 0 decimals). Useful when the final result will be in Int.
    :return: Function (as a string)
    """
    rounding = 'out = np.round(out, 0)' if round_output else ''
    return f"""
import numpy as np

def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    p, w = np.split(np.array(in_ar), 2, axis=0)
    n_overlaps = np.sum(p!={no_data}, axis=0)
    w_sum = np.sum(w, axis=0, dtype=np.float32) 
    p_sum = np.sum(p, axis=0, dtype=np.float32) 
    weighted = np.sum(p*w, axis=0, dtype=np.float32)
    out = np.where((n_overlaps>1) & (w_sum>0) , weighted/w_sum, p_sum/n_overlaps)
    {rounding}
    out_ar[:] = out
    return out_ar
"""
