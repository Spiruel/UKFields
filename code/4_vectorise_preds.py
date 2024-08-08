from osgeo import gdal
gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')
gdal.UseExceptions()

from utils import setup_environment, get_vrt_metadata, generate_contours, merge_contours

if __name__ == "__main__":
    out_name = "demo.vrt"
    pred_dir = "/home/eesjb/Documents/segment-anything/segment-anything-eo/predictions/utm27700"
    weight_file = 'weights.tif'
    shape = (1024, 1024)
    buffer = 128
    contours_dir = 'contours'
    output_file = 'merged.gpkg'

    setup_environment(out_name, pred_dir, weight_file, shape, buffer)
    meta, vrt_dim, transform = get_vrt_metadata(out_name)
    #if large vrt file, run on high memory machine
    generate_contours(out_name, vrt_dim, buffer, contours_dir)
    merge_contours(vrt_dim, buffer, contours_dir, output_file)
