import tensorflow as tf
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.crs import CRS
import numpy as np
import json
import gzip

# Path to the mixer.json file and TFRecords
mixer_json_path = 'tfrecords/ukfields-mixer.json'
tfrecords_path = 'tfrecords/ukfields-00000.tfrecord.gz'

# Read mixer.json for scaling and offset
with open(mixer_json_path, 'r') as mixer_file:
	mixer_data = json.load(mixer_file)

crs = CRS.from_string(mixer_data['projection']['crs'])
affine = Affine(*mixer_data['projection']['affine']['doubleMatrix'])

patch_width = mixer_data['patchDimensions'][0]
patch_height = mixer_data['patchDimensions'][1]
patches_per_row = mixer_data['patchesPerRow']


for n in range(129):
	tfrecords_path = f'tfrecords/ukfields-00{str(n).zfill(3)}.tfrecord.gz'
	# Create a TensorFlow dataset for reading compressed TFRecords
	dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type='GZIP')

	# Loop through each TFRecord
	for num, string_record in enumerate(dataset):
		num+=33*n
		example = tf.train.Example()
		example.ParseFromString(string_record.numpy())

		# Extract image data
		r_data = example.features.feature['red'].bytes_list.value[0]
		g_data = example.features.feature['green'].bytes_list.value[0]
		b_data = example.features.feature['blue'].bytes_list.value[0]

		# Get image dimensions
		overlap = 256//2
		height, width = np.array(mixer_data['patchDimensions'])+2*overlap

		# Convert to numpy array
		r_array = np.frombuffer(r_data, dtype=np.int8).reshape(height, width)
		g_array = np.frombuffer(g_data, dtype=np.int8).reshape(height, width)
		b_array = np.frombuffer(b_data, dtype=np.int8).reshape(height, width)

		image_array = np.stack([r_array, g_array, b_array], axis=0).squeeze()

		row = num // patches_per_row
		col = num % patches_per_row

		x = col * mixer_data['patchDimensions'][0]
		y = row * mixer_data['patchDimensions'][1]

		transform = affine * Affine.translation(x, y) * Affine.translation(-overlap, -overlap)

		# Create a GeoTIFF file using rasterio
		output_path = f'tfrecords/ukfields_export/{row}_{col}_{num}.tif'
		with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width,
					   count=3, dtype='int8', crs=crs, transform=transform) as dst:
		    dst.write(image_array)

	print('Conversion complete.', num+1)