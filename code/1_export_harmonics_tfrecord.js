var START_DATE = '2022-10-01'
var END_DATE = '2023-06-02'
var CLOUD_FILTER = 60;
var CLD_PRB_THRESH = 50;
var NIR_DRK_THRESH = 0.15;
var CLD_PRJ_DIST = 1;
var BUFFER = 50;

var aoi = ee.FeatureCollection('FAO/GAUL/2015/level0').filterMetadata('ADM0_CODE','equals',256);

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
             .filterDate(START_DATE, END_DATE)
             .filterBounds(aoi);
             
var classification = dw.select('label');
var dwComposite = classification.reduce(ee.Reducer.mode());

// Clip the composite and add it to the Map.
var croplands = dwComposite.updateMask(dwComposite.clip(aoi).eq(4));
var grass = dwComposite.updateMask(dwComposite.clip(aoi).eq(2));
var combinedMask = croplands.blend(grass);

var get_s2_sr_cld_col = function(aoi, start_date, end_date) {
    // Import and filter S2 SR.
    var s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    // Import and filter s2cloudless.
    var s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    // Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply({
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals({
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))
};

var add_cloud_bands = function(img) {
    // Get s2cloudless image, subset the probability band.
    var cld_prb = ee.Image(img.get('s2cloudless')).select('probability');

    // Condition s2cloudless by the probability threshold value.
    var is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds');

    // Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]));
}

var add_shadow_bands = function(img) {
    // Identify water pixels from the SCL band.
    var not_water = img.select('SCL').neq(6)

    // Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    var SR_BAND_SCALE = 1e4
    var dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    // Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    var shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    // Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    var cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject({'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    // Identify the intersection of dark pixels with cloud shadow projection.
    var shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    // Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))
}

var add_cld_shdw_mask = function(img) {
  // Add cloud component bands.
    var img_cloud = add_cloud_bands(img)

  //  Add cloud shadow component bands.
    var img_cloud_shadow = add_shadow_bands(img_cloud)

  //  Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    var is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

  //  Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
  //  20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject({'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

  //  Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)
    }
    
var apply_cld_shdw_mask = function(img) {
    // Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    var not_cld_shdw = img.select('cloudmask').not()

    // Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)
}

var s2_sr_cld_col_eval = get_s2_sr_cld_col(geometry, START_DATE, END_DATE)
var l8toa = s2_sr_cld_col_eval.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)

//var roi = ee.Geometry.Point(-6.173543904111879, 8.062628227578857);
var roi = geometry2 //ee.Geometry.Point([0.334626864725478,52.444419628380246]).buffer(10);
    
Map.centerObject(roi, 10);

// This field contains UNIX time in milliseconds.
var timeField = 'system:time_start';

// Use this function to mask clouds in Landsat 8 imagery.
var maskClouds = function(image) {
  var quality = image.select('BQA');
  var cloud01 = quality.eq(61440);
  var cloud02 = quality.eq(53248);
  var cloud03 = quality.eq(28672);
  var mask = cloud01.or(cloud02).or(cloud03).not();
  return image.updateMask(mask);
};

// Use this function to add variables for NDVI, time and a constant
// to Landsat 8 imagery.
var addVariables = function(image) {
  // Compute time in fractional years since the epoch.
  var date = ee.Date(image.get(timeField));
  var years = date.difference(ee.Date('1970-01-01'), 'year');
  // Return the image with the added bands.
  return image
    // Add an NDVI band.
    .addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI')).float()
    // Add a time band.
    .addBands(ee.Image(years).rename('t').float())
    // Add a constant band.
    .addBands(ee.Image.constant(1));
};

// Remove clouds, add variables and filter to the area of interest.
var filteredLandsat = l8toa
  .filterBounds(geometry)
  // .map(maskClouds)
  .map(addVariables);

// Plot a time series of NDVI at a single location.
var l8Chart = ui.Chart.image.series(filteredLandsat.select('NDVI'), roi)
    .setChartType('ScatterChart')
    .setOptions({
      title: 'Landsat 8 NDVI time series at ROI',
      trendlines: {0: {
        color: 'CC0000'
      }},
      lineWidth: 1,
      pointSize: 3,
    });
print(l8Chart);

// Linear trend ----------------------------------------------------------------
// List of the independent variable names
var independents = ee.List(['constant', 't']);

// Name of the dependent variable.
var dependent = ee.String('NDVI');

// Compute a linear trend.  This will have two bands: 'residuals' and 
// a 2x1 band called coefficients (columns are for dependent variables).
var trend = filteredLandsat.select(independents.add(dependent))
    .reduce(ee.Reducer.linearRegression(independents.length(), 1));
// Map.addLayer(trend, {}, 'trend array image');

// Flatten the coefficients into a 2-band image
var coefficients = trend.select('coefficients')
  .arrayProject([0])
  .arrayFlatten([independents]);

// Compute a de-trended series.
var detrended = filteredLandsat.map(function(image) {
  return image.select(dependent).subtract(
          image.select(independents).multiply(coefficients).reduce('sum'))
          .rename(dependent)
          .copyProperties(image, [timeField]);
});

// Plot the detrended results.
var detrendedChart = ui.Chart.image.series(detrended, roi, null, 30)
    .setOptions({
      title: 'Detrended Landsat time series at ROI',
      lineWidth: 1,
      pointSize: 3,
    });
print(detrendedChart);

// Harmonic trend ----------------------------------------------------------------
// Use these independent variables in the harmonic regression.
var harmonicIndependents = ee.List(['constant', 't', 'cos', 'sin']);

// Add harmonic terms as new image bands.
var harmonicLandsat = filteredLandsat.map(function(image) {
  var timeRadians = image.select('t').multiply(2 * Math.PI);
  return image
    .addBands(timeRadians.cos().rename('cos'))
    .addBands(timeRadians.sin().rename('sin'));
});
  
// The output of the regression reduction is a 4x1 array image.
var harmonicTrend = harmonicLandsat
  .select(harmonicIndependents.add(dependent))
  .reduce(ee.Reducer.linearRegression(harmonicIndependents.length(), 1));

// Turn the array image into a multi-band image of coefficients.
var harmonicTrendCoefficients = harmonicTrend.select('coefficients')
  .arrayProject([0])
  .arrayFlatten([harmonicIndependents]);

// Compute fitted values.
var fittedHarmonic = harmonicLandsat.map(function(image) {
  return image.addBands(
    image.select(harmonicIndependents)
      .multiply(harmonicTrendCoefficients)
      .reduce('sum')
      .rename('fitted'));
});

// Plot the fitted model and the original data at the ROI.
print(ui.Chart.image.series(
  fittedHarmonic.select(['fitted','NDVI']), roi, ee.Reducer.mean(), 30)
    .setSeriesNames(['NDVI', 'fitted'])
    .setOptions({
      title: 'Harmonic model: original and fitted values',
      lineWidth: 1,
      pointSize: 3,
}));

// Compute phase and amplitude.
var phase = harmonicTrendCoefficients.select('cos').atan2(
            harmonicTrendCoefficients.select('sin'));
            
var amplitude = harmonicTrendCoefficients.select('cos').hypot(
                harmonicTrendCoefficients.select('sin'));

// Use the HSV to RGB transform to display phase and amplitude
var rgb = phase.unitScale(-Math.PI, Math.PI).addBands(
          amplitude.multiply(2.5)).addBands(
          filteredLandsat.select('NDVI').median()).hsvToRgb();
var rgb = rgb.updateMask(combinedMask)
        
Map.addLayer(l8toa.filterDate('2023-05-01','2023-06-01').median(), {'bands':['B4','B3','B2'], min:0, max:3000}, 'Sentinel 2', false)
Map.addLayer(l8toa.filterDate('2019-05-01','2019-06-01').median(), {'bands':['B4','B3','B2'], min:0, max:0.3}, 'Landsat 8', false)
Map.addLayer(rgb, {}, 'phase (hue), amplitude (saturation)');

// use unit scale to normalize the pixel values
var rgb = rgb.unitScale(0, 1);

var eightbitRGB = rgb.multiply(255).toByte();

print(eightbitRGB)

Map.addLayer(eightbitRGB, {min:0, max:255}, 'rgb_')

var image_export_options = {
  'patchDimensions': [768, 768],
  'maxFileSize': 104857600,
  'kernelSize': [256, 256],
  'compressed': true
}
var geometry = ee.FeatureCollection('FAO/GAUL/2015/level0').filterMetadata('ADM0_CODE','equals',256);
Export.image.toDrive({
  image: eightbitRGB.clip(geometry), 
  description: "Export_Task",
  folder: 'extra_county',
  fileNamePrefix: "extra_county", 
  region: geometry, 
  scale: 10, 
  crs: "EPSG:27700", 
  maxPixels: 29073057936, 
  formatOptions: image_export_options,
  fileFormat: 'TFRecord',
})