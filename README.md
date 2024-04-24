<p align="center">
  <img src="ukfields.jpeg" alt="ukfieldslogo" width="300"/>
</p>

# UK Fields

The ukfields dataset is a publicly accessible Earth Engine asset containing over 2.3 million automatically delineated field boundaries spanning England, Wales, Scotland, and Northern Ireland. 

### Credits
This dataset was produced by [Samuel Bancroft](https://github.com/Spiruel) (University of Leeds) and [Jake Wilkins](https://github.com/jakenotjay) (EarthBlox).

### Dataset Details
The ukfields dataset contains field boundaries for the United Kingdom, derived from harmonic composites of Sentinel 2 imagery captured in 2021. The delineation process leveraged the Segment Anything Model (SAM) from Meta, ensuring efficient field segmentation at scale. The segmented fields have been masked to a 2021 Dynamic World composite of cropland.

Each field boundary polygon in the dataset includes the following attributes:

**Year**: The year the imagery was captured, which is 2021 for this dataset.

**County**: The county within the UK where the field is located.

**Country**: The country within the UK (England, Wales, Scotland, or Northern Ireland) where the field is located.

**Area**: The area of the field in square metres.

**Field ID**: A unique identifier for each field polygon.

### Why Open Data Matters
The availability of accurate field boundary data is of high importance, especially in regions where such information is scarce. Releasing the ukfields datase and the associated code is a step towards democratising access to parcel information over large geographic areas and serves as a blueprint for replicating similar efforts in other regions all over the world.

### Getting Started
To access the ukfields dataset in Google Earth Engine, use the following asset path:

`var ukfields = ee.ImageCollection('projects/earthblox/ukfields');`

You can also download a local .gpkg copy [here](https://gws-access.jasmin.ac.uk/public/sensecdt/eesjb/ukfields.gpkg).

For more information on accessing and working with the dataset, feel free to contact the authors or raise an issue in this Github repository.
