import rasterio
from utils import config
import multiprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fiona
import rasterio
import os
from rasterio.mask import mask
from osgeo import gdal, osr, ogr
from PIL import Image
import glob

Image.MAX_IMAGE_PIXELS = None 

def in_polygon(raster,vector):
    ''''
    Check if polygon and georeferenced satellite image 
    overlap
    
    # Arguments
        raster_path: path to the satellite image
        vector_path: path to the polygon
    # Return
        True/ False
    ''''
    
    raster = gdal.Open(raster_path)
    vector = ogr.Open(vector_path)

    # Get raster geometry
    transform = raster.GetGeoTransform()
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    xLeft = transform[0]
    yTop = transform[3]
    xRight = xLeft+cols*pixelWidth
    yBottom = yTop+rows*pixelHeight

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xLeft, yTop)
    ring.AddPoint(xLeft, yBottom)
    ring.AddPoint(xRight, yBottom)
    ring.AddPoint(xRight, yTop)
    ring.AddPoint(xLeft, yTop)
    rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
    rasterGeometry.AddGeometry(ring)
    # Get vector geometry
    layer = vector.GetLayer()
    feature = layer.GetFeature(0)
    vectorGeometry = feature.GetGeometryRef()
    
    return rasterGeometry.Intersect(vectorGeometry)

def tif_to_png(tif_path,png_path):
    ''''
    Convert Geotiff file to a PNG file 
    
    # Arguments
        tif_path: path to the Geotiff image
        png_path: path to the PNG image
    # Return
        
    ''''
    gdal.Translate(
        png_path,
        tif_path,
        options=options_string
    )


def color_4(val):
    ''''
    Create masks: pixel color references per class for 
    the 4 class category
    
    # Arguments
        val: String representing what the label of the 
        pixel should be
        
    # Return
        The RGB Value corresponding to the string
    ''''
    
    if val== 'Collective_living_quarters':
        return [201,201,201]

    elif val == 'Commercial':
        return [7,7,7] 
        
    elif val == 'Farm':
        return [124,124,124]
    
    elif val == 'Informal_settlement':
        return [201,201,201] 
        
    elif val == 'Formal_Residential':
        return [124,124,124]
    
    elif val == 'Industrial':
        return [7,7,7] 
        
    elif val == 'Parks_and_recreation':
        return [7,7,7]
    
    elif val == 'Smallholdings':
        return [124,124,124]     
    
    elif val == 'Traditional_residential':
        return [201,201,201] 
        
    elif val == 'Vacant':
        return [7,7,7]
    
    elif val == 'Other':
        return [7,7,7] 
    
    elif val == 'Township':
        return [201,201,201]  
    
    else:
        return [255,255,255]  
    
def color_12(val):
    ''''
    Create masks: pixel color references per class for 
    the 12 class category
    
    # Arguments
        val: String representing what the label of the 
        pixel should be
        
    # Return
        The RGB Value corresponding to the string
    ''''
    
    if val== 'Collective_living_quarters':
        return [200,200,200]

    elif val == 'Commercial':
        return [64,128,64] 
        
    elif val == 'Farm':
        return [0,255,0]
    
    elif val == 'Informal_settlement':
        return [255,0,0] 
        
    elif val == 'Formal_Residential':
        return [0,255,255]
    
    elif val == 'Industrial':
        return [255,0,255] 
        
    elif val == 'Parks_and_recreation':
        return [128,64,128]
    
    elif val == 'Smallholdings':
        return [0,0,255]     
    
    elif val == 'Traditional_residential':
        return [100,100,100] 
        
    elif val == 'Vacant':
        return [0,0,0]
    
    elif val == 'Other':
        return [0,0,0] 
        
    elif val == 'Township':
        return [255,255,0]
    
    else:
        return [255,255,255]    

def splt_shp_file(src,shp_folder):
    ''''
    Split shapefile to make each row an individual 
    file
    
    # Arguments
        src: the source shapefile with multiple rows
        shp_folder: path to store the individual 
        shapefiles
    # 
        
    ''''    
    
    if not os.path.exists(shp_folder):
    os.makedirs(shp_folder)
    
    id_=0
    with fiona.open(src) as source:

        meta = source.meta
        print(source[0]['properties']['EA_TYPE'])
        for f in source:

            outfile = os.path.join(shp_folder, 
            str(id_)+ "__%s.shp" % f['properties']
            ['EA_TYPE'].replace(' ','_'))
            id_ = id_+1
            with fiona.open(outfile, 'w', **meta) 
            as sink:

                sink.write(f) 

def create_empty_png(im_size,dst):
    ''''
    create emptpy pngs
    
    # Arguments
        im_size: image size (n x n)
        dst: path to store the pngs
    # 
        
    ''''   
   
    Image.MAX_IMAGE_PIXELS = None
    img = Image.new('RGB', (im_size,im_size), 
                    (255, 255, 255))
    
    if not os.path.exists(dst):
        img.save(dst, "PNG")
        
        
def create_mask(in_image,blank_png,color):
    ''''
    create emptpy pngs
    
    # Arguments
        im_size: image size (n x n)
        dst: path to store the pngs
    # 
        
    ''''       
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            if in_image[i,j,0] >0 and in_image[i,j,1] >0 and 
            in_image[i,j,2] >0:
                blank_png[i,j] = color
    return blank_png



def png_to_geotif(ref_tif_path, input_png_path, output_tif_path):
    
    ''''
    Convert PNG files to Geotiff. The assumption is that there is
    a pre-existing satellite image that you want to copy metadata 
    from to convert a pre-existsing PNG mask of that 
    satellite image into Geotiff format.
    
    # Arguments
        ref_tif_path: satellite image to copy metadata from
        input_png_path: PNG mask to be converted to Geotiff
        output_tif_path: output Geotiff mask
        
    ''''
    
    ref_tif = rasterio.open(ref_tif_path)
    west, south, east, north =data_.ref_tif
    
    input_png = rasterio.open(input_png_path, 'r')
    bands = [1, 2, 3]
    data = input_png.read(bands)
    transform = rasterio.transform.from_bounds(west, south, east, 
                                               north, data.shape[1],
                                               data.shape[2])
    crs = config.crs
    with rasterio.open(output_file_path, 'w', driver='GTiff',
                       width=data.shape[1], height=data.shape[2],
                       count=3, dtype=data.dtype, nodata=0,
                       transform=transform, crs=crs) as dst:
        dst.write(data, indexes=bands)
        
def tile_image(image_path):
    ''''
    Tile large image
    
    # Arguments
        image_path: image path
        chopsize: tile size
    # 
        
    ''''  
    chopsize = config.chopsize   
    for filepath in glob.iglob(image_path):

        tmp =filepath.split('/')[-1].split('.')[0]

        infile = filepath

        infile_= infile.split('/')[-1].split('.')[0]
        img = Image.open(infile)
        img_ = Image.open( config.base + config.image_path + infile_ + '.tif' )

        width, height = img.size

        # Save Chops of original image
        for x0 in range(0, width, chopsize):
            for y0 in range(0, height, chopsize):
                box = (x0, y0,
                     x0+chopsize if x0+chopsize < width else width - 1,
                     y0+chopsize if y0+chopsize < height else height - 1)
                if img.crop(box).size ==(256, 256):
                   # print(img.crop(box).size)
                    img_.crop(box).save(config.base_path+ config.tiled_image_path+
                                        '%s___%03d_%03d.png' % (infile_, x0, y0))
                    img.crop(box).save(config.base_path+ config.tiled_mask_path+
                                       '%s___%03d_%03d.png' % (infile_, x0, y0))