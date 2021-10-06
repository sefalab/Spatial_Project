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


## Check if polygon and georeferenced satellite image overlap
def in_polygon(raster,vector):
    raster = gdal.Open(raster)
    vector = ogr.Open(vector)

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

    gdal.Translate(
        png_path,
        tif_path,
        options=options_string
    )

##Create mask: pixel color references per class
def color_4(val):

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
    ##split shapefile

    if not os.path.exists(shp_folder):
    os.makedirs(shp_folder)
    
    id_=0
    with fiona.open(src) as source:

        meta = source.meta
        print(source[0]['properties']['EA_TYPE'])
        for f in source:

            outfile = os.path.join(shp_folder, str(id_)+ "__%s.shp" % f['properties']['EA_TYPE'].replace(' ','_'))
            id_ = id_+1
            with fiona.open(outfile, 'w', **meta) as sink:

                sink.write(f) 

def create_empty_png(im_size,dst):
   
    Image.MAX_IMAGE_PIXELS = None
    img = Image.new('RGB', (im_size,im_size), (255, 255, 255))
    
    if not os.path.exists(dst):
        img.save(dst, "PNG")
        
def create_mask(in_image,blank_png,color):
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            if in_image[i,j,0] >0 and in_image[i,j,1] >0 and in_image[i,j,2] >0:
                blank_png[i,j] = color
    return blank_png
    
    

shp = '7__Farms.shp'
shp_folder ='/dair/data/2011/shp/train/'
image = '/dair/data/2011/images/2628A.tif'
mask_png = 'final_mask.png'
image_size = 21760

#split shape file by neighbourhood type
splt_shp_file(shp,shp_folder)

#walk through all shape files per neighbourhood type
for _root, _dirs, __files in os.walk(shp_folder):
    for _file in __files:
        if _file.endswith(".shp"): 
            polygon = shp_folder + _file
            print(polygon)

                tif_mask =''.join(polygon.split('/')[-1].split('.')[0],'mask.png')
                
                ## Check if the image overlapse with the polygon
                if in_polygon(image,polygon): 
                    print(image + ':  in poly')
                    #read the polygon
                    with fiona.open(polygon, "r") as shapefile:
                        geoms = [feature["geometry"] for feature in shapefile]
                        type_= [feature_['properties']['EA_TYPE'] for feature_ in shapefile]
                        
                    #Mask out the non-overlapping parts
                    with rasterio.open(image) as src:
                        out_image, out_transform = rasterio.mask.mask(src, geoms,crop= False,invert= True)
                        out_meta = src.meta.copy()

                    out_meta.update({"driver": "GTiff",
                                     "height": out_image.shape[1],
                                     "width": out_image.shape[2],
                                     "transform": out_transform})
                    #save masked image as png
                    with rasterio.open(tif_mask, "w", **out_meta) as dest:
                        dest.write(out_image) 

                    #create blank png
                    create_empty_png(image_size,mask_png)
                    
                    im_tif = cv2.imread(tif_mask)
                    
                    im_mask = cv2.imread(mask_png)
        
                    c_4 = color_4(polygon.split('/')[-1].split('.')[0])
                    cv2.imwrite(create_mask(im_tif,im_mask,c_4))
                
                    c_12 = color_12(polygon.split('/')[-1].split('.')[0])
                    cv2.imwrite(create_mask(im_tif,im_mask,c_12))

                    
                else:
                    print('polygon does not overlap with image')