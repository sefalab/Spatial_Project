import multiprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fiona
import rasterio
import os,csv
from rasterio.mask import mask
from osgeo import gdal, osr, ogr
import os.path
from PIL import Image
from utils import config
Image.MAX_IMAGE_PIXELS = None


#walk through all image files 
for root_, dirs, files__ in os.walk(config.base_path+config.blank_images):
    files__.reverse()
    for file_ in files__:
        if file_.endswith(".tif"):  
            image = os.path.join(root_, file_)
            
            #walk through all shape files 
            for _root, _dirs, __files in os.walk(config.base_path + config.shp+ "shp_farm/"):
                __files.reverse()
                for _file in __files:
                    if _file.endswith(".shp"): 
                        polygon = config.base_path + config.shp + 'shp_farm/' + _file
                        if in_polygon(image,polygon): #find the image the polygon corresponds with 
#                             print(image + ':  in poly')
#                             print(polygon)
                            _type =_file.split('.')[0]
                            with fiona.open(polygon, "r") as shapefile:
                                geoms = [feature["geometry"] for feature in shapefile]
                                type_= [feature_['properties']['EA_TYPE'] for feature_ in shapefile]

                            if not os.path.isfile("".join([config.base_path, config.final,'done_pngs/',image.split('/')[-1].split('.')[0],'-',polygon.split('/')[-1].split('.')[0],'-',type_[0],'_val.png'])):   

                                with rasterio.open(image) as src:
                                    out_image, out_transform = rasterio.mask.mask(src, geoms,crop= False)
                                    out_meta = src.meta.copy()

                                out_meta.update({"driver": "GTiff",
                                                 "height": out_image.shape[1],
                                                 "width": out_image.shape[2],
                                                 "transform": out_transform})
                                #save mask in masks folder (mask/ imagename - polygon_id.tif)

                                t_= "".join([config.base_path, config.final,'done_pngs/',image.split('/')[-1].split('.')[0],'-',polygon.split('/')[-1].split('.')[0],'-',type_[0],'_val.png'])
                                x = np.moveaxis(out_image,0,-1)

#                                 print(x.shape)
                                
                                fin =Image.fromarray(x)
                                tmp =t_.replace(' ', '_')    
                                fin.save(tmp)
                                print(file_)
                                print('-------------')

                        else:
                            print('not in')

