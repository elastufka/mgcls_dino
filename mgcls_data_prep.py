import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.visualization import astropy_mpl_style
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import patches
import os
import pandas as pd
import glob
import warnings
from skimage.exposure import rescale_intensity
import json

def mapdata_from_fits(fitsfile, return_wcs = True, return_header = False):
    """
    This function reads a FITS file and extracts the data (first dimension only) along with optional WCS and primary header information.

    Parameters:
    fitsfile (str): The path to the FITS file.
    return_wcs (bool): If True, return the WCS information along with the map data. Defaults to True.
    return_header (bool): If True, return the FITS header along with the map data. Defaults to False.

    Returns:
    tuple or numpy.ndarray: Depending on the input parameters, returns either:
                            - (WCS, numpy.ndarray): If return_wcs is True.
                            - (FITS header, numpy.ndarray): If return_header is True.
                            - (WCS, FITS header, numpy.ndarray): If both return_wcs and return_header are True.
                            - numpy.ndarray: If neither return_wcs nor return_header is True."""
    with fits.open(fitsfile) as f:
        header = f[0].header
        wcs = WCS(header)
        dat = f[0].data.squeeze()

    if return_wcs and not return_header:
        return wcs, dat
    elif return_header and not return_wcs:
        return header, dat
    elif return_wcs and return_header:
        return wcs, header, dat
    else: 
        return dat

def file_writer(array, name = None, ext='.fits', header = None):
    """
    This function writes the given array to a file with the specified name and format.
    It supports writing to FITS (.fits) and NumPy (.npy) formats.

    Parameters:
    array (numpy.ndarray): The array to be written to the file.
    name (str): The name of the output file. If None, a default name will be used. Defaults to None.
    ext (str): The file extension indicating the format ('.fits' for FITS, '.npy' for NumPy). Defaults to '.fits'.
    header (astropy.io.fits.Header or None): The FITS header to be included if writing to a FITS file. 
                                              Ignored if writing to a NumPy file. Defaults to None.

    Returns:
    None
    """
    if not ext: 
        ext = name[name.rfind("."):]
    if ext == '.fits' and header:
        hdu = fits.PrimaryHDU(array)
        if header:
            hdu.header = header
        hdul = fits.HDUList([hdu])
        hdul.writeto(f"{name}{ext}")
    elif ext == '.npy':
        print(f"{name}{ext}")
        np.save(f"{name}{ext}", array)

def MGCLS_find_file(source, ftype = "5pln", enhanced = True):
    """This function searches for an MGCLS FITS file associated with the provided source name. The file can be either in the 'basic' or 'enhanced' subdirectory depending on the value of the 'enhanced' parameter.
    
    Assumes the following directory structure:
    data/
        enhanced/
            FITS/
        basic/
            FITS/

    Parameters:
    source (str): The name of the MGCLS source.
    ftype (str): The file type to search for. Defaults to "5pln".
    enhanced (bool): If True, search for the file in the 'enhanced' subdirectory. 
                     If False, search in the 'basic' subdirectory. Defaults to True.

    Returns:
    str or False: The path to the found MGCLS file if found, or False if the file is not found.
    """
    subdir = "enhanced" if enhanced else "basic"
    globstr = f"{subdir}/FITS/{source.strip().replace(' ','*')}"
    if enhanced:
        globstr += f"_*{ftype}*" 
    globstr += ".fits"
    res = sorted(glob.glob(globstr))
    if len(res) == 0 and not globstr.endswith("*.fits"):
        try:
            res = [MGCLS_find_file(f"{source.replace(' ','')}*", ftype = ftype, enhanced=enhanced)]
            foo = res[0]
        except Exception as e:
            return False
    elif len(res) > 1:
        warnings.warn(f"Multiple files found for {source}; using first file in list: {res}") 
    return res[0]

def MGCLS_format_source_name(source):
    source = source.strip().replace("†","")
    if not source.startswith("RXC"):
        source = source.replace(" ", "_")
    else: 
        source = source.replace(" ", "")
    return source

def get_MGCLS_targets():
    """This function reads a CSV file named 'Table1_MGCLS_targets.csv', extracts the 'ClusterName' column,
    and filters out any entries containing 'Bullet'. The resulting list contains the names of MGCLS targets.

    Returns:
    list or None: A list of MGCLS targets if the CSV file is found and processed successfully.
                  Returns None if the CSV file is not found.
    """
    try:
        sources = pd.read_csv("Table1_MGCLS_targets.csv")["ClusterName "]
        bullet = sources.str.contains("Bullet") #Bullet gets designation J0658.8-5556
        sources = sources[~bullet].values
    except FileNotFoundError:
        return None  
    return sources

def MGCLS4ML(parent_dir = '.', ftype = "5pln", crop_shape = (256,256), sname = None, stridex = None, stridey = None, overwrite = False, writenpy = False, output_dir = None):
    """Crops MGCLS FITS images to the desired shape. If no stride is specified, it will be calculated to fit the largest possible integer number of crops with the given shape (assuming symmetric image). Either saves the resulting crops and crop coordinates as numpy files, or returns them.

    Parameters:
    parent_dir (str): The parent directory containing 'basic' and 'enhanced' subdirectories. Defaults to current directory.
    ftype (str): The FITS cube type to search for. Defaults to "5pln".
    crop_shape (tuple): The desired shape (height, width) of the cropped images. Defaults to (256, 256).
    sname (str): Name of individual source to extract crops and coordinates for. Defaults to None.
    stridex (int): The stride along the x-axis for cropping. Defaults to None.
    stridey (int): The stride along the y-axis for cropping. Defaults to None.
    overwrite (bool): If True, overwrite existing cropped images. Defaults to False.
    log (bool): If True, enable logging to a file named 'MGCLS4ML_log.txt' in `parent_dir`. Defaults to True.
    crop_dir (str): The directory to save cropped images. Defaults to None, which saves in the current directory.

    Raises:
    AssertionError: If 'basic' or 'enhanced' subdirectories are missing in `parent_dir`.
    IndexError: If basic or enhanced files are missing for any source.

    Returns:
    tuple or None: Tuple of numpy arrays (crops, coords) if writenpy = False, else None.
    """
    sources = get_MGCLS_targets() #get source list - can get it from the web if not in dir
    os.chdir(parent_dir)

    if sname is not None: #just do one given source
        sources = [sname]

    for source in sources:
        sname = MGCLS_format_source_name(source)
        
        #fix inconsistencies between source name and file names
        if sname == "RXCJ1314.4-2515":
            sname = "RXCJ1314"
        elif sname == "RXCJ0225.1-2928":
            sname = "RXCJ0225.1-22928"
        elif sname == "El_Gordo":
            sname = "ElGordo"
        try:
            enhanced = MGCLS_find_file(sname, ftype, enhanced = True)
        except IndexError:
            print(f"Skipping source {sname}, file not found..")
            continue
        if os.path.exists(f"{enhanced[:-5]}_crops.npy") and not overwrite:
            print(f"Crops {enhanced[:-5]}_crops.npy already exist and overwrite=False!")
            continue

        print(f"Using file {enhanced}")
        ewcs, earr = mapdata_from_fits(enhanced)
        if not stridex and not stridey:
            n = earr[0].shape[0]//crop_shape[0] #assuming square image for now
            stridex = calc_stride(earr[0].shape[0], crop_shape[0], n+1) #some overlap is okay, but minimize it
            stridey = stridex
        crops, coords = crops_and_coords(ewcs, earr, crop_shape = crop_shape, stridex = stridex, stridey = stridey)
        if writenpy:
            fname = enhanced[enhanced.rfind("/")+1:enhanced.find(".fits")]
            if output_dir is None:
                output_dir = parent_dir 
            for i,c in enumerate(crops):
                np.save(f"{os.path.join(output_dir, fname)}_crop_{i}.npy",c.data) #don't save header for now...
            #file_writer(crops, f"{os.path.join(output_dir,enhanced[:-5])}_crops", ext='.npy')
            file_writer(coords, f"{os.path.join(output_dir,fname)}_coords", ext='.npy')
        #elif writefits: #write FITS files instead
            #TBI
        else:
            # testcrop = crops[50]
            # testwcs = WCS(testcrop.header)
            # #print(testwcs.celestial.pixel_to_world(0,0).to_string("hmsdms"), testwcs.celestial.pixel_to_world(255,255).to_string("hmsdms"))
            # ax=plt.subplot(projection=testwcs.celestial)
            # #ax = plt.gca()
            # #ra = ax.coords[0]
            # #ra.set_format_unit('degree')
            # plt.imshow(testcrop.data,vmin=1e-7,vmax=1e-4)
            # plt.savefig(f"{sname}_test_crop2.png")
            # print(coords[50][0].to_string("hmsdms"))
            # print(coords[50][1].to_string("hmsdms"))
            return crops, coords
        
def crops_and_coords(wcs, dat, crop_shape = (256,256), stridex = None, stridey = None, reject_nan = True, percent_nan = .4, header_stats = None):
    """
    Get coordinates for top-left and bottom-right of crops from MGCLS enhanced image product.

    This function extracts crops from the MGCLS enhanced image product and calculates the coordinates of the top-left and bottom-right corners of each crop.

    Parameters:
    wcs (astropy.wcs.WCS): The WCS (World Coordinate System) of the image.
    dat (numpy.ndarray): The data array of the MGCLS image.
    crop_shape (tuple): The shape (height, width) of the crop. Defaults to (256, 256).
    stridex (int or None): The stride along the x-axis for cropping. Defaults to None.
    stridey (int or None): The stride along the y-axis for cropping. Defaults to None.
    reject_nan (bool): If True, reject crops containing NaN values. Defaults to True.
    percent_nan (float): The threshold percentage of NaN values in a crop to reject it. Defaults to 0.4.

    Returns:
    tuple: A tuple containing two lists:
           - A list of HDU objects, each containing a crop of the MGCLS image.
           - A list of tuples, each containing the coordinates of the top-left and bottom-right corners of a crop.
    """    
    if dat.ndim == 2:
        imshape = dat.shape
    elif len(dat) > 1: #5pln cubes
        dat = dat[0] 
        imshape = dat.shape
    else:
        dat = dat[0,0,:,:]
        imshape = dat.shape[-2:]
    nanmask = np.ma.masked_invalid(dat).mask
    
    crops, coords = [], []
    xpix, ypix = imshape
    xhalf = crop_shape[1]/2
    yhalf = crop_shape[0]/2
    x,y = 0,0
    bad_crops = 0
    #i=0
    
    while y < ypix:
        x = 0
        while x < xpix:
            #i+=1
            pos = (x + xhalf, y + yhalf) 
            skypos = wcs.celestial.pixel_to_world(*pos)
            cc = Cutout2D(dat, skypos, crop_shape, mode = 'partial', wcs = wcs.celestial) #should be crop_shape.T but if symmetric ok
            nn = np.ma.masked_invalid(cc.data).mask 
            if cc.data.shape != crop_shape:
                break
            
            #combine header and data into HDU object and return that
            hdu = fits.PrimaryHDU(data=cc.data, header = cc.wcs.to_header())
            
            if reject_nan and np.sum(nn) == 0: 
                crops.append(hdu)
                coords.append(bltr_coords(wcs.celestial,x,y,crop_shape))
                # if i == 50 or i==100:
                #     print("coords from global WCS:")
                #     print(bltr_coords(wcs.celestial,x,y,crop_shape)[0].to_string("hmsdms"),bltr_coords(wcs.celestial,x,y,crop_shape)[1].to_string("hmsdms"))
                #     print("coords from cutout WCS:")
                #     testwcs = WCS(hdu.header)
                #     print(testwcs.celestial.pixel_to_world(0,0).to_string("hmsdms"), testwcs.celestial.pixel_to_world(255,255).to_string("hmsdms"))
                    
            elif not reject_nan: #keep if fraction of non-nans is below threshold
                pnan = np.sum(nn)/np.product(nn.shape)
                hdu.header["PERCENT_NAN"] = pnan
                if pnan < percent_nan: 
                    crops.append(hdu)
                    coords.append(bltr_coords(wcs,x,y,crop_shape))
            else:
                bad_crops +=1
                #i-=1
            if not stridex:
                x += crop_shape[0]
            else:
                x += stridex
        if not stridey:
            y += crop_shape[0]
        else:
            y += stridey
    print(f"{bad_crops} crops rejected due to NaN!")
    return crops, coords

def bltr_coords(wcs, x,y,crop_shape):
    """
    Get celestial coordinates for the bottom-left and top-right corners of a crop.

    This function calculates the celestial coordinates (Right Ascension and Declination) 
    corresponding to the bottom-left and top-right corners of a crop based on the provided WCS.

    Parameters:
    wcs (astropy.wcs.WCS or None): The WCS (World Coordinate System) transformation to use for coordinate conversion.
                                    If None, returns pixel coordinates instead of celestial coordinates.
    x (int): The x-coordinate of the bottom-left corner of the crop.
    y (int): The y-coordinate of the bottom-left corner of the crop.
    crop_shape (tuple): The shape (height, width) of the crop.

    Returns:
    list: A list containing two tuples:
          - The celestial coordinates (RA, Dec) of the bottom-left corner.
          - The celestial coordinates (RA, Dec) of the top-right corner.
    """
    if not wcs:
        return [(x,y),(x+crop_shape[0],y+crop_shape[1])] #pixel coords
    else:
        blc = wcs.pixel_to_world(x,y)
        trc = wcs.pixel_to_world(x+crop_shape[0],y+crop_shape[1])
        return [[blc, trc]]
    
def calc_stride(dim, crop_dim, n):
    """
    Calculate the stride length for evenly dividing a dimension into 'n' crops.

    This function calculates the stride length required to evenly divide a dimension into 'n' crops of size 'crop_dim'.
    It adjusts the stride to ensure that the crops cover the entire dimension.

    Parameters:
    dim (int): The dimension to be divided.
    crop_dim (int): The size of each crop dimension.
    n (int): The number of crops to be generated.

    Returns:
    int: The calculated stride length.
    """
    ideal = crop_dim*n
    rest = ideal - dim
    margin = rest//(n-1)
    stride = crop_dim - margin
    return stride

def MGCLS_FITS_metadata(data_dir='.', ftype='5pln', plow=2, phigh=98):
    sources = get_MGCLS_targets() #get source list - can get it from the web if not in dir
    os.chdir(data_dir)

    headlist = []
    for source in sources:
        sname = MGCLS_format_source_name(source)
        
        #fix inconsistencies between source name and file names
        if sname == "RXCJ1314.4-2515":
            sname = "RXCJ1314"
        elif sname == "RXCJ0225.1-2928":
            sname = "RXCJ0225.1-22928"
        elif sname == "El_Gordo":
            sname = "ElGordo"
        try:
            enhanced = MGCLS_find_file(sname, ftype, enhanced = True)
        except IndexError:
            print(f"Skipping source {sname}, file not found..")
            continue

        print(f"Using file {enhanced}")
        _, ehead, earr = mapdata_from_fits(enhanced, return_header=True)
        kk = [k for k in ehead.keys()]
        vv = [k for k in ehead.values()]
        dd = {k:v for k,v in zip(kk,vv)}
        df = pd.DataFrame(dd, index = pd.Index([0]))
        df["source_name"] = sname
        df["filename"] = enhanced[enhanced.rfind("/")+1:]
        if "P2" not in df.keys():
            df["P2"] = np.nanpercentile(earr[0],plow)
        if "P98" not in df.keys():
            df["P98"] = np.nanpercentile(earr[0],phigh)
        headlist.append(df)
    meta = pd.concat(headlist)
    meta.to_csv("MGCLS_FITS_metadata.csv")

def scale_crops(crop_dir, out_dir, meta, method="cs", **kwargs):
    """
    This function scales the crops located in the 'crop_dir' directory according to the parameters extracted from the entire image.
    It reads metadata from the CSV file specified in the 'meta' parameter.

    Parameters:
    crop_dir (str): The directory containing the crops to be scaled.
    out_dir (str): The directory where scaled crops will be saved.
    meta (str): The path to the CSV file containing metadata (dataframe with metadata).
    method (str): The scaling method to be used. Defaults to "cs" (contrast_stretch).
                  Other supported methods include "powerlaw".
    **kwargs: Additional keyword arguments to be passed to the scaling method.

    Returns:
    None
    """
    meta = pd.read_csv(meta) #dataframe with metadata
    if 'file_prefix' not in meta.keys():
        meta['file_prefix'] = [m[:m.rfind('.fits')] for m in meta.filename]
    crops = sorted(glob.glob(f"{os.path.join(crop_dir, '*crop*')}"))
    cprev=''

    for c in crops:
        ccurrent = c[c.rfind('/')+1:c.find("crop")-1]
        if ccurrent != cprev:
            pvals = meta.where(meta.file_prefix == ccurrent).dropna(how='all')[['P2','P98']].values
        cdat = np.load(c) 
        if method == 'cs': #contrast_stretch
            cnew = rescale_intensity(cdat, in_range=tuple(pvals[0]),out_range=(0,1))
        #elif method == 'zscale':
        #    zint = ZScaleInterval(**kwargs)
        #    zdat = zint() #original fits data, scaled
        elif method == "powerlaw":
            #replace lowest and highest 2% of values
            p2 = pvals[0][0]
            p98 = pvals[0][1]
            cdat[cdat < p2] = p2
            cdat[cdat > p98] = p98
            I_norm = (cdat/p98)**power
            cnew = (I_norm - 0.5)/0.5
        #save scaled crop
        np.save(f"{os.path.join(out_dir,c[c.rfind('/')+1:])}",cnew)
        cprev = ccurrent

def MGCLS_fmt_cat4coco(filename):
    """Format MGCLS compact source catalog for pyBDSF_to_COCO.

    This function reads a compact source catalog from the specified file and formats it to be compatible with the pyBDSF_to_COCO tool.
    It converts angular measurements from arcseconds to degrees and renames columns to match the expected format.

    Parameters:
    filename (str): The path to the input compact source catalog file.

    Returns:
    None
    """
    df = pd.read_csv(filename)
    df['smax_asec'] = [float(row.smax_asec)*u.arcsec.to(u.deg) for _, row in df.iterrows()] 
    df['smin_asec'] = [float(row.smin_asec)*u.arcsec.to(u.deg) for _, row in df.iterrows()]
    df.rename(columns={"RA_deg":"RA","Dec_deg":"DEC","smax_asec":"Maj","smin_asec":"Min","spa_deg":"PA"}, inplace=True)
    units=["","DEG","DEG","DEG","DEG","mJy","mJy","mJyb","mJyb","DEG","DEG","DEG",""]
    df.columns = pd.MultiIndex.from_tuples([*zip(df.columns, units)])
    df.to_csv(filename)

def crop_catalog_aggs(cats):
    """
    Calculate catalog statistics for a list of COCO JSON files.

    This function takes a list of COCO JSON files representing compact source catalogs and calculates various statistics
    to be used as metadata. It includes the number of sources in each crop, average flux, average size, and more.

    Parameters:
    cats (list): A list of paths to COCO JSON files.

    Returns:
    pandas.DataFrame: A DataFrame containing catalog statistics aggregated from the input COCO JSON files.
                      The DataFrame includes columns such as 'iscrowd', 'area', 'root_filename', and aggregated statistics.
    """
    catlist = []
    for cat in cats:
        with open(cat) as f:
           ff = json.load(f)
        anns = ff['annotations']
        df = pd.DataFrame(anns)
        if not df.empty:
            #groupby image/crop id, do some aggregations
            gdf = df.groupby(df.image_id)[["iscrowd","area"]].agg(["count","sum","mean"])
            gdf["root_filename"] = cat[:cat.rfind("annotations")]
            catlist.append(gdf)
        else: 
            print(f"No annotations found for {cat}!")
    cropdf = pd.concat(catlist)
    return cropdf

    def merge_catalog_meta():
        pass

def catalog_qc(coco_json,im_dir = ".", segmentations=False, **kwargs):
    """plot all crops and catalogs to check that they are ok"""
    with open(coco_json,'r') as f:
        jj = json.load(f)
    imdf = pd.DataFrame(jj['images'])
    anndf = pd.DataFrame(jj['annotations'])
    
    #get all images in current folder
    ims = glob.glob(os.path.join(im_dir,"*.npy"))
    if os.path.exists("train"):
        ims.extend(glob.glob("train/*.npy"))
    if os.path.exists("val"):
        ims.extend(glob.glob("val/*.npy"))

    ims2 = [i[i.rfind('/')+1:] for i in ims]

    these_files = []
    for f in imdf.file_name.values:
        try:
            these_files.append(ims[ims2.index(f)])
        except ValueError:
            continue
        
    plt.style.use(astropy_mpl_style)
    with PdfPages(f"{coco_json[:coco_json.rfind('_')]}_qc.pdf") as pp:
        npages = 1
        while len(these_files) >  0:
            fig = plot_image_catalog_page(these_files[:30],anndf, segmentations=segmentations, **kwargs)
            pp.savefig(fig)
            plt.close()
            these_files = these_files[30:]
            npages +=1
        
def plot_image_catalog_page(images, annotations, bounding_boxes = True, segmentations = False, **kwargs):
    """Plot the image and the associated bounding boxes and/or segmentations on top for verification"""
    n=6
    fig,ax = plt.subplots(n,5, figsize = (8.3,11.7))
    for i,image in enumerate(images):
        dat = np.load(image)
        crop_number = int(image[image.rfind("_")+1:image.rfind(".")])
        jdf = annotations.where(annotations.image_id == crop_number).dropna()

        if bounding_boxes:
            bboxes = jdf.bbox
        if segmentations:
            segs = jdf.segmentation
 
        ax[i%n][i//n].imshow(dat, **kwargs)
        if bounding_boxes:
            for box in bboxes:
                #draw rectangle
                rect = patches.Rectangle((box[0],box[1]),box[2],box[3], edgecolor='c', facecolor='none', linewidth=0.5)
                ax[i%n][i//n].add_patch(rect)
        if segmentations:
            for seg in segs:
                #draw ellipse
                if len(seg) == 1:
                    segx = [x for i,x in enumerate(seg[0])if i%2 == 0]
                    segy = [x for i,x in enumerate(seg[0])if i%2 == 1]
                else: 
                    segx = seg[0]
                    segy = seg[1]
                ax[i%n][i//n].plot(segx,segy,'m',linewidth=0.5)
        ax[i%n][i//n].axis('off')
        ax[i%n][i//n].set_title(crop_number)
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    #example usage for the source Abell 13
    datadir = "/home/users/l/lastufka/scratch/MGCLS_data"
    output_dir = "/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep"
    output_dir_cs = "/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs"
    metafile = "/home/users/l/lastufka/scratch/MGCLS_data/enhanced/MGCLS_FITS_metadata.csv"
    #MGCLS4ML(datadir, output_dir=output_dir,  writenpy=True) #sname="Abell_13",
    #MGCLS_FITS_metadata() #generate metadata file from FITS headers if one does not yet exist
    scale_crops(output_dir,  output_dir_cs, meta = metafile)
    # try:
    #     #pyBDSF compact source catalogs to COCO format
    #     os.system(f"python ~/pyBDSF_to_COCO/pyBDSF_to_COCO.py --image {datadir}/enhanced/FITS/Abell_13_noFix_pol_I_Farcsec_5pln_cor.fits --catalog {datadir}/compact_catalogs/Abell-13_compact_source_catalog.csv --crop_coords {output_dir}/Abell_13_noFix_pol_I_Farcsec_5pln_cor_coords.npy --output_file {output_dir}/Abell_13_noFix_pol_I_Farcsec_5pln_cor_annotations.json --crop_prefix Abell_13_noFix_pol_I_Farcsec_5pln_cor_ --category_names source")
    # except Exception as e:
    #     print("No COCO annotations generated!")
    #     sys.exit(1)
    # catalog_qc(f"{output_dir}/Abell_13_noFix_pol_I_Farcsec_5pln_cor_annotations.json",im_dir=output_dir_cs)
    # cagg = crop_catalog_aggs([f"{output_dir}/Abell_13_noFix_pol_I_Farcsec_5pln_cor_annotations.json"])
    # print(cagg.describe()) #see some overall statistics
    # add per-crop source counts, etc to metadata

    #sname="Abell_13",El_Gordo file not found