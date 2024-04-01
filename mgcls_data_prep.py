import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
#from astropy.coordinates import SkyCoord as sc
from astropy.nddata import Cutout2D
from astropy import units as u
from matplotlib import pyplot as plt
import os
import pandas as pd
import glob
import warnings
from skimage.exposure import rescale_intensity
import json

def mapdata_from_fits(fitsfile, return_wcs = True, return_header = False):
    """
    Extract map data from a FITS file.

    This function reads a FITS file and extracts the map data along with optional WCS and header information.

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
        #print(f.info())
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
    Write an array to a file with the specified format.

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

# def update_headers(imfolder, arrfolder):
#     bb=glob.glob(f"{arrfolder}/*.npy")
#     bb = [b[:b.find("_crops")] for b in bb]
#     print(len(bb))
#     sources = list(np.unique(bb))
#     sources = [f"{imfolder}{s[s.rfind('/'):]}.fits" for s in sources]
#     print(len(sources))
#     for s in sources:
#         with fits.open(s, mode='update') as f:
#             header = f[0].header
#             if not "DATAMEAN" in header:
#                 data = f[0].data
#                 header.append(("DATAMEAN",np.nanmean(data)))
#                 header.append(("DATASTD",np.nanstd(data)))
#             f.flush()

# def normalize_crops(imfolder, arrfolder):
#     aa=glob.glob(f"{arrfolder}/*.npy")
#     bb = [b[:b.find("_crops")] for b in aa]
#     print(len(bb))
#     sources = list(np.unique(bb))
#     sources = [f"{imfolder}{s[s.rfind('/'):]}.fits" for s in sources]
#     print(len(sources))
#     for s in sources:
#         sourcename = s[s.rfind('/'):-5]
#         print(sourcename)
#         with fits.open(s) as f:
#             header = f[0].header
#             if not "DATAMEAN" in header:
#                 print(s)
#             dmean = header["DATAMEAN"]
#             dstd = header["DATASTD"]
#         cc = [b for b in aa if sourcename in b]
#         for c in cc:
#             arr = np.load(c)
#             arr -= dmean
#             arr /= dstd
#             np.save(c,arr)

# def split_basic_cube(basic):
#     pass

# def split_enhanced_cube(enhanced, ext = '.fits'):
#     """write to separate fits files or numpy arrays"""
#     if "5pln" in enhanced: #get intensity image, first one
#         header, dat = mapdata_from_fits(enhanced, return_wcs = False, return_header= True)
#         file_writer(dat[0], name = f"{enhanced[:-5]}_cont", ext = ext, header = header)
#     elif "fcube" in enhanced: #get all 
#         cfreqs = [908, 952, 996, 1044, 1093, 1145, 1318, 1382, 1448, 1482, 1594, 1656]
#         header, dat = mapdata_from_fits(enhanced, return_wcs = False, return_header= True)
#         for i,d in enumerate(dat):
#             file_writer(d, name = f"{enhanced[:-5]}_{cfreqs[i]}MHz", ext = ext, header = header)

def MGCLS_find_file(source, ftype = "5pln", enhanced = False):
    """This function searches for an MGCLS file associated with the provided source name. The file can be either in 
    the 'basic' or 'enhanced' subdirectory depending on the value of the 'enhanced' parameter.

    Parameters:
    source (str): The name of the MGCLS source.
    ftype (str): The file type to search for. Defaults to "5pln".
    enhanced (bool): If True, search for the file in the 'enhanced' subdirectory. 
                     If False, search in the 'basic' subdirectory. Defaults to False.

    Returns:
    str or False: The path to the found MGCLS file if found, or False if the file is not found.
    """
    subdir = "enhanced" if enhanced else "basic"
    globstr = f"{subdir}/FITS/{source.strip().replace(' ','*')}"
    if enhanced:
        globstr += f"_*{ftype}*" #remove first * from *_*
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
    """Prepare MGCLS FITS images for ML

    Parameters:
    parent_dir (str): The parent directory containing 'basic' and 'enhanced' subdirectories. Defaults to current directory.
    ftype (str): The file type to search for. Defaults to "5pln".
    crop_shape (tuple): The shape (height, width) of the cropped images. Defaults to (256, 256).
    stridex (int): The stride along the x-axis for cropping. Defaults to None.
    stridey (int): The stride along the y-axis for cropping. Defaults to None.
    overwrite (bool): If True, overwrite existing cropped images. Defaults to False.
    log (bool): If True, enable logging to a file named 'MGCLS4ML_log.txt' in `parent_dir`. Defaults to True.
    crop_dir (str): The directory to save cropped images. Defaults to None, which saves in the current directory.

    Raises:
    AssertionError: If 'basic' or 'enhanced' subdirectories are missing in `parent_dir`.
    IndexError: If basic or enhanced files are missing for any source.

    Returns:
    None
    """
    sources = get_MGCLS_targets() #get source list - can get it from the web if not in dir
    os.chdir(parent_dir)

    if sname: #just do one given source
        sources = [sname]

    for source in sources:
        sname = MGCLS_format_source_name(source)
        
        #fix inconsistencies between source name and file names
        if sname == "RXCJ1314.4-2515":
            sname = "RXCJ1314"
        elif sname == "RXCJ0225.1-2928":
            sname = "RXCJ0225.1-22928"
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
            stridex = calc_stride(earr[0].shape[0], crop_shape[0], n)
            stridey = stridex
            #print(stridex)
        #crops, coords = MGCLS_crop_coords(ewcs, earr, crop_shape = crop_shape, stridex = stridex, stridey = stridey)
        crops, coords = crops_and_coords(ewcs, earr, crop_shape = crop_shape, stridex = stridex, stridey = stridey)
        if writenpy:
            if output_dir is None:
                output_dir = parent_dir 
            for i,c in enumerate(crops):
                np.save(f"{os.path.join(output_dir, enhanced[:-5])}_crop_{i}.npy",c)
            #file_writer(crops, f"{os.path.join(output_dir,enhanced[:-5])}_crops", ext='.npy')
            file_writer(coords, f"{os.path.join(output_dir,enhanced[:-5])}_coords", ext='.npy')
        #elif writefits: #write FITS files instead
            #TBI
        else:
            testcrop = crops[50]
            testwcs = WCS(testcrop.header)
            #print(testwcs.celestial.pixel_to_world(0,0).to_string("hmsdms"), testwcs.celestial.pixel_to_world(255,255).to_string("hmsdms"))
            ax=plt.subplot(projection=testwcs.celestial)
            #ax = plt.gca()
            #ra = ax.coords[0]
            #ra.set_format_unit('degree')
            plt.imshow(testcrop.data,vmin=1e-7,vmax=1e-4)
            plt.savefig(f"{sname}_test_crop2.png")
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
    header_stats (dict or None): Statistics calculated from the header to be included in the FITS header of each crop. Defaults to None.

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

    # if isinstance(header_stats,dict):
    #     statdict = {}
    #     for statname, statfn in header_stats.items(): #calculate stats of original image for use later
    #         if statname == "percentile":
    #             if isinstance(statfn, list):
    #                 for p in statfn:
    #                     statdict[f"percentile{p}"] = np.nanpercentile(dat, p)
    #             else:
    #                 statdict[f"percentile{statfn}"] = np.nanpercentile(dat, statfn)
    #         else:
    #             statdict[statname] = statfn(dat)
    i=0
    while y < ypix:
        x = 0
        while x < xpix:
            i+=1
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
                i-=1
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
        return [blc, trc]
    
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

# def MGCLS_crop_coords(wcs, dat, crop_shape = (256,256), stridex = None, stridey = None, reject_nan = True, percent_nan = .4):
#     """Get coordinates for top-left and bottom-right of crops from MGCLS enhanced image product"""
#     if dat.ndim == 2:
#         imshape = dat.shape # (y,x) because FITS
#     elif len(dat) > 1: #5pln cubes
#         dat = dat[0] 
#         imshape = dat.shape
#     else: #fcubes
#         dat = dat[0,0,:,:]
#         imshape = dat.shape[-2:]
#     nanmask = np.ma.masked_invalid(dat).mask
    
#     crops, coords = [], []
#     xpix, ypix = imshape
#     x,y = 0,0

#     while y < ypix:
#         x = 0
#         while x < xpix:
#             cc = dat[y:y+crop_shape[0],x:x+crop_shape[1]] #assuming from FITS file, x is last axis)
#             nn = nanmask[y:y+crop_shape[0],x:x+crop_shape[1]]
#             if cc.shape != crop_shape:
#                 break
#             if reject_nan and np.sum(nn) == 0:
#                 crops.append(cc)
#                 coords.append(bltr_coords(wcs,x,y,crop_shape))
#             elif not reject_nan: #keep if fraction of non-nans is below threshold
#                 pnan = np.sum(nn)/np.product(nn.shape)
#                 if pnan < percent_nan: 
#                     crops.append(cc)
#                     coords.append(bltr_coords(wcs,x,y,crop_shape))
#             if not stridex:
#                 x += crop_shape[0]
#             else:
#                 x += stridex
#         if not stridey:
#             y += crop_shape[0]
#         else:
#             y += stridey
#     return crops, coords

# def MIGHTEE_crops(arr, crop_shape = (256,256), stridex = None, stridey = None, reject_nan = True, percent_nan = .4):
#     """wrapper for MGCLS_crop_coords that does the cropping for MIGHTEE"""
#     if isinstance(arr, str):
#         arr = np.load(arr)
    
#     crops, coords = MGCLS_crop_coords(None, arr, crop_shape = crop_shape, stridex = stridex, stridey = stridey, reject_nan = reject_nan, percent_nan = percent_nan)
#     return crops, coords

# def check_crop_extent(bpix, crop_shape = (256,256)):
#     """Check that extent of all potential crops is the same to within 1 pix"""
#     bcorr = []
#     for b in bpix:
#         bl, tr = b
#         blx, bly, xtr, ytr= int(np.round(bl[0],0)), int(np.round(bl[1],0)),int(np.round(tr[0],0)),int(np.round(tr[1],0))
#         ext = (xtr - blx, ytr - bly)
#         if ext[0] != crop_shape[0] or ext[1] != crop_shape[1]:
#             #print(ext)
#             newc = fix_crop_extent(bl,tr, blx,bly, xtr, ytr, crop_shape = crop_shape)
#             #print("new extent", newc[1][0] - newc[0][0], newc[1][1]-newc[0][1])
#             assert (newc[1][0] - newc[0][0], newc[1][1]-newc[0][1]) == crop_shape
#             bcorr.append(newc)
#         else:
#             bcorr.append(([blx,bly],[xtr,ytr]))
#     return bcorr
            
# def fix_crop_extent(bl, tr, blx,bly,xtr,ytr, crop_shape= (256,256)):
#     newbl, newtr = [blx,bly], [xtr,ytr]
#     if xtr - blx > crop_shape[0]: #x extent greater than it should be
#         if bl[0]%1 > tr[0]%1: #round up blx
#             newbl[0] += 1
#         else:
#             newtr[0] -= 1
#     if xtr - blx < crop_shape[0]: #x extent less than it should be
#         if bl[0]%1 > tr[0]%1: #round up blx
#             newbl[0] -= 1
#         else:
#             newtr[0] += 1    
#     if ytr - bly > crop_shape[1]:
#         if bl[1]%1 > tr[1]%1: #round up blx
#             newbl[1] += 1
#         else:
#             newtr[1] -= 1
#     if ytr - bly < crop_shape[1]:
#         if bl[1]%1 > tr[1]%1: #round up blx
#             newbl[1] -= 1
#         else:
#             newtr[1] += 1
#     return (newbl, newtr)

# def crop_pixels(arr, pix):
#     crops = []
#     for p in pix:
#         bl, tr = p
#         crops.append(arr[bl[0]:tr[0],bl[1]:tr[1]])
#     return crops

# def calc_unused(barr,bpix):
#     """Calculate percentage of image that is unused, ie not included in a crop (and not counting NaN pixels)"""
#     nanmask = np.ma.masked_invalid(barr[0]).mask #1 if nan
#     #print(bpix[0],bpix[-1])
#     for b in bpix:
#         bl, tr = b
#         nanmask[bl[0]:tr[0],bl[1]:tr[1]] = 1
#     corresponding_mask = nanmask[bpix[0][0][0]:bpix[-1][0][0],bpix[0][1][0]:bpix[-1][0][1]]
#     print(f"{(1 - np.sum(corresponding_mask)/np.product(corresponding_mask.shape))*100:.2f}% of the Basic image product is not included in a crop.")

# def MGCLS_source_labels(dirname):
#     """construct label vector based on source list, for clustering visualization purposes"""
#     snames = sorted(glob.glob(f"{dirname}/*crop*.npy")) #hopfeully this is the same sorting as dataloader
#     targets = get_MGCLS_targets()
#     ftargets = [MGCLS_format_source_name(t) for t in targets]
#     source_key = {"source":[],"index":[]}
#     for s in snames:
#         for i,t in enumerate(ftargets):
#             if f"{t}_" in s:
#                 source_key["source"].append(t)
#                 source_key["index"].append(i)
#     return source_key

# def MGCLS_crop_index(dirname):
#     """construct label vector based on crop index, for clustering visualization purposes"""
#     snames = sorted(glob.glob(f"{dirname}/*crop*.npy")) #hopfeully this is the same sorting as dataloader
#     indices = []
#     for s in snames:
#         indices.append(int(s[s.find("_crops_")+7:s.rfind(".")]))
#     return indices

# def crop_source_mask(wcs, df, crop_number, crop_coords, maj_only = True, crop_shape = (256,256)):
#     empty = np.zeros(crop_shape)
#     bl = wcs.world_to_pixel(crop_coords[0][0])
#     tr = wcs.world_to_pixel(crop_coords[1][0])
#     print(bl,tr)
#     sra = df.where(df.crop == crop_number).dropna(how='all')["RA"].values
#     sra = [s[0] for s in sra]
#     sdec = df.where(df.crop == crop_number).dropna(how='all')["DEC"].values
#     sdec = [s[0] for s in sdec]
#     smaj = df.where(df.crop == crop_number).dropna(how='all')["IM_MAJ"].values
#     smaj = np.array([s[0] for s in smaj])
#     #ssc = SkyCoord(sra*u.deg, sdec*u.deg)
#     #spix = wcs.world_to_pixel(ssc)
#     sbottomleft = SkyCoord((sra-smaj/2)*u.deg, (sdec-smaj/2)*u.deg)
#     stopright = SkyCoord((sra+smaj/2)*u.deg, (sdec+smaj/2)*u.deg)
#     sblp = wcs.world_to_pixel(sbottomleft)
#     strp = wcs.world_to_pixel(stopright)
    
#     #adjust to 0,256
#     sblx = np.round(sblp[0]- bl[0]).astype(int)
#     sbly = np.round(sblp[1] - bl[1]).astype(int)
#     strx = np.round(strp[0]- bl[0]).astype(int)
#     stry = np.round(strp[1] - bl[1]).astype(int)
#     for x1,y1,x2,y2 in zip(sblx,sbly,strx,stry):
#         xlow = min(x1,x2)
#         xhigh = max(x1,x2)
#         if xhigh > crop_shape[0]-1:
#             xhigh = crop_shape[0] -1
#         ylow = min(y1,y2)
#         yhigh = max(y1,y2)
#         if yhigh > crop_shape[1] - 1:
#             yhigh = crop_shape[1]- 1
#         xx,yy = np.meshgrid(range(xlow,xhigh),range(ylow,yhigh))
#         empty[yy,xx] = 1
#     #
#     return empty

# def coords_to_crops(coords, sourcearr, wcs, output_folder, crop_shape = (256,256), prefix='', indices=None):
#     """given a list of SkyCoords, get the corresponding crops. Useful when no more NaNs left in source array due to processing"""
#     cc = np.load(coords, allow_pickle = True)
#     if not isinstance(sourcearr, np.ndarray):
#         im = np.load(sourcearr) #verify axis order...
#     else: 
#         im = sourcearr
#     #print(im.shape) #should be fine
#     bpix = MGCLS_corresponding_pix(cc, wcs, crop_shape = crop_shape)
#     if indices is None:
#         iterator = enumerate(bpix)
#     else: 
#         iterator = zip(indices, bpix)
#     for i,b in iterator:
#         crop = im[b[0][1]:b[1][1],b[0][0]:b[1][0]] #assume FITS order
#         #print(f"{output_folder}/{prefix}crop_{i}.npy", b, crop.shape)
#         np.save(f"{output_folder}/{prefix}crop_{i}.npy", crop)

# def MGCLS_image_percentiles(dir, plow, phigh):
#     ff = glob.glob(f"{dir}/*.fits")
#     fn, low, high = [],[],[]
#     for f in ff:
#         with fits.open(f) as o:
#             dat = o[0].data
#             low.append(np.percentile(dat, plow))
#             high.append(np.percentile(dat, phigh))
#         fn.append(f[:-5])
#     df = pd.DataFrame({"file_name":fn, f"p{plow}":low, f"p{phigh}":phigh})
#     df.to_csv("MGCLS_pertentiles.csv")


def scale_crops(crop_dir, out_dir, meta, method="cs", **kwargs):
    """
    Scale crops according to entire image parameters.

    This function scales the crops located in the 'crop_dir' directory according to the parameters extracted from the entire image.
    It reads metadata from the CSV file specified in the 'meta' parameter.

    Parameters:
    crop_dir (str): The directory containing the crops to be scaled.
    out_dir (str): The directory where scaled crops will be saved.
    meta (str): The path to the CSV file containing metadata (dataframe with metadata).
    method (str): The scaling method to be used. Defaults to "cs" (contrast_stretch).
                  Other supported methods include "mt" (manual thresholding).
    **kwargs: Additional keyword arguments to be passed to the scaling method.

    Returns:
    None
    """
    meta = pd.read_csv(meta) #dataframe with metadata
    crops = sorted(glob.glob(f"{os.path.join(crop_dir, '*crop*')}"))
    cprev=''

    for c in crops:
        ccurrent = c[c.rfind('/')+1:c.find("crop")-1]
        while ccurrent != cprev:
            pvals = meta.where(meta.file_prefix == ccurrent).dropna(how='all')[['p2','p98']].values
        #    zint = ZScaleInterval(**kwargs)
        #    zdat = zint() #original fits data, scaled
        cdat = np.load(c) #what if it's a FITS file?
        if method == 'cs': #contrast_stretch
            cnew = rescale_intensity(cdat, tuple(pvals[0]))
        #elif method == 'zscale':
        elif method == "mt":
            #replace lowest and highest 2% of values
            p2 = pvals[0][0]
            p98 = pvals[0][1]
            cdat[cdat < p2] = p2
            cdat[cdat > p98] = p98
            I_norm = (cdat/p98)**power
            cnew = (I_norm - 0.5)/0.5
        #save scaled crop
        np.save(f"{os.path.join(out_dir,c[c.rfind('/')+1])}",cnew)
        cprev = ccurrent

# def MGCLS_crop_COCO(crop_dir, out_dir, crop_shape, meta, n):
#     fitsf = glob.glob(os.path.join(crop_dir, '*.fits'))
#     for f in fitsf:
#         bn = f[f.rfind('/')-1:f.rfind(".")]
#         wcs, arr = mapdata_from_fits(f)
#         stride = calc_stride(arr.shape[-1], crop_shape[0], n)
#         crops, coords = MGCLS_crop_coords(wcs, arr[0,0,:,:], crop_shape = crop_shape, stridex = stride, stridey = stride)
#         print(f"{len(crops)} crops from {f}")
#         np.save(f"{bn}_large_coords.npy", coords)
#         for i,c in enumerate(crops):
#             np.save(f"{out_dir}/{bn}_crop_{i}.npy",c)
#     scale_crops(out_dir, f"{out_dir}_cs", meta)

#################### catalog functions #############

# def load_MIGHTEE_txt_catalogs():
#     cats = glob.glob("/home/glados/unix-Documents/AstroSignals/data/MIGHTEE/early_science/*.txt")
#     cdf = pd.read_fwf(cats[0], skiprows=8, header=[0,1])
#     cdf.drop(0,inplace=True)
#     xdf = pd.read_fwf(cats[1], skiprows=8, header=[0,1])
#     xdf.drop(0,inplace=True)
#     return cdf, xdf

# def split_cat(coords, cat, index, source = "COSMOS"):
#     bl = coords[index][0][0]
#     blra = bl.ra.value
#     tr = coords[index][1][0]
#     trra = tr.ra.value
#     bldec =bl.dec.value
#     trdec = tr.dec.value
#     dd = cat.where(cat.RA.DEG > min((blra,trra))).where(cat.RA.DEG < max((blra,trra))).dropna(how='all')
#     res = dd.where(dd.DEC.DEG > min((bldec,trdec))).where(dd.DEC.DEG < max((bldec,trdec))).dropna(how='all')
#     res.to_json(f"{source}_cat_crop_{index}.json")

# # def identity(x):
# #     return x

# def cat_hist(df, key, log = False, df2 = None, names=["XMM-LSS","COSMOS"], agg=None):
#     try:
#         subkey = df[key].keys()[0][0]
#     except TypeError:
#         subkey = ""
#     title = f"{key} ({subkey})"

#     fig = go.Figure()
#     func = np.log10 if log else identity
#     if log:
#         title = "log10 " + title
    
#     if agg:
#         title += f" {agg}"
#         vals = df[key][subkey][agg].values
#     elif subkey != "": 
#         vals = df[key].values[:,0]
#     else: 
#         vals = df[key].values
#     fig.add_trace(go.Histogram(x = func(vals).astype(list), name = names[0]))
#     if df2 is not None:
#         if agg:
#             vals2 = df2[key][subkey][agg].values
#         elif subkey != "":
#             vals2 = df2[key].values[:,0]
#         else: 
#             vals2 = df2[key].values
#         fig.add_trace(go.Histogram(x = func(vals2).astype(list), name = names[1]))
#         fig.update_layout(barmode = "overlay")
#         fig.update_traces(opacity=0.75)
#     fig.update_layout(title = title, xaxis_title =  title)
#     return fig

# def plot_cat_sources(catdf, sfac = 1e4, bounds = None, color_key = "S_INT", title=None):
#     logint = np.log10(catdf[color_key].values[:,0])
#     cmin = logint.min()
#     cmax = logint.max()
#     mdict = dict(cmax= cmax, cmin = cmin, color =logint.astype(list), colorbar=dict(title=color_key))
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=catdf.RA.values[:,0].astype(list), y=catdf.DEC.values[:,0].astype(list), hovertext=catdf["#              NAME"].values[:,0].astype(list), marker = mdict, mode = 'markers'))
#     if bounds is not None:
#         xb, yb = bl_tr_to_bounds(*bounds)
#         fig.add_trace(go.Scatter(x=xb,y=yb))
    
#     fig.update_layout(xaxis_title = "RA (deg)", yaxis_title = "DEC (deg)", height=450, title=title)
#     fig.update_yaxes(scaleanchor='x', scaleratio=1)
#     return fig

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

if __name__ == '__main__':
    MGCLS4ML("/home/users/l/lastufka/scratch/MGCLS_data")
    #scale_crops(out_dir, f"{out_dir}_cs", meta)