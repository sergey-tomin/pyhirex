"""
Created on Tue Mar 14 14:01:46 2017

@author: Svitozar Serkez
"""


from copy import deepcopy
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as colors
from scipy import signal
import scipy
import glob
import h5py
import numpy
import sys
import time


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Import numba if available
try:
    import numba as nb
    from numba import jit
    numba_avail = True
except ImportError:
    print("math_op.py: module Numba is not installed. Install it if you want speed up correlation calculations.")
    numba_avail = False

# # Import ocelot if available
# try:
    # import ocelot
    # from ocelot.utils.xfel_utils import *
    # from ocelot.optics.wave import imitate_1d_sase
    # ocelot_avail = True
# except ImportError:
    # print("ocelot is not imported.")
ocelot_avail = False

# Configure some plotting parameters.
rcParams['image.cmap'] = 'viridis'
#rcParams['figure.dpi'] = 100
# rcParams['figure.figsize'] = [8, 6]
# rcParams['font.size'] = 15.0
# Physical constants.
from scipy.constants import e, hbar, c, h
from math import pi

# Some useful conversions (### move to separate library?)
h_eV_s = h/e # = 4.135667516e-15 eV s
hr_eV_s = h_eV_s / 2. / pi
speed_of_light = c # = 299792458.0 m/s



def find_nearest_idx(array, value):
    if value == -numpy.inf:
        value = numpy.amin(array)
    if value == numpy.inf:
        value = numpy.amax(array)
    return (numpy.abs(array-value)).argmin()
    
def find_nearest(array, value):
    return array[find_nearest_idx(array, value)]

def n_moment(x, counts, c, n):
    x = numpy.squeeze(x)
    if x.ndim != 1:
        raise ValueError("scale of x should be 1-dimensional")
    if x.size not in counts.shape:
        raise ValueError("operands could not be broadcast together with shapes %s %s" %(str(x.shape), str(counts.shape)))
    
    if numpy.sum(counts)==0:
        return 0
    else:
        if x.ndim == 1 and counts.ndim == 1:
            return (numpy.sum((x-c)**n*counts) / numpy.sum(counts))**(1./n)
        else:
            
            if x.size in counts.shape:
                dim_ = [i for i, v in enumerate(counts.shape) if v == x.size]
                counts = numpy.moveaxis(counts, dim_, -1)
                return (numpy.sum((x-c)**n*counts, axis=-1) / numpy.sum(counts, axis=-1))**(1./n)
                
        
def std_moment(x, counts):
    mean=n_moment(x, counts, 0, 1)
    return n_moment(x, counts, mean, 2)


def fwhm3(valuelist, height=0.5, peakpos=-1, total=1):
    ### comments:
    ### handle exception if valuelist not proper input (e.g. not a list or array)
    ### rename peakpos to peak_index
    ### None as default for peakpos
    ### Rename total to start_search_from_edges
    ### Type of parameter "total" should be bool (default=True)
    """calculates the full width at half maximum (fwhm) of the array.
    the function will return the fwhm with sub-pixel interpolation.
    It will start at the maximum position and 'walk' left and right until it approaches the half values.
    if total==1, it will start at the edges and 'walk' towards peak until it approaches the half values.
    INPUT:
    - valuelist: e.g. the list containing the temporal shape of a pulse
    OPTIONAL INPUT:
    -peakpos: position of the peak to examine (list index)
    the global maximum will be used if omitted.
    if total = 1 -
    OUTPUT:
    - peakpos(index), interpolated_width(npoints), [index_l, index_r]
    """
    if peakpos == -1:  # no peakpos given -> take maximum
        peak = numpy.max(valuelist)
        peakpos = numpy.min(numpy.nonzero(valuelist == peak))
    peakvalue = valuelist[peakpos]
    phalf = peakvalue * height
    if total == 0:
        # go left and right, starting from peakpos
        ind1 = peakpos
        ind2 = peakpos
        while ind1 > 2 and valuelist[ind1] > phalf:
            ind1 = ind1 - 1
        while ind2 < len(valuelist) - 1 and valuelist[ind2] > phalf:
            ind2 = ind2 + 1
        grad1 = valuelist[ind1 + 1] - valuelist[ind1]
        grad2 = valuelist[ind2] - valuelist[ind2 - 1]
        if grad1 == 0 or grad2 == 0:
            width = None
        else:
            # calculate the linear interpolations
            # print(ind1,ind2)
            p1interp = ind1 + (phalf - valuelist[ind1]) / grad1
            p2interp = ind2 + (phalf - valuelist[ind2]) / grad2
            # calculate the width
            width = p2interp - p1interp
    else:
        # go to center from edges
        ind1 = 1
        ind2 = valuelist.size-2
        # print(peakvalue,phalf)
        # print(ind1,ind2,valuelist[ind1],valuelist[ind2])
        while ind1 < peakpos and valuelist[ind1] < phalf:
            ind1 = ind1 + 1
        while ind2 > peakpos and valuelist[ind2] < phalf:
            ind2 = ind2 - 1
        # print(ind1,ind2)
        # ind1 and 2 are now just above phalf
        grad1 = valuelist[ind1] - valuelist[ind1 - 1]
        grad2 = valuelist[ind2 + 1] - valuelist[ind2]
        if grad1 == 0 or grad2 == 0:
            width = None
        else:
            # calculate the linear interpolations
            p1interp = ind1 + (phalf - valuelist[ind1]) / grad1
            p2interp = ind2 + (phalf - valuelist[ind2]) / grad2
            # calculate the width
            width = p2interp - p1interp
        # print(p1interp, p2interp)
    
    return (p1interp, p2interp)

def mode(ndarray, axis=0):
    ### Complete docstring
    '''
    by Devin Cairns
    '''
    # Check inputs
    ndarray = numpy.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
    if all([ndim == 1,
            int(numpy.__version__.split('.')[0]) >= 1,
            int(numpy.__version__.split('.')[1]) >= 9]):
        modals, counts = numpy.unique(ndarray, return_counts=True)
        index = numpy.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = numpy.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = numpy.roll(numpy.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = numpy.concatenate([numpy.zeros(shape=shape, dtype='bool'),
                                 numpy.diff(sort, axis=axis) == 0,
                                 numpy.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = numpy.cumsum(strides)
    counts[~strides] = numpy.concatenate([[0], numpy.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = numpy.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = numpy.ogrid[slices]
    index.insert(axis, numpy.argmax(counts, axis=axis))
    return sort[index], counts[index]

class ImageArray():
    """
    :class ImageArray: Container class for a stack of images.
    """

    def __init__(self,
                image_stack=None,
                train_id=None,
                ):
        '''
        Constructor of the ImageArray class.

        :param image_stack: Stack of images to process. Default: None (will be set by caller).
        :type image_stack: numpy.array (3D).

        :param train_id: Sequence of train IDs corresponding to images in the stack. Default: None (will be set by caller).
        :type train_id: 1D iterable

        '''

        if image_stack is None:
            self.im = numpy.zeros((1,1,1)) # event, y, x
            self.x = None #x pixel number array
            self.y = None #y pixel number array
        else:
            self.im = image_stack
            self.x = numpy.arange(self.im.shape[2])
            self.y = numpy.arange(self.im.shape[1])
        
        self.pixsize_x = None
        self.pixsize_y = None
        
        self.train_id = train_id
        
        self.hdf5 = None
        self.file_path = None
        self.phen = None
        ### Could expose members in constructor.

    def integr(self):
        return numpy.sum(self.im, axis=(1, 2))


    def remove_zero_trains(self):
        '''
        Remove trains with train id 0
        '''

        idx = numpy.where(self.train_id != 0)[0]
        self.im = self.im[idx,:,:]
        self.train_id = self.train_id[idx]

    def remove_zero_images(self):
        '''
        Removes images with 0 total count.
        '''
        integr = self.integr()
        idx = numpy.where(integr != 0)[0]
        self.im = self.im[idx,:,:]
        self.train_id = self.train_id[idx]

    def add_spec_scale(self, E_center=9350, dE=0.1):
        '''
        Allows to add photon energy scale.

        :param E_center: Photon energy at the central pixel of the dispersive direction [eV].

        :param dE: Energy resolution per pixel [eV]
        '''

        linsp = numpy.arange(self.im.shape[1])
        linsp = linsp - numpy.mean(linsp)
        self.phen = linsp * dE + E_center

    def yield_spectrum_old(self, slice_min=-numpy.inf, slice_max=numpy.inf, E_center=9350, dE=0.1, spec_direction='y'):
        '''
        Calculates line-out between slice_min and slice_max
        to obtain 2d data matrix of single-shot spectra
        returns SpectrumArray() object.

        :param slice_min: Minimum spectral energy [eV].

        :param slice_max: Maximum spectral energy [eV].

        :param E_center: Central energy [eV]. Default: E_center=9350.

        :param dE: Energy increment [eV]. Default: dE=0.1.


        :param spec_direction: Dispersive direction in the 2D image. Default: disp_dir='y'.
        :type spec_direction: str

        :return: Array of single-shot spectra.
        :rtype: SpectrumArray

        '''

        spar = SpectrumArray()
        linsp = self.y - numpy.mean(self.y)
        spar.omega = (linsp * dE + E_center) / hr_eV_s
        idx = numpy.logical_and(self.x >= slice_min, self.x <= slice_max)
        spar.train_id = self.train_id
        spar.path = self.file_path
        if self.im.squeeze().ndim == 3:

            if sum(idx) == 1:
                spar.spec = self.im[:,:,idx].T
            else:
                spar.spec = numpy.mean(self.im[:,:,idx],2).T
            return spar

        else:
            spar.spec = self.im[:,:].squeeze().T
            return spar


    def yield_spectrum(self, slc=slice(None,None,1), E_center=9350, dE=0.1, disp_dir='y'):
        '''
        Calculates line-out over a given slice
        to obtain 2d data matrix of single-shot spectra.
        Returns SpectrumArray() object

        :param slc: Slice over which to extract the spectra.
        :type slc: slice

        :param E_center: Central energy for spectral axis [eV]. Default: E_center=9350

        :param dE: Energy increment for spectral axis [eV]. Default: dE=0.1

        :param disp_dir: Dispersive direction in the 2D image. Default: disp_dir='y'.
        :type disp_dir: str

        :return: Array of single-shot spectra.
        :rtype: SpectrumArray

        '''
        spar = SpectrumArray()
        if disp_dir == 'y':
            linsp = self.y - numpy.mean(self.y)
        elif disp_dir == 'x':
            linsp = self.x - numpy.mean(self.x)
        else:
            raise ValueError('disp_dir should be "x" or "y"')

        spar.omega = (linsp * dE + E_center) / hr_eV_s
        # idx = numpy.logical_and(self.x >= slice_min, self.x <= slice_max)
        spar.train_id = self.train_id
        spar.path = self.file_path
        if self.im.squeeze().ndim == 3:

            if disp_dir == 'y':
                spar.spec = numpy.mean(self.im[:,:,slc],2).T
            elif disp_dir == 'x':
                spar.spec = numpy.mean(self.im[:,slc,:],1).T
            else:
                raise ValueError('disp_dir should be "x" or "y"')
            # if sum(idx) == 1:
                # spar.spec = self.im[:,:,idx].T
            # else:
                # spar.spec = numpy.mean(self.im[:,:,idx],2).T
            # return spar

        else:
            spar.spec = self.im[:,:].squeeze().T

        return spar

    def remove_background_camx(self, sl=slice(1, 10), averaged=1):
        '''
        Removes background individually for each x position
        (useful when the image has characteristic noise equal at all 'y'-dispersive positions)

        :param sl: Slice object with range to calculate background. Default: sl=slice(1,10).
        :type sl: slice

        :param averaged: Flag to control wheather to treat each event individually. Default: averaged=1 (treat each event individually). Set to averaged=0 to turn this behaviour off.
        :type averaged: int

        '''

        if averaged:
            bkg = numpy.mean(self.im[:, sl, :], axis=(0, 1))[numpy.newaxis,numpy.newaxis,:].astype(self.im.dtype)
            #bkg = numpy.mean(self.im[:, :, sl], axis=(0, 2))[numpy.newaxis,:,numpy.newaxis].astype(self.im.dtype)
        else:
            bkg = numpy.mean(self.im[:, sl, :], axis=1)[:,numpy.newaxis,:].astype(self.im.dtype)
            #bkg = numpy.mean(self.im[:, :, sl], axis=2)[:, :, numpy.newaxis].astype(self.im.dtype)
        import ipdb; ipdb.set_trace()
        self.im = (self.im - bkg)

    def remove_background_camy(self, sl=slice(1, 10), averaged=1):
        '''
        Removes background individually for each y position
        (useful when the image has characteristic noise equal at all 'x'-dispersive positions)

        :param sl: Slice object with range to calculate background. Default: sl=slice(1,10).
        :type sl: slice

        :param averaged: Flag to control wheather to treat each event individually. Default: averaged=1 (treat each event individually). Set to averaged=0 to turn this behaviour off.
        :type averaged: int
        '''

        if averaged:
            bkg = numpy.mean(self.im[:, :, sl], axis=(0, 2))[numpy.newaxis,:,numpy.newaxis].astype(self.im.dtype)
        else:
            bkg = numpy.mean(self.im[:, :, sl], axis=1)[numpy.newaxis,numpy.newaxis,:].astype(self.im.dtype)
        self.im = (self.im - bkg)

    def remove_background_spec(self, sl=slice(1, 10), averaged=1):
        '''
        Removes background individually for each x position
        (useful when the image has characteristic noise equal at all 'y'-dispersive positions)

        :param sl: Slice object with range to calculate background. Default: sl=slice(1,10).
        :type sl: slice

        :param averaged: Flag to control wheather to treat each event individually. Default: averaged=1 (treat each event individually). Set to averaged=0 to turn this behaviour off.
        :type averaged: int

        '''
        if averaged:
            bkg = numpy.mean(self.im[:, sl, :]).astype(self.im.dtype)
            self.im = (self.im - bkg)
        else:
            bkg = numpy.mean(self.im[:, sl, :], axis=(1, 2))[:,numpy.newaxis,numpy.newaxis].astype(self.im.dtype)
            self.im = (self.im - bkg)

    def remove_background_imar(self, imar, turn_integer=0):
        '''
        Removes average background level from each image.

        :param sl: Slice object with range to calculate background. Default: sl=slice(1,10).
        :type sl: slice

        :param turn_integer: Flag to control wheather to convert array elements to integers. Default: turn_integer=0 (no conversion).
        :type turn_integer: int

        '''

        if self.im.ndim == imar.im.ndim == 3:
            im_av = numpy.mean(imar.im, axis=0)
        else:
            im_av = imar.im

        if turn_integer:
            im_av = int(im_av)
            self.im -= im_av
        else:
            self.im = self.im - im_av

    def subtract_event(self, idx):
        """ Subtract a given event from all other events.

        :param idx: The subtracted event's index.

        """
        self.im = self.im - self.im[idx,:,:][numpy.newaxis,:,:]
        #self.im = numpy.delete(self.im, idx, axis=0)
        #self.train_id = numpy.delete(self.im, idx, axis=0)

    def find_darkest_event(self):
        """ Find the event with the lowest total intensity.

        :return: Index of the darkest event.
        :rtype: int

        """
        integr = self.integr()
        idx = numpy.where(integr == numpy.amin(integr))[0][0]
        return idx

    def subtract_darkest_event(self):
        """ Subtract the darkest event from all other events."""
        self.subtract_event(self.find_darkest_event())


    def remove_low_intensity_events(self, thresh=0.5, relative=1):
        '''
        Delete events with sum per image below a threshold.

        :param thresh: The total intensity threshold (if relative==0) or scaling factor to determine threshold via threshold = thresh * average_value from all events. Default: thresh=0.5

        :param relative: Flag to control the meaning of thresh (absolute threshold or relative to average value).j

        '''
        integr = numpy.sum(self.im, axis=(1,2))
        if relative:
            idx = numpy.where(integr > numpy.mean(integr) * thresh)[0]
        else:
            idx = numpy.where(integr > thresh)[0]
        self.im = self.im[idx,:,:]
        if self.train_id is not None:
            self.train_id = self.train_id[idx]

    def fix_negative_values(self):
        """ Set negative exposures to 0.0 """
        self.im[self.im < 0] = 0

    def plot_event(self, event_n, fignum=None):
        '''
        Plot single camera image.

        :param event_n: Index of event to plot.
        :type event_n: int

        :param fignum: Number of figure to pass to matplotlib.pyplot.figure().
        :type fignum: int

        '''

        if self.pixsize_x != None:
            x = self.x * self.pixsize_x * 1000
            xlabel_txt = 'x [mm]'
        else:
            x = self.x
            xlabel_txt = 'x direction'
            
        if self.pixsize_y != None:
            y = self.y * self.pixsize_y * 1000
            ylabel_txt = 'y [mm]'
        else:
            y = self.y
            ylabel_txt = 'x direction'
        
        plt.figure(fignum)
        plt.clf()
        if self.im[event_n,:,:].squeeze().ndim == 2:
            # print(self.im[1,:,:].T.shape)
            # print(self.x.shape)
            # print(self.y.shape)
            plt.pcolormesh(x, y, self.im[event_n,:,:])
            plt.axis('tight')
            plt.ylabel(ylabel_txt)
            plt.xlabel(xlabel_txt)
            plt.colorbar()
        else:
            plt.plot(self.im[event_n,:,:].T)

        plt.show()

    #def cut(self, xsl=slice(None,None,1), ysl=slice(None,None,1), event=slice(None,None,1)):
        #imar = ImageArray()
        #imar.x = self.x[xsl]
        #imar.y = self.y[ysl]
        #imar.train_id = self.train_id[event]
        #imar.im = self.im[event, ysl, xsl]
        #return imar

    def projections(self):
        '''
        Calculate 3 projections of the image array and return in ImageProjections() object (plottable)
        useful to analyze the evolution of the spot position and intensity.

        :return: Projections of the image stack in x,y, and event.
        :rtype: ImageProjections
        '''

        proj = ImageProjections()
        proj.proj_cam = numpy.sum(self.im, axis=0)
        proj.proj_y = numpy.sum(self.im, axis=2) # spec_av -> proj_y
        proj.proj_x = numpy.sum(self.im, axis=1) # pos_av -> proj_x
        proj.integr = numpy.mean(self.im, axis=(1, 2))
        proj.train_id = self.train_id
        proj.path = self.file_path
        proj.hdf5 = self.hdf5
        proj.x = self.x
        proj.y = self.y
        
        proj.pixsize_x = self.pixsize_x
        proj.pixsize_y = self.pixsize_y

        proj_x_fix = proj.proj_x# - numpy.amin(proj.proj_x)
        proj_y_fix = proj.proj_y# - numpy.amin(proj.proj_y)
        xint = numpy.sum(proj_x_fix, 1)
        yint = numpy.sum(proj_y_fix, 1)

        # Calculate center of mass"
        # Sum up x-coordinates x projected values and divide by sum of projected values. This must be secured against division by 0, using where xint!=0.
        proj.x_centmass_pos = numpy.divide(numpy.dot(proj.x , proj_x_fix.T), xint, out=xint.astype(numpy.float), where=xint!=0).astype(numpy.int)
        proj.y_centmass_pos = numpy.divide(numpy.dot(proj.y , proj_y_fix.T), yint, out=yint.astype(numpy.float), where=yint!=0).astype(numpy.int)

        max_pos = [numpy.unravel_index(numpy.argmax(im, axis=None), im.shape) for im in self.im]
        proj.x_max_pos = numpy.array([i[0] for i in max_pos])
        proj.y_max_pos = numpy.array([i[1] for i in max_pos])

        return proj


class ImageProjections():
    '''
    :class ImageProjections: Container class to store 3 cartesian projections of the Image Array object.
    '''
    def __init__(self):
        self.proj_cam = None
        self.proj_y = None
        self.proj_x = None
        self.integr = None
        self.train_id = None
        self.path = None
        self.hdf5 = None
        self.x = None
        self.y = None
        self.y_max_pos = None
        self.y_max_pos = None
        self.x_centmass_pos = None
        self.y_centmass_pos = None
        pass

    def plot(self, plot_pos=1, plot_text=1, fignum=None, figsize=(15,10)):
        """ Plot the projections in matplotlib subplots.

        :param plot_pos: Flag to control whether to plot the position. Default: plot_pos=1 (True).

        :param fignum: Figure number to pass to matplotlib.pyplot.figure().

        :param figsize: Size of figure in cm. Default: figsize=(15,10)

        """
        events = self.proj_y.shape[0]
        xscale = self.x
        yscale = self.y
        
        if self.pixsize_x != None:
            mult_x = self.pixsize_x * 1000
            xlabel_txt = 'x [mm]'
        else:
            mult_x = 1
            xlabel_txt = 'x direction'
            
        if self.pixsize_y != None:
            mult_y = self.pixsize_y * 1000
            ylabel_txt = 'y [mm]'
        else:
            mult_y = 1
            ylabel_txt = 'x direction'
            
        xscale = xscale * mult_x
        yscale = yscale * mult_y

        
        #yscale = numpy.arange(self.proj_cam.shape[0])
        #xscale = numpy.arange(self.proj_cam.shape[1])
        #if self.train_id.size == events and numpy.nansum(self.train_id)>0:
        if self.train_id is not None:
            eventscale = self.train_id
        else:
            eventscale = numpy.arange(events)

        plt.figure(fignum, figsize=figsize)
        plt.clf()
        plt.subplots_adjust(wspace=.2, hspace=.2)

        ax_xy = plt.subplot(2,2,1)
        if self.proj_cam.squeeze().ndim == 2:
            plt.pcolormesh(xscale, yscale, self.proj_cam)
            plt.xlabel(xlabel_txt)
            plt.ylabel(ylabel_txt)
            plt.axis('tight')
            #ax_xy = plt.gca()
            if plot_text:
                plt.text(0.98, 0.98, '[{:.1f} : {:.1f}]'.format(numpy.amin(self.proj_cam), numpy.amax(self.proj_cam)),
                        horizontalalignment='right', verticalalignment='top',
                        transform=ax_xy.transAxes, color='white')
                plt.text(0.98, 0.02, '[{:.1f} : {:.1f}]'.format(numpy.amin(self.proj_cam), numpy.amax(self.proj_cam)),
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax_xy.transAxes, color='black')
                        
        ax_ty = plt.subplot(2,2,2, sharey=ax_xy)
        if self.proj_y.squeeze().ndim == 2:
            plt.pcolormesh(eventscale, yscale, self.proj_y.T)
            if plot_pos:
                if hasattr(self,'y_centmass_pos'):
                        plt.plot(eventscale, self.y_centmass_pos * mult_y, color='red', linewidth=0.5)
                if hasattr(self,'y_max_pos'):
                    plt.plot(eventscale, self.y_max_pos * mult_y, color='green', linewidth=0.5)
            plt.xlabel('event')
            plt.ylabel(ylabel_txt)
            plt.axis('tight')
            if plot_text:
                plt.text(0.98, 0.98, '[{:.1f} : {:.1f}]'.format(numpy.amin(self.proj_y), numpy.amax(self.proj_y)),
                        horizontalalignment='right', verticalalignment='top',
                        transform=ax_ty.transAxes, color='white')
                plt.text(0.98, 0.02, '[{:.1f} : {:.1f}]'.format(numpy.amin(self.proj_y), numpy.amax(self.proj_y)),
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax_ty.transAxes, color='black')

        ax_xt = plt.subplot(2,2,3, sharex=ax_xy)
        if self.proj_x.squeeze().ndim == 2:
            plt.pcolormesh(xscale, eventscale, self.proj_x)
            if plot_pos:
                if hasattr(self,'x_centmass_pos'):
                    plt.plot(self.x_centmass_pos * mult_x, eventscale, color='red', linewidth=0.5)
                if hasattr(self,'x_max_pos'):
                    plt.plot(self.x_max_pos * mult_x, eventscale, color='green', linewidth=0.5)
            plt.ylabel('event')
            plt.xlabel(xlabel_txt)
            plt.axis('tight')
            if plot_text:
                plt.text(0.98, 0.98, '[{:.1f} : {:.1f}]'.format(numpy.amin(self.proj_x), numpy.amax(self.proj_x)),
                        horizontalalignment='right', verticalalignment='top',
                        transform=ax_xt.transAxes, color='white')
                plt.text(0.98, 0.02, '[{:.1f} : {:.1f}]'.format(numpy.amin(self.proj_x), numpy.amax(self.proj_x)),
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax_xt.transAxes, color='black')

        ax_ti = plt.subplot(2,2,4, sharex=ax_ty)
        plt.plot(eventscale, self.integr)
        plt.axis('tight')
        #plt.ylim(bottom=0)
        plt.xlabel('event')
        plt.ylabel('flux [a.u]')

        try:
            ax_xy.set_xlim([numpy.amin(xscale), numpy.amax(xscale)])
            ax_xy.set_ylim([numpy.amin(yscale), numpy.amax(yscale)])
        except:
            pass

        plt.show()


def read_im_arr_hdf5(
        file_path,
        dtype=numpy.int32,
        source='bragg',
        xsl=slice(None,None,1),
        ysl=slice(None,None,1),
        event=slice(None,None,1),
        rotate=0,
        
        ):
    """ Function to read (a stack of) images and relevant metadata from hdf5.

    :param file_path: The path to the file to be read.
    :type file_path: str

    :param dtype: The type in which the images to read is stored. Default: dtype=numpy.int32.

    :param source: The data source (camera or detector) from which to read the images. Default: camera=bragg. Possible values: 'bragg' || gotthard' || path to hdf5 dataset, e.g. "INSTRUMENT/SA1_XTD_HIREX/CAM/BRAGG:daqOutput/data/image/pixels"
    :type source: str

    :param xsl: Slice of indices on third axis (x) to read. Default: xsl=slice(None, None, 1) [-> read all x values]
    :type xsl: slice

    :param ysl: Slice of indices on second axis (y) to read. Default: ysl=slice(None, None, 1) [-> read all y values]
    :type ysl: slice

    :param event: Slice of indices on first axis (time/events) to read. Default: event=slice(None, None, 1) [-> read all events]
    :type event: slice

    """
    # Construct the image array.
    imar = ImageArray()

    with h5py.File(file_path, 'r') as f:
        if source == 'bragg':
            pixels = f['INSTRUMENT/SA1_XTD9_HIREX/CAM/BRAGG:daqOutput/data/image/pixels']
        elif source == 'gotthard':
            pixels = f['INSTRUMENT/SA1_XTD9_HIREX/DAQ/GOTTHARD:daqOutput/data/adc']
        else:
            pixels = f[source]

        # Take out the numpy array and close file safely.
        pixels = pixels.value

        # Attempt to read the trainIDs.
        try:
            train_id = f['INSTRUMENT/SA1_XTD9_HIREX/CAM/BRAGG:daqOutput/data/trainId'].value
            imar.train_id = train_id[event]
        except:
            # raise
            pass

    # Convert to requested type and shape.
    imar.im = (pixels[event,ysl,xsl]).astype(dtype)
    
    shape_orig = imar.im.shape
    if rotate == 90:
        imar.im = numpy.reshape(imar.im, (shape_orig[0], shape_orig[2], shape_orig[1])) #temporary fix for improperly recorded rotated images
    
    if source == 'gotthard':
        imar.im = numpy.rollaxis(imar.im,2,1)
        imar.x = numpy.arange(imar.im.shape[1])
        imar.y = numpy.arange(imar.im.shape[2])
    else:
        imar.x = numpy.arange(imar.im.shape[2])
        imar.y = numpy.arange(imar.im.shape[1])

    imar.file_path = file_path

    print('  done')

    return imar

def read_im_arr_raw(
        file_path_mask,
        dtype=numpy.int32,
        resolution=(1080,1920),
        xsl=slice(None,None,1),
        ysl=slice(None,None,1),
        event=slice(None,None,1),
        ):
    """ """
    """ undocumented """

    print('reading')
    filenames = glob.glob(file_path_mask)
    d = []
    for filename in filenames:
        #print(filename)
        l0 = numpy.fromfile(filename, dtype=numpy.uint16)
        l = l0.reshape(resolution)
        d.append(l)

    imar = ImageArray()
    im = numpy.rollaxis(numpy.array(d),0)
    if dtype is None:
        dtype=numpy.uint16
    imar.x = numpy.arange(im.shape[2])[xsl]
    imar.y = numpy.arange(im.shape[1])[ysl]
    imar.train_id = numpy.arange(im.shape[0])[event]
    imar.im = im[event,ysl,xsl].astype(dtype)
    imar.file_path = file_path_mask

    print('   done, processed {:d} files'.format(len(d)))
    return imar

def read_im_arr_hdf5_LCLS(file_path, xsl=slice(None,None,1), ysl=slice(None,None,1), event=slice(None,None,1), dtype=numpy.int32, cryst='C'):
    """ """
    """ undocumented """
    print('reading')
    f = h5py.File(file_path, 'r')
    if cryst == 'C':
        pixels = f['/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/XcsEndstation.1:Opal1000.2/image']
    elif cryst == 'Si':
        pixels = f['/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/XcsEndstation.1:Opal1000.1/image']
    else:
        pixels = f[cryst]
        # raise ValueError('cryst should be "Si" or "C"')
#     train_id = f['INSTRUMENT/SA1_XTD9_HIREX/CAM/BRAGG:daqOutput/data/trainId']
    imar = ImageArray()
    im = pixels[event]
    if im.ndim==2:
        im = im[numpy.newaxis, :,:]
#     imar.x = numpy.arange(im.shape[1])[xsl]
#     imar.y = numpy.arange(im.shape[2])[ysl]
    im = im[:,ysl,xsl]
#     imar.train_id = train_id[event]
    imar.im = im.astype(dtype)
    imar.x = numpy.arange(imar.im.shape[2])
    imar.y = numpy.arange(imar.im.shape[1])
#     imar.im = numpy.rollaxis(imar.im,2,1)
    imar.file_path = file_path
    print('  done')
    return imar

def read_spec_arr_hdf5_FLASH(file_path, wav0=None, dwav = None, remove_empty=False):
    """ """
    """ undocumented """
    f = h5py.File(file_path, 'r')
    pixels = f['/Photon Diagnostic/Wavelength/PG2 spectrometer/photon wavelength']
    if dwav is None:
        dwav = f['/Photon Diagnostic/Wavelength/PG2 spectrometer/photon wavelength increment']
    if wav0 is None:
        wav0 = f['/Photon Diagnostic/Wavelength/PG2 spectrometer/photon wavelength start value']
    wav = (numpy.nanmean(wav0) + numpy.arange(pixels.shape[1]) * numpy.nanmean(dwav)) * 1e-9

    spar = SpectrumArray()
    spar.spec = numpy.array(pixels).T
    spar.phen = h_eV_s * speed_of_light / wav
    spar.train_id = None#numpy.arange(pixels.shape[0])
    spar.path = file_path
    spar.spec[numpy.isnan(spar.spec)] = 0
    if remove_empty:
        idx_nonzero = numpy.where(spar.integral() != 0)[0]
        spar.spec = spar.spec[:,idx_nonzero]
        spar.train_id = spar.train_id[idx_nonzero]
    return spar
    # x = f['/Photon Diagnostic/GMD/Beam position/position BDA x'][idx_nonzero, 0]
    # y = f['/Photon Diagnostic/GMD/Beam position/position BDA y'][idx_nonzero, 0]
    # e_el = f['/Electron Diagnostic/Electron energy/pulse resolved energy'][idx_nonzero, 0]

def read_spec_arr_hdf5_FLASH_online(file_path, wav0=None, dwav = None, remove_empty=False):
    """ """
    """ Read hdf5 files yielded by PG2 spectrometer in FLASH """
    f = h5py.File(file_path, 'r')
    pixels = f['/Photon Diagnostic/Wavelength/PG2 spectrometer/photon wavelength']
    if dwav is None:
        dwav = pixels.attrs['inc']
    if wav0 is None:
        wav0 = pixels.attrs['start']
    wav = (numpy.nanmean(wav0) + numpy.arange(pixels.shape[1]) * numpy.nanmean(dwav)) * 1e-9

    spar = SpectrumArray()
    spar.spec = numpy.array(pixels).T
    spar.phen = h_eV_s * speed_of_light / wav
    spar.train_id = None#numpy.arange(pixels.shape[0])
    spar.path = file_path
    spar.spec[numpy.isnan(spar.spec)] = 0
    if remove_empty:
        spar.remove_zero_int()
    return spar


def read_spec_arr_hdf5_SA3(file_path, phen_channel='/INSTRUMENT/SA3_XTD10_SPECT/MDL/FEL_BEAM_SPECTROMETER:output/data/photonEnergy', spec_channel='/INSTRUMENT/SA3_XTD10_SPECT/MDL/FEL_BEAM_SPECTROMETER:output/data/intensityDistribution', train_id_channel='/INSTRUMENT/SA3_XTD10_SPECT/MDL/FEL_BEAM_SPECTROMETER:output/data/trainId'):
    print('reading')
    f = h5py.File(file_path, 'r')
    spar = SpectrumArray()
    phen = f[phen_channel]
    spec = f[spec_channel]
    train_id = f[train_id_channel]
    #    spar.phen[0,:]
    idx = phen[0,:] != 1
    spar.phen = phen[0, idx]
    spar.spec = spec[:, idx].T
    #    spar.phen = phen * dE + E_center
    spar.train_id = train_id[:]
    spar.path = file_path
    print('  done')
    return spar

def simulate_spar_random(n_events=5,
        spec_center=9350,
        spec_width=20,
        spec_res=0.1,
        pulse_length=1,
        flattop=0,
        spec_extend=3
        ):
    """
    Simulate an array ot FEL spectra (interface function to ocelot.optics.wave.imitate_1d_sase).

    :param spec_center: Central energy [eV]. Default: spec_center=9350.
    :type spec_center: numeric

    :param spec_width: Spetral width (full width at half maximum) [eV]. Default: spec_width=20.
    :type spec_width: numeric

    :param spec_res: Spectral resolution [eV]. Default: spec_res=0.1.
    :type spec_res: numeric

    :param pulse_length: Pulse duration [fs]. Default: pulse_length=1.
    :type pulse_length: numeric

    :param flattop: Flag to control whether pulse shape is flat-top (rectangular). Default: flattop=0 (-> not rectangular).
    :type flattop: int

    :param spec_extend: ???

    """
    t, td, phen,fd = imitate_1d_sase(n_events=n_events, spec_center=spec_center, spec_width=spec_width, spec_res=spec_res, pulse_length=pulse_length, flattop=flattop, spec_extend=spec_extend)
    #    del(_)
    spar = SpectrumArray()
    spar.spec = abs(fd)**2
    #    power = abs(td)**2
    spar.omega = phen / hr_eV_s
    spar.pow = abs(td)**2
    spar.t = t

    return spar


class SpectrumArray():
    """ :class SpectrumArray: Container class for FEL spectra (intensity vs. energy).
    """

    def __init__(self,
            n_spec=1,
            n_event=1,
            ):
        """
        SpectrumArray constructor.

        :param n_spec: Number of spectra to construct. Default: n_spec=1.
        :type n_spec: int

        :param n_event: Number of events for which to construct spectra. Default: n_event=1.
        :type n_event: int

        """

        self.spec = numpy.zeros((n_spec, n_event))
        self.omega = numpy.zeros((n_spec))
        self.path = None

    @property
    def phen(self):
        return self.omega * hr_eV_s

    @phen.setter
    def phen(self, energy):
        self.omega = energy / hr_eV_s

    @property
    def events(self):
        return self.spec.shape[1]

    @property
    def fluct_sigma(self):
        with numpy.errstate(divide='ignore'):
            return numpy.std(self.spec, 1) / numpy.mean(self.spec, 1)

    def integral(self):
        return numpy.sum(self.spec, 0)

    def remove_zero_int(self):
        integr = self.integral()
        idx = numpy.where(integr != 0)[0]
        self.spec = self.spec[:,idx]

    def phen_av(self):
        return numpy.sum(self.phen[:,numpy.newaxis] * self.spec, 0) / numpy.sum(self.spec, 0)

    def ignore_empty_train_ids(self):
        idx = numpy.where(self.train_id == 0)
        del self.spec[:,idx]

    def conv_gauss(self, dE=0.1):
        if dE > 0:
            instr_f = 1 / numpy.sqrt(2 * numpy.pi * dE**2) * numpy.exp( -(self.phen - numpy.mean(self.phen))**2 / (2 * dE**2) )
            instr_f /= numpy.sum(instr_f)
            self.spec = signal.convolve(self.spec, instr_f[:,numpy.newaxis], mode='same')

            
    def plot_integral(self, fignum=None, plot_av=0):
        fig = plt.figure(fignum)
        plt.clf()
        events = numpy.arange(self.spec.shape[1])
        
        integral = self.integral()
        # if log_scale:
            # spec_min = numpy.amin(self.spec[self.spec>0])
            # self.spec[self.spec<0] = spec_min
            # spec_max = numpy.amax(self.spec)
            # plt.pcolormesh(events, self.phen, self.spec, 
                       # norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                       # vmin=spec_min, vmax=spec_max))
        plt.figure(fignum)
        plt.plot(events, integral)

        if plot_av:
            plt.plot(events, numpy.ones_like(integral) * numpy.mean(integral))

        plt.ylabel('Integral [arb.units]')
        plt.xlabel('event')
        plt.axis('tight')
        plt.show()
            
    def plot_evo(self, fignum=None, plot_av=0, log_scale=False, cmap='viridis'):
        plt.figure(fignum)
        plt.clf()
        events = numpy.arange(self.spec.shape[1])
        if log_scale:
            spec_min = numpy.amin(self.spec[self.spec>0])
            self.spec[self.spec<0] = spec_min
            spec_max = numpy.amax(self.spec)
            plt.pcolormesh(events, self.phen, self.spec, 
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                       vmin=spec_min, vmax=spec_max))
        else:
            plt.pcolormesh(events, self.phen, self.spec, cmap=cmap)

        if plot_av:
            plt.plot(events, self.phen_av(), color = "red")

        plt.ylabel('$E_{photon}$ [eV]')
        plt.xlabel('event')
        plt.axis('tight')
        plt.show()

    def plot_lines(self, fignum=None, figsize=None, event=[0], log_scale=False, xlim=(None, None)):
        plt.figure(fignum, figsize=figsize)
        plt.clf()
        
        event = numpy.array(event)
        
        if log_scale:
            spec_min = numpy.amin(self.spec[self.spec>0])
            self.spec[self.spec<0] = spec_min
        
        spec_min = numpy.amin(self.spec, axis=1)
        spec_max = numpy.amax(self.spec, axis=1)
        plt.fill_between(self.phen, spec_min, spec_max, facecolor=[0.9,0.9,0.9], edgecolor='w')
        #plt.plot(phen,spec,color=[0.9,0.9,0.9])
        if len(event) == 1:
            plt.plot(self.phen, self.spec[:,event], color=[0.5,0.5,0.5])
        else:
            plt.plot(self.phen, self.spec[:,event])
        plt.plot(self.phen, numpy.mean(self.spec, axis=1), 'k', linewidth=3)
        plt.ylabel('Spec.density [arb.units]')
        plt.xlabel('E [eV]')
        plt.axis('tight')
        
        if log_scale:
            plt.yscale('log', nonposy='clip')
        else:
            print('ylim')
            print(numpy.nanmin(spec_min))
            plt.ylim([numpy.nanmin(spec_min), numpy.nanmax(spec_max)])
        
        if xlim != (None, None):
            plt.xlim([xlim[0], xlim[1]])
        else:
            print('xlim')
            plt.xlim([numpy.nanmin(self.phen), numpy.nanmax(self.phen)])
        
        plt.show()

    def plot_fluct(self, fignum=None):
        plt.figure(fignum)
        plt.clf()
        spec_mean = numpy.mean(self.spec, axis=1)
        with numpy.errstate(divide='ignore'):
            scatter_size = numpy.abs(spec_mean / numpy.amax(spec_mean))**2 * 10 + 1e-10
        #scatter_size = numpy.ones_like(scatter_size) * 1e-20
        plt.scatter(self.phen, self.fluct_sigma**2, scatter_size)
        plt.ylabel('Spec.density fluctuations')
        plt.xlabel('E [eV]')
        plt.axis('tight')
        plt.ylim([0, 2])
        plt.show()


    def add_jitter_ev(self, jitter_ev):
        jitter_points = jitter_ev / abs(self.phen[1] - self.phen[0])
        if jitter_points > 0:
            for i in range(self.events):
                ri = (numpy.random.standard_normal(1) * jitter_points).astype(int)
                self.spec[:,i] = numpy.roll(self.spec[:,i],ri)

    def add_jitter_en(self, jitter_en_percent, positive=False):
        if jitter_en_percent > 0:
            for i in range(self.events):
                rand = (numpy.random.standard_normal(1) / 100 * jitter_en_percent)
                if positive:
                    ri = 1 + abs(rand)
                else:
                    ri = 1 + rand #not nice
                self.spec[:,i] *= abs(ri)

    def cut_spec(self, E_low=-numpy.inf, E_high=+numpy.inf, skip=0):
        idx = numpy.logical_and(self.phen > E_low, self.phen < E_high)
        if not idx.any() and E_low > self.phen.min() and E_high < self.phen.max():
            # if E_low == E_high:
            idx = find_nearest_idx(self.phen, numpy.mean([E_low, E_high]))
            self.spec = self.spec[idx, numpy.newaxis, :]
            self.omega = self.omega[idx]
        else:
            self.spec = self.spec[idx,:]
            self.omega = self.omega[idx]
            self.purge_spec(skip)

    def purge_spec(self, skip=0):
        self.spec = self.spec[::skip+1,:]
        self.omega = self.omega[::skip+1]


    def subtract_offset(self, offset=None):
        if offset is None:
            offset = numpy.nanmin(numpy.mean(self.spec, axis=1))
        self.spec = self.spec - offset

    def subtract_offset_beyond(self, Emin=-numpy.inf, Emax=numpy.inf):
        idx = numpy.where(numpy.logical_or(self.phen < Emin, self.phen > Emax))[0]
        offset = numpy.nanmean(self.spec[idx,:])
        print('offset=', offset)
        self.subtract_offset(offset)
    def fix_negative_values(self):
        self.spec[self.spec < 0] = 0              

    def correlate_center(self, dE=None, t_resolution=0.1e-15, norm=0):
        # print('correlating')
        n_omega, n_events = self.spec.shape
        omega = self.omega
        omega_step = (numpy.amax(omega) - numpy.amin(omega)) / n_omega
        if omega_step == 0:
            omega_step = 1
        t_window = 2 * numpy.pi / omega_step / 2
        # print('   estimated reconstruction window size = {:.2f} fs'.format(t_window * 1e15))

        if dE is None:
            if t_resolution == 0:
                cor_range = numpy.inf
            else:
                cor_range = int(t_window / t_resolution)
        else:
            cor_range = int(dE / hr_eV_s / omega_step)

        if cor_range > n_omega:
            cor_range = n_omega
        # print('   number of correlation points = {:d}'.format(cor_range))

        # estimate reconstruction spatial resolutionpr
        res_domega = omega_step * cor_range
        t_resolution_act = 2 * numpy.pi / res_domega
        # print('   estimated reconstruction spatial resolution = {:.3f} fs'.format(t_resolution_act * 1e15))
        #zeros = numpy.zeros((cor_range, n_events))
        #spec_int = numpy.r_[zeros, self.spec, zeros].astype(numpy.float32)
        # print('   correlating...')
        t0 = time.time()
        corr = SpectrumCorrelationsCenter()
        corr.corr = correlation2d_center(cor_range, self.spec, norm=norm, use_numba=numba_avail)
        #corr.corr = corr_i(n_omega, cor_range, self.spec, norm=norm)

        corr.domega = numpy.linspace(0, numpy.abs(omega[1] - omega[0])*cor_range, cor_range)
        corr.omega = omega
        corr.omega_orig = omega
        corr.spec = self.spec
        t1 = time.time()
        # print('   done in {:.2f} sec'.format(t1 - t0))
        corr.corr[numpy.isnan(corr.corr)] = 0
        corr.corr[numpy.isinf(corr.corr)] = 0
        return corr

    def correlate_full(self, n_skip=1, norm=0):
        print('correlating')
        t0 = time.time()
        corr = SpectrumCorrelationsFull()
        corr.corr = correlation2d(self.spec, norm=norm, n_skip=n_skip, use_numba=numba_avail)
        #corr.corr = corr_f(self.spec, n_skip=n_skip, norm=norm)
        corr.omega = self.omega[::n_skip]
        t1 = time.time()
        print('   done in {:.2f} sec'.format(t1 - t0))

        corr.corr[numpy.isnan(corr.corr)] = 0
        corr.corr[numpy.isinf(corr.corr)] = 0
        corr.norm = norm

        return corr
    
    def calc_histogram(self, E=[-numpy.inf, +numpy.inf], W_lim=[-numpy.inf, +numpy.inf], normed=1, bins=50):
        if numpy.size(E) == 2:
            E_low, E_high = E
        elif numpy.size(E) == 1:
            E_low = E_high = E
        else:
            raise ValueError
        idx = numpy.logical_and(self.phen >= E_low, self.phen <= E_high)
        # print('plotting gamma distribution for {:.0f} points'.format(numpy.sum(idx)))
        if not idx.any() and E_low > self.phen.min() and E_high < self.phen.max():
            # if E_low == E_high:
            idx = find_nearest_idx(self.phen, numpy.mean([E_low, E_high]))
            spec = self.spec[idx, numpy.newaxis, :]
            phen = self.phen[idx]
        else:
            spec = self.spec[idx,:]
            phen = self.phen[idx]
        W = numpy.mean(spec, 0)
      
        W_min = numpy.amax([W_lim[0], W.min()])
        W_max = numpy.amin([W_lim[1], W.max()])
        # if Wlim[0] is None:
            # W_min = W.min()
        # else:
            # W_min = hist_min
        # W_max = W.max()
        W_hist, W_bins = numpy.histogram(W, bins=bins, density=normed, range=(W_min, W_max))
        return W, W_hist, W_bins
    
    def plot_gamma(self, fignum=None, E=[-numpy.inf, +numpy.inf], W_lim=[-numpy.inf, +numpy.inf], bins=50, **kwargs):
        
        W, W_hist, W_bins = self.calc_histogram(E=E, W_lim=W_lim, bins=bins)
        
        # W = numpy.mean(spec, 0)
        Wm = numpy.mean(W) #average power calculated
        sigm2 = numpy.mean((W - Wm)**2) / Wm**2 #sigma square (power fluctuations)
        M_calc = 1 / sigm2 #calculated number of modes        
        
        plt.figure(fignum)
        ax=plt.subplot(1,1,1)
        plt.bar(W_bins[:-1], W_hist, width=(W_bins[1]-W_bins[0]))        
        fit_p0 = [numpy.mean(W), Wm**2 / numpy.mean((W - Wm)**2)]
        _, fit_p = fit_gamma_dist(W_bins[:-1], W_hist, gamma_dist_function, fit_p0)
        Wm_fit, M_fit = fit_p # fit of average power and number of modes   
        plt.plot(W_bins[:-1], gamma_dist_function(W_bins[:-1], Wm_fit, M_fit), color='red', linewidth=2)        
        plt.text(0.98, 0.98, 'M_calc={:.2f}\n M_fit={:.2f}'.format(M_calc, M_fit), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        plt.xlabel('I [arb.units]')
        plt.ylabel('P(I)')
        plt.show()


#def corr_i(n_omega, cor_range, spec, norm=1):
#    n_event = spec.shape[1]
#    zeros = numpy.zeros((cor_range, n_event))
#    spec_int = numpy.r_[zeros, spec, zeros].astype(numpy.float32)
#    corr = numpy.zeros([n_omega, cor_range])
#    for i in range(n_omega):
#        for jj in range(cor_range):
#            if not jj%2:
#                ind_l = int(i - jj/2 + cor_range)
#                ind_r = int(i + jj/2 + cor_range)
#            else:
#                ind_l = int(i - (jj-1)/2 + cor_range)
#                ind_r = int(i + (jj-1)/2 + 1 + cor_range)
#            I_l = spec_int[ind_l,:]
#            I_r = spec_int[ind_r,:]
#            I_l_mean = numpy.mean(I_l)
#            I_r_mean = numpy.mean(I_r)
#            if norm:
#                if I_l_mean * I_r_mean != 0:
#                    corr[i,jj] = numpy.mean(I_l * I_r) / (I_l_mean * I_r_mean) # 2 order norm
#            else:
#                corr[i,jj] = numpy.mean(I_l * I_r) - (numpy.mean(I_l) * numpy.mean(I_r))
##                corr[i,jj] = numpy.mean(spec_int[ind_l,:] * spec_int[ind_r,:]) - (numpy.mean(spec_int[ind_l,:]) * numpy.mean(spec_int[ind_r,:])) # non-norm
#    return corr

##from numba import jit
##import numba as nb
##@nb.jit
##@nb.autojit
#def corr_i_f(spec_int, n_skip=1, norm=1):
#    count = 1
#    N = int(spec_int.shape[0]/n_skip)
#    corr = numpy.zeros([N,N])
#
#    if norm:
#        for i in range(N):
#            if count:
#                sys.stdout.write('\r')
#                sys.stdout.write('slice %i of %i' %(i,N-1))
#                sys.stdout.flush()
#            for j in range(N):
#                corr[i,j] = numpy.mean(spec_int[i*n_skip,:] * spec_int[j*n_skip,:]) / (numpy.mean(spec_int[i*n_skip,:]) * numpy.mean(spec_int[j*n_skip,:])) # 2 order norm
#    else:
#        for i in range(N):
#            if count:
#                sys.stdout.write('\r')
#                sys.stdout.write('slice %i of %i' %(i,N-1))
#                sys.stdout.flush()
#            for j in range(N):
#                corr[i,j] = numpy.mean(spec_int[i*n_skip,:] * spec_int[j*n_skip,:]) - (numpy.mean(spec_int[i*n_skip,:]) * numpy.mean(spec_int[j*n_skip,:])) # non-norm
#    if norm:
#        corr[numpy.isnan(corr)] = 1
#    else:
#        corr[numpy.isnan(corr)] = 0
#    return corr

def correlation2d(val, norm=0, n_skip=1, use_numba=numba_avail):
    N = int(val.shape[0] / n_skip)
    corr = numpy.zeros([N,N])
    if use_numba:
        corr_f_nb(corr, numpy.double(val), n_skip, norm, count=0)
    else:
        corr_f_np(corr, numpy.double(val), n_skip, norm, count=0)
    return corr

def correlation2d_center(n_corr, val, norm=0, use_numba=numba_avail):
    n_val, n_event = val.shape
    zeros = numpy.zeros((n_corr, n_event))
    val = numpy.r_[zeros, val, zeros]
    corr = numpy.zeros([n_val, n_corr])
    if use_numba:
        corr_c_nb(corr, n_corr, val, norm)
    else:
        corr_c_np(corr, n_corr, val, norm)
    return corr

@jit('void(double[:,:], double[:,:], int32, int32, int32)', nopython=True, nogil=True)
def corr_f_nb(corr, val, n_skip=1, norm=1, count=0):
    n_val = corr.shape[0]
    n_event = val.shape[1]
    for i in range(n_val):
        for j in range(n_val):
            means = 0
            meanl = 0
            meanr = 0
            for k in range(n_event):
                means += val[i*n_skip,k] * val[j*n_skip,k]
                meanl += val[i*n_skip,k]
                meanr += val[j*n_skip,k]
            means /= n_event
            meanl /= n_event
            meanr /= n_event
            if meanl == 0 or meanr == 0:
                if norm:
                    corr[i,j] = 1
                else:
                    corr[i,j] = 0
            else:
                if norm:
                    corr[i,j] = means / meanl / meanr
                else:
                    corr[i,j] = means - meanl * meanr


def corr_f_np(corr, val, n_skip=1, norm=1, count=0):
    n_val = corr.shape[0]
    for i in range(n_val):
        if count:
            sys.stdout.write('\r')
            sys.stdout.write('slice %i of %i' %(i, n_val-1))
            sys.stdout.flush()
        for j in range(n_val):
            means = numpy.mean(val[i*n_skip,:] * val[j*n_skip,:])
            meanl = numpy.mean(val[i*n_skip,:])
            meanr = numpy.mean(val[j*n_skip,:])
            if norm:
                corr[i,j] = means / meanl / meanr
            else:
                corr[i,j] = means - meanl * meanr

    if norm:
        corr[numpy.isnan(corr)] = 1
    else:
        corr[numpy.isnan(corr)] = 0

@jit('void(double[:,:], int32, double[:,:], int32)', nopython=True, nogil=True)
def corr_c_nb(corr, n_corr, val, norm):
    n_event = len(val[0])
    n_val = len(val) - n_corr*2
    for i in range(n_val):
        for j in range(n_corr):
            if not j%2:
                ind_l = int(i - j/2 + n_corr)
                ind_r = int(i + j/2 + n_corr)
            else:
                ind_l = int(i - (j-1)/2 + n_corr)
                ind_r = int(i + (j-1)/2 + 1 + n_corr)
            means = 0
            meanl = 0
            meanr = 0
            for k in range(n_event):
                means += val[ind_l, k] * val[ind_r, k]
                meanl += val[ind_l, k]
                meanr += val[ind_r, k]
            means /= n_event
            meanl /= n_event
            meanr /= n_event

            if meanl == 0 or meanr == 0:
                if norm:
                    corr[i,j] = 1
                else:
                    corr[i,j] = 0
            else:
                if norm:
                    corr[i,j] = means / meanl / meanr
                else:
                    corr[i,j] = means - meanl * meanr


def corr_c_np(corr, n_corr, val, norm):
    for i in range(n_val):
        for j in range(n_corr):
            if not j%2:
                ind_l = int(i - j/2 + n_corr)
                ind_r = int(i + j/2 + n_corr)
            else:
                ind_l = int(i - (j-1)/2 + n_corr)
                ind_r = int(i + (j-1)/2 + 1 + n_corr)
            means = numpy.mean(val[ind_l,:] * val[ind_r,:])
            meanl = numpy.mean(val[ind_l,:])
            meanr = numpy.mean(val[ind_r,:])

            if meanl == 0 or meanr == 0:
                corr[i,j] = 0
            else:
                if norm:
                    corr[i,j] = means / meanl / meanr
                else:
                    corr[i,j] = means - meanl * meanr



def read_spec_arr_genout(file_path, units='eV'):

    print('reading')

    spar = SpectrumArray()
    arr = numpy.genfromtxt(file_path)
    if units in ['eV', 'ev']:
        spar.omega = deepcopy(arr[:, 0]) / hr_eV_s
    elif units == 'nm':
        spar.omega = 2 * numpy.pi * speed_of_light / deepcopy(arr[:, 0]) * 1e9
    else:
        raise ValueError('units should be eithger "ev" or "nm"')
    spar.spec = numpy.delete(arr,(0,1),1)
    spar.path = file_path

    res_pow = len(spar.omega) * numpy.mean(spar.omega) / numpy.abs(spar.omega[-1] - spar.omega[0])
    print('  done')
    print('res_pow', res_pow)

    return spar

def read_spec_arr_raw(file_path_mask, E_center=9350, dE=0.1, resolution=(1080,1920), x_pixel=900):

    print('reading')

    # file_path_mask = r'D:\ownColud\projects\2017_HIREX_spectra_analysis_scripts\binary_old\hirex_Si333_*.raw'
    # dir = os.path.dirname(file_path_mask)
    # rcParams["savefig.directory"] = dir
    filenames = glob.glob(file_path_mask)
    #filename = r'D:\ownColud\projects\2017_HIREX_spectra_analysis_scripts\hirex_Si333_E9350eV_000010.raw'
    d = []
    for filename in filenames:
        # print(filename)
        l0 = numpy.fromfile(filename, dtype=numpy.uint16)
        l = l0.reshape(resolution)[:,x_pixel]
        d.append(l)

    spar = SpectrumArray()
    spar.spec = numpy.array(d).T.astype(numpy.float)
    if spar.spec.size == 0:
        print('  none found')
        return None
    # spec = spec[:,0:500]
    # spec -= 480
    # spec[spec < 0] = 0
    phen = numpy.arange(spar.spec.shape[0])
    phen = phen - numpy.mean(phen)

    spar.phen = phen * dE + E_center - numpy.amax(phen * dE) / 2
    spar.path = file_path_mask

    print('  done')
    return spar

def read_spec_arr_hdf5(file_path, E_center=9350, dE= 0.1, x_pixel=900):
    f = h5py.File(file_path, 'r')
    pixels = f['INSTRUMENT/SA1_XTD9_HIREX/CAM/BRAGG:daqOutput/data/image/pixels']
    train_id = f['INSTRUMENT/SA1_XTD9_HIREX/CAM/BRAGG:daqOutput/data/trainId']
    spar = SpectrumArray()
    spar.spec = pixels[:,:,x_pixel].T
    phen=numpy.arange(spar.spec.shape[0])
    phen = phen - numpy.mean(phen)
    spar.phen = phen * dE + E_center

    spar.train_id = train_id
    spar.path = file_path

    print('  done')
    return spar

class SpectrumCorrelationsCenter():
    """ :class SpectrumCorrelationsCenter: Class to represent correlation of a spectrum.  """
    def __init__(self):
        self.corr = numpy.array([])
        self.domega = numpy.array([])
        self.omega = numpy.array([])
        pass

    def bin_omega_step(self, freq_bin=1):
        omega = self.omega
        omega_b = omega[:(omega.shape[0] // freq_bin) * freq_bin]
        self.omega = omega_b[::freq_bin]
        corr_b = self.corr[:(self.corr.shape[0] // freq_bin) * freq_bin]
        self.corr = numpy.mean(corr_b.reshape(int(corr_b.shape[0] / freq_bin), freq_bin, corr_b.shape[1]), axis=1)
        pass

    def bin_phen(self, dE=None):
        # dE_orig = self.domega[1] * hr_eV_s
        phen = self.phen
        dE_orig = abs(phen[1] - phen[0])
        if dE_orig == 0:
            return
        freq_bin = int(dE / dE_orig)
        if freq_bin > 1:
            self.bin_omega_step(freq_bin)
        else:
            pass

    @property
    def phen(self):
        return self.omega * hr_eV_s

    @property
    def dphen(self):
        return self.domega * hr_eV_s

    def mirror(self):
        corr_symm = numpy.c_[numpy.fliplr(self.corr[:,1:]), self.corr] # second order
        domega = numpy.linspace(-self.domega[-1], self.domega[-1], corr_symm.shape[1])
        return corr_symm, domega

    def plot(self, fignum=None):
        corr_symm, domega = self.mirror()
        plt.figure(fignum)
        plt.clf()
        plt.pcolormesh(domega * hr_eV_s, self.omega * hr_eV_s, corr_symm)
        plt.xlabel('dE [eV]')
        plt.ylabel('E [eV]')
        plt.axis('tight')
        plt.show()

    def fit_g2func(self, func, thresh=0.1):
        'if norm then correlation function is normalized as g2 = g2/g2[0] + 1'

        corr = self.corr
        omega = self.omega
        domega = self.domega

        idx = numpy.array([numpy.argmin(numpy.abs(self.omega_orig - omegai)) for omegai in self.omega])
        intens = numpy.mean(self.spec[idx,:], axis=1)
        intens /= numpy.amax(intens)

        def func_fit(domega, g2, func, fit_p0=None):
            import scipy.optimize as opt
            if fit_p0 is None:
                fit_p0 = [2 * numpy.pi / domega[int(len(domega)/2)], g2[0] - g2[-1], g2[-1]]
            errfunc = lambda t, x, y: (func(x, t[0], t[1], t[2])-1)**1 - (y-1)**1
            fit_p, success = opt.leastsq(errfunc, fit_p0[:], args=(domega, g2), xtol=1e-5, gtol=1e-5, maxfev=500)
            if success not in [1,2,3,4]:
                fit_p = fit_p0
            fit_p[0] = abs(fit_p[0])
#            fit_t, success = opt.leastsq(errfunc, t0[:], args=(domega, g2))
            g2f = func(domega, fit_p[0], fit_p[1], fit_p[2])
            return (g2f, fit_p)

#        def func_fit_freqjitter(domega, g2, func, fit_p0=None):
#            import numpy as np
#            import scipy.optimize as opt
#            if fit_p0 is None:
#                fit_p0 = [2 * numpy.pi / domega[int(len(domega)/2)], g2[0] - g2[-1], g2[-1], 2 * numpy.pi / domega[int(len(domega)-1)] / 3]
#            errfunc = lambda t, x, y: (func(x, t[0], t[1], t[2], t[3])-1)**1 - (y-1)**1
#            fit_p, success = opt.leastsq(errfunc, fit_p0[:], args=(domega, g2), xtol=1e-5, gtol=1e-5, maxfev=500)
#            if success not in [1,2,3,4]:
#                fit_p = fit_p0
#            fit_p[0] = abs(fit_p[0])
##            fit_t, success = opt.leastsq(errfunc, t0[:], args=(domega, g2))
#            g2f = func(domega, fit_p[0], fit_p[1], fit_p[2], fit_p[3])
#            return (g2f, fit_p)

        g2_measured = []
#        fit_params = []
        fit_t = []
        fit_contrast = []
        fit_pedestal = []
        g2_fit = []
        fit_p = None

        for intens_i, g2i in zip(intens, corr):
#            if norm:
#                g2i = sl/sl[0] + 1
#            else:
            g2_measured.append(g2i)
#            if g2i is not None:
#            print(g2i[0], domega[0], func)
            fit_p = None
            if intens_i > thresh:
                g2ti, fit_p = func_fit(domega, g2i, func, fit_p)
#                g2ti, fit_p = func_fit_freqjitter(domega, g2i, func, fit_p)
#                func_fit_freqjitter
            else:
                g2ti = numpy.ones_like(g2i)
                fit_p = [0, 1, 1]
#            print(ti)
            fit_t.append(float(fit_p[0]))
            fit_contrast.append(float(fit_p[1]))
            fit_pedestal.append(float(fit_p[2]))
#            fit_params.append(fit_p)
            g2_fit.append(g2ti)
#            else:
#                fit_t.append(numpy.nan)
#                g2_fit.append(numpy.zeros_like(g2i))

        g2fit = FitResult()

#        g2fit.intens = intens
        g2fit.g2_fit = numpy.array(g2_fit)
        g2fit.g2_measured = numpy.array(g2_measured)
        g2fit.fit_t = numpy.array(fit_t)
        g2fit.fit_contrast = numpy.array(fit_contrast)
        g2fit.fit_pedestal = numpy.array(fit_pedestal)
#        g2fit.fit_params = fit_params
        g2fit.domega = domega
        g2fit.omega = omega
        g2fit.func = func
        g2fit.spec = self.spec

        return g2fit

    def fft(self):

        rec = Reconstr()

        corr_symm, domega = self.mirror()

        rec.spec = self.spec
        rec.omega_bin = self.omega
        rec.domega = self.domega
        rec.omega_orig = self.omega_orig

        recon = numpy.fft.ifftshift(corr_symm, 1)
        recon = numpy.fft.fft(recon, axis=1)
        recon = numpy.fft.fftshift(recon, 1)
        #reconstr_f = numpy.roll(reconstr_f, -1, 1)

        #reconstr_fn = reconstr_f
        rec.recon = abs(recon) #notnormalized
        #reconstr_fn = abs(reconstr_f) * sp_mean[:, numpy.newaxis] #normalized (was good)
        #20
        rec.m_idx = int((recon.shape[1] - 1) / 2) #middle index
        return rec

class SpectrumCorrelationsFull():
    """ :class SpectrumCorrelationsFull: Class to represent correlations of a spectrum.  """
    def __init__(self):
        pass

    def plot(self, limits=None, fignum=None):
        plt.figure(fignum)
        if limits is None:
            plt.pcolormesh(self.omega * hr_eV_s, self.omega * hr_eV_s, self.corr)
        else:
            plt.pcolormesh(self.omega * hr_eV_s, self.omega * hr_eV_s, self.corr, vmin=limits[0], vmax=limits[1])
        plt.axis('tight')
        plt.xlabel('E [eV]')
        plt.ylabel('E [eV]')
        plt.colorbar()
        plt.show()

class Reconstr():
    """ :class Reconstr: Class representing reconstruction (???). """
    def __init__(self):
        self.recon = numpy.zeros((1,1))
        pass

    def abs(self):
        self.recon = abs(self.recon)

    def time_scale(self):
        Tw = 2 * numpy.pi / (self.domega[2] - self.domega[1])
        return numpy.linspace(-Tw, Tw, self.recon.shape[1]) * 1e15 / 2

    def plot(self, fignum = None, show_um=0, plot_lineoffs=1, autosc_thresh=0, xlim = (None,None), ylim = (None,None), cmap='viridis', line_offs_height = 0.45, figsize = (6.4, 4.8)):

        spec = self.spec / numpy.nanmax(self.spec)
        omega_bin = self.omega_bin
        omega = self.omega_orig
        recon = self.recon
        m_idx = self.m_idx
        freq_bin = self.omega_orig.size / self.omega_bin.size
        timeval = self.time_scale()
        if show_um:
            timeval *= speed_of_light * 1e6 / 1e15

        lineoff_colors = ['r', 'g', 'c', 'b', 'm']
        # lineoff_colors = ['m', 'b', 'c', 'g', 'r']
        left, width = 0.18, 0.57
        bottom, height = 0.14, 1-line_offs_height
        left_h = left + width + 0.02 - 0.02
        bottom_h = bottom + height + 0.02 - 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, line_offs_height-0.2]
        rect_histy = [left_h, bottom, 0.15, height]

        plt.figure(fignum, figsize = figsize)
        plt.clf()
        ax1 = plt.axes(rect_scatter)
        ax0 = plt.axes(rect_histx, sharex=ax1)
        ax2 = plt.axes(rect_histy, sharey=ax1)
        plt.subplots_adjust(wspace=0, hspace=0)

        #pulse autocorr. shape plot
        reconstr_pproj = numpy.mean(recon, axis=0) #power-like projection
        reconstr_pproj /= numpy.amax(reconstr_pproj)
        halfline = numpy.ones_like(reconstr_pproj) / 2
        ax0.plot(timeval[m_idx:], halfline[m_idx:], color='grey', linewidth=1, linestyle = '--')
        # ax0.plot(timeval[m_idx:], reconstr_pproj[m_idx:], color='k', linewidth=2)

        # spectrum plot
        spec_min = numpy.amin(spec, axis=1)
        spec_max = numpy.amax(spec, axis=1)
        spec_mean = numpy.mean(spec, axis=1)
        ax2.fill_betweenx(hr_eV_s*omega, spec_min, spec_max, facecolor=[0.9,0.9,0.9], edgecolor='w')
        ax2.plot(spec[:,0], hr_eV_s*omega, color='gray', linewidth=1)
        ax2.plot(spec_mean, hr_eV_s*omega, color='k', linewidth=2)

        #main figure
        ax1.pcolormesh(timeval[m_idx:], hr_eV_s*omega_bin, recon[:,m_idx:], cmap=cmap)

        if plot_lineoffs is not None and plot_lineoffs != 0 and type(plot_lineoffs) in [list,numpy.ndarray]:
            spec_arr_bin = numpy.array([find_nearest_idx(numpy.array(hr_eV_s*omega_bin), i) for i in plot_lineoffs])
            spec_arr = numpy.array([find_nearest_idx(numpy.array(hr_eV_s*omega), i) for i in plot_lineoffs])
        else:
            lineoffs = numpy.arange(5)
            fwhm_spec = fwhm3(spec_mean)
            spec_arr_bin = [int((fwhm_spec[2][0] + numpy.floor(fwhm_spec[2][1] - fwhm_spec[2][0]) / (len(lineoffs) - 1) * lineoff) / freq_bin) - 1 for lineoff in lineoffs]
            spec_arr = [int((fwhm_spec[2][0] + numpy.floor(fwhm_spec[2][1] - fwhm_spec[2][0]) / (len(lineoffs) - 1) * lineoff)) - 1 for lineoff in lineoffs]
        print(spec_arr_bin)
        print(spec_arr)
        
        for i, s in enumerate(spec_arr_bin):
            reconstr_lineoff = deepcopy(recon[s,:])
            reconstr_lineoff /= numpy.amax(reconstr_lineoff)
            ax0.plot(timeval[m_idx:], reconstr_lineoff[m_idx:], color=lineoff_colors[i], linewidth=1)

        for i, s in enumerate(spec_arr):
            ax2.plot([0, numpy.amax(spec_mean)], [hr_eV_s*omega[s], hr_eV_s*omega[s]], color=lineoff_colors[i], linewidth=2)
            ax1.plot([timeval[m_idx],timeval[-1]], [hr_eV_s*omega[s], hr_eV_s*omega[s]], color=lineoff_colors[i], linewidth=2, alpha=0.3)

        for tl in ax2.get_yticklabels():
            tl.set_visible(False)
        for tl in ax2.get_xticklabels():
            tl.set_visible(False)
        for tl in ax0.get_xticklabels():
            tl.set_visible(False)

        ax0.set_ylim(bottom=0)
        ax2.set_xlim(left=0)

        if show_um:
            ax1.set_xlabel('s [um]')
        else:
            ax1.set_xlabel('t [fs]')

        ax2.set_xlabel('spectra')
        ax1.set_ylabel('$E_{photon}$ [eV]')
        ax1.axis('tight')


        if autosc_thresh > 0:
            t_idx_lim = numpy.where(reconstr_pproj > numpy.amax(reconstr_pproj) * autosc_thresh)[0][-1]
            e_idx_lim1 = numpy.where(spec_mean > numpy.amax(spec_mean) * autosc_thresh)[0][-1]
            e_idx_lim2 = numpy.where(spec_mean > numpy.amax(spec_mean) * autosc_thresh)[0][0]
            ax2.set_ylim([hr_eV_s * omega[e_idx_lim2], hr_eV_s * omega[e_idx_lim1]])
            ax0.set_xlim([0, timeval[t_idx_lim] * 1.1])
            
        if xlim != (None, None):
            ax1.set_xlim([xlim[0], xlim[1]])
        if ylim != (None, None):
            ax1.set_ylim([ylim[0], ylim[1]])
            
        plt.show()

#def g2_ft(domega, T, bg):
#    x = domega * T / 2
#    g2 = 1 + (numpy.sin(x)**2 / x**2) * (1 - bg) + bg
#    g2[0] = 2
#    return g2
#
#def g2_gauss(domega, T, bg):
#    x = domega * T / (2 * numpy.sqrt(2 * numpy.log(2)))
#    g2 = 1 + numpy.exp(-x**2) * (1 - bg) + bg
#    g2[0] = 2
#    return g2

def g2_ft(domega, T, g20, offset):
    x = domega * T / 2
#    g2 = (offset + (numpy.sin(x)**2 / x**2)) * g20
#    g2[0] = (offset + 1) * g20
    g2 = offset + numpy.sinc(x/numpy.pi)**2 * g20

    return g2

def g2_gauss(domega, T, g20, offset):
    x = domega * T / (2 * numpy.sqrt(2 * numpy.log(2)))
    g2 = offset + numpy.exp(-x**2) * g20
    return g2

def g2_gauss1(domega, T, g20, offset, T1):
    x = domega * T / (2 * numpy.sqrt(2 * numpy.log(2)))
    x1 = domega * T1 / (2 * numpy.sqrt(2 * numpy.log(2)))
    g2 = offset + numpy.exp(-x**2) * g20 - numpy.exp(-x**2) + 1
    return g2

#def g2_gauss(domega, T, g20, offset):
#    x = domega * T / (2 * numpy.sqrt(2 * numpy.log(2)))
#    g2 = offset + numpy.exp(-x**2) * (g20 - 1) * offset
#    return g2

class FitResult():
    def __init__(self):
        self.fit_t = numpy.array([0])
        self.omega = numpy.array([0])
        self.fit_contrast = numpy.array([0])
        self.fit_pedestal = numpy.array([0])
        pass

    def plot_t(self, fignum=None, spar=None, thresh=0.2, xlim=(None, None)):

        plt.figure(fignum)
        plt.clf()

        if spar is not None:

            ax_spec = plt.gca()
            spec_min = numpy.amin(spar.spec, axis=1)
            spec_max = numpy.amax(spar.spec, axis=1)
            ax_spec.fill_between(spar.omega * hr_eV_s, spec_min, spec_max, facecolor=[0.9,0.9,0.9], edgecolor='w')
            #plt.plot(phen,spec,color=[0.9,0.9,0.9])
            ax_spec.plot(spar.omega * hr_eV_s, spar.spec[:,0], color=[0.5, 0.5, 0.5])
            ax_spec.plot(spar.omega * hr_eV_s, numpy.mean(spar.spec, axis=1), 'k', linewidth=3)
            ax_spec.set_ylabel('Spec.density [arb.units]')
            plt.xlabel('E [eV]')
            ax_spec.axis('tight')
            ax_spec.set_ylim(bottom=0)

            ax_t = ax_spec.twinx()
            ax_t.yaxis.label.set_color('r')
            ax_t.tick_params(axis='y', which='both', colors='r')

            idx = numpy.array([numpy.argmin(numpy.abs(spar.omega - omegai)) for omegai in self.omega])
            intens = numpy.mean(spar.spec[idx,:], axis=1)

            idx = numpy.where(intens > numpy.amax(intens) * thresh)[0]
            ax_t.scatter(self.omega[idx] * hr_eV_s, self.fit_t[idx] * 1e15, s=(intens[idx] / numpy.amax(intens[idx]))**2 * 10, color='r')
            ax_t.set_ylim([0, numpy.nanmax(self.fit_t[idx]) * 1.5 * 1e15])

        else:
            ax_t = plt.gca()
            ax_t.scatter(self.omega * hr_eV_s, self.fit_t * 1e15, color='r')
            ax_t.set_ylim([0, numpy.nanmax(self.fit_t) * 1.5 * 1e15])


        #        ax_t.axis('tight')

        ax_t.set_ylabel('fwhm group duration at E [fs]')
        ax_t.set_xlabel('E [eV]')
        plt.text(0.98, 0.98, 'fit function: {:s}'.format(self.func.__name__), horizontalalignment='right', verticalalignment='top', transform=ax_t.transAxes)

        if xlim != (None, None):
            ax_t.set_xlim([xlim[0], xlim[1]])
        plt.show()

    def plot_g2(self, phen=None, fignum=None, plot_measured=True, plot_fit=True):

        if phen is None:
            idx = int(self.omega.size / 2)
        else:
            idx = (numpy.abs(self.omega * hr_eV_s - phen)).argmin()

        plt.figure(fignum)
        plt.clf()
        if plot_measured:
            plt.plot(self.domega * hr_eV_s, self.g2_measured[idx])
        if plot_fit:
            plt.plot(self.domega * hr_eV_s, self.g2_fit[idx])
        plt.axis('tight')
#        plt.ylim([0.9, 2.1])
        plt.xlim(left=0)
        plt.plot([self.domega[0] * hr_eV_s, self.domega[-1] * hr_eV_s], [1,1], '--k')
        plt.ylabel('g2')
        plt.xlabel('dE [eV]')
        ax = plt.gca()
        plt.text(0.98, 0.98, 'T={:.2f} fs @ E={:.2f} eV with {:s}'.format(self.fit_t[idx]*1e15 , self.omega[idx] * hr_eV_s, self.func.__name__), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        plt.show()
        

def fit_gamma_dist(W_bins, W_hist, func, fit_p0=None):
   if fit_p0 is None:
       fit_p0 = [numpy.mean(W_bins), 100]
   errfunc = lambda t, x, y: func(x, t[0], t[1]) - y
   fit_p, success = scipy.optimize.leastsq(errfunc, fit_p0[:], args=(W_bins, W_hist), xtol=1e-5, gtol=1e-5, maxfev=500)
   if success not in [1,2,3,4]:
       fit_p = fit_p0
   fit_p[0] = abs(fit_p[0])
   #            fit_t, success = opt.leastsq(errfunc, t0[:], args=(domega, g2))
   fit_r = func(W_bins, fit_p[0], fit_p[1])
   return (fit_r, fit_p)

def gamma_dist_function(W, Wm, M):
   Pw = M**M/scipy.special.gamma(M) * (W/Wm)**(M-1) * numpy.exp(-M * W / Wm) / Wm
   return Pw
