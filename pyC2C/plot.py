import numpy as np
from .helperFunctions import (calc_sample_rate, intensity_matrix, shift_intensity_matrix,
                              intensity_matrix_padding, _shift_2drt, _solvent_cutoff)
from .cmaps import seb_map
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
from matplotlib import animation
import matplotlib.colors as colors
import os


def plot1d(ax, rt, intensity, linecolor='blue', linewidth=None, label=None):

    """

    Parameters
    ----------
    ax : matplotlib.pyplot.ax
        Axis created by plt.subplots
    rt : list
        Array of retention times
    intensity : list
        Array of intensities to plot
    linecolor : str, optional
        Color of line in plot, e.g. 'black', 'blue', 'orange', 'green'
    linewidth : float, optional
        Width of line in plot
    label : str, optional
        Label of plotted line

    """

    ax.plot(rt, intensity, color=linecolor, lw=linewidth, label=label)
    ax.set_xlabel("retention time [min]")
    ax.set_ylabel("intensity")


def _plot2d(fig, ax, matrix, rt, mod_time, color_range=(None, None), cmap=seb_map, relative=False, axis_labels=None, cbar=True):

    """
    :param fig: fig of plt.subplots
    :param ax: ax of plt.subplots
    :param matrix: 2d array of intensities
    :param rt: array of retention times in minutes
    :param mod_time: modulation time in seconds
    :param color_range: minimum and maximum value for color bar
    :param relative: if true, intensities will be normalized and the ticks of colorbar are set
    :param axis_labels:
    """

    dim_x, dim_y = np.shape(matrix)

    run_time = np.round(rt[-1], decimals=0)
    rt_offset = rt[0]

    y, x = np.meshgrid(np.linspace(0, mod_time, num=dim_y), np.linspace(rt_offset, run_time, num=dim_x))

    if relative:
        matrix = matrix / np.max(matrix) * 100

    cmin = color_range[0]
    cmax = color_range[1]

    if cmin is None:
        cmin = np.min(matrix)

    if cmax is None:
        cmax = np.max(matrix)

    c = ax.contourf(x, y, matrix, cmap=cmap, vmin=cmin, vmax=cmax, levels=500)

    ax.axis([x.min(), x.max(), y.min(), y.max()])

    if axis_labels is None:
        ax.set_xlabel('$^1D$ time [min]')
        ax.set_ylabel('$^2D$ time [s]')
    else:
        ax.set_xlabel('$^1D-$' + axis_labels[0] + ' [min]')
        ax.set_ylabel('$^2D-$' + axis_labels[1] + ' [s]')

    if cbar:
        if relative:
            cbar = fig.colorbar(c, ax=ax, aspect=30, ticks=range(0, 100, 10))
        else:
            cbar = fig.colorbar(c, ax=ax, aspect=30)

        cbar.formatter.set_powerlimits((0, 3))
        cbar.formatter.set_useMathText(True)

    return c


def plot2d(fig, ax, rt, intensity, mod_time, shift=0, solvent_cut=None, color_range=(None, None), cmap=seb_map, axis_labels=None, cbar=True, relative=True):

    """

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        object created by plt.subplots()
    ax : matplotlib.pyplot.ax
        object created by plt.subplots()
    rt : list
        Array of retention times in minutes
    intensity : list
        Array of intensities to plot
    mod_time : float
        Modulation time in seconds
    shift : float, optional
        Seconds to shift the 2D time axis. Negative values are allowed.
    solvent_cut : float, optional
        1d retention time in minutes. All intensities before this time will be set to zero.
    color_range : tuple, optional
        Set minimum and maximum value for color bar. Use None for default value.
    cmap : matplotlib.cmap, optional
        Colormap of matplotlib (or maps defined in pyPAAC.cmaps)
    axis_labels : tuple of str, optional
        Column names of 1D and 2D. Set to None to get only >1D time< and >2D time<.
    cbar : bool, optional
        Set False to remove color bar.
    relative : bool, optional
        Set True for relative intensities and False for absolute intensities.

    Returns
    -------
    2d matrix of intensities

    """

    sample_rate = calc_sample_rate(rt)

    if solvent_cut is not None:
        intensity = _solvent_cutoff(rt, intensity, solvent_cut)

    if shift != 0:

        rt, intensity = _shift_2drt(rt, intensity, sample_rate, shift, mod_time)

    if rt is not None:

        matrix = intensity_matrix(rt, intensity, mod_time, sample_rate)

        # if shift != 0:
        #    matrix = shift_intensity_matrix(matrix, shift, sample_rate)

        plot = _plot2d(fig, ax, matrix, rt, mod_time, color_range, cmap,
                        relative=relative, axis_labels=axis_labels, cbar=cbar)

        return plot


def plot2d_padding(fig, ax, rt, intensity, mod_time, mod_tol, mod_offset, shift=0,
                   color_range=(None, None), cmap=seb_map, axis_labels=None, cbar=True, test_threshold=1.01):
    """
    Creates 2d plot with corretion of shifting modulation times with autocorrelation
    between the current modulation and the next.

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        figure object created with plt.subplots()
    ax : matplotlib.pyplot.ax
        axis object created with plt.subplots()
    rt : list
        Array of retention times in minutes
    intensity : list
        Array of intensities for 2D plot
    mod_time : float
        Modulation time in seconds
    mod_tol : float
        Time tolerance of modulation in seconds. It defines the intervall that contains the real modulation time
    mod_offset : float
        Time delay of the flow modulator in minutes. Time points under the offset will be removed
        from retention time array for 2d plots
    shift : float, optional
        Seconds to shift the 2D time axis. Negative values are allowed.
    color_range : tuple, optional
        Set minimum and maximum value for color bar. Use None for default value.
    cmap : matplotlib.cmap, optional
        Colormap of matplotlib (or maps defined in pyC2C.cmaps)
    axis_labels : tuple of str, optional
        Column names of 1D and 2D. Set to None to get only >1D time< and >2D time<.
    cbar : bool, optional
        Set to False, to remove color bar.
    test_threshold : float
        To test, if there is a maximum in calculation of autocorrelation, the mean with and without the maximum is calculated.
        The test value is the division of mean with to mean without the maximum. Is must be lager than 1. If quotient is lower than
        the threshold, the modulation will not be corrected and function will take the expected modulation time.

    Returns
    -------
    2d array of intensities
    """

    sample_rate = calc_sample_rate(rt)

    rt_corrected, matrix, breaks = intensity_matrix_padding(rt, intensity, mod_time, mod_tol, mod_offset,
                                                            sample_rate, test_threshold)

    if shift != 0:
        matrix = shift_intensity_matrix(matrix, shift, sample_rate)

    plot = _plot2d(fig, ax, matrix, rt_corrected, mod_time, color_range, cmap, relative=True, axis_labels=axis_labels,
                   cbar=cbar)

    return plot


def create_animation(plot_dir, width, height, interval=500):

    list_of_im_paths = [plot_dir + "/" + f for f in os.listdir(plot_dir)
                        if f.endswith('.png')]

    fig = plt.figure(figsize=(width, height), dpi=300)
    plt.axis('off')

    # initiate an empty  list of "plotted" images
    myimages = []

    # loops through available png:s
    for fname in list_of_im_paths:
        # Read in picture
        img = mgimg.imread(fname)
        imgplot = plt.imshow(img)

        # append AxesImage object to the list
        myimages.append([imgplot])

    # create an instance of animation
    my_anim = animation.ArtistAnimation(fig, myimages, interval=interval)

    path_out = plot_dir + "/all_plots.gif"
    my_anim.save(path_out)

def _plot2d_scale(fig, ax, matrix, rt, mod_time,
                  color_range=(None, None),
                  color_scale="linear",
                  cmap=seb_map,
                  relative=False,
                  axis_labels=None,
                  cbar=True):

    dim_x, dim_y = np.shape(matrix)

    run_time = np.round(rt[-1], decimals=0)
    rt_offset = rt[0]

    y, x = np.meshgrid(
        np.linspace(0, mod_time, num=dim_y),
        np.linspace(rt_offset, run_time, num=dim_x)
    )

    if relative:
        matrix = matrix / np.max(matrix) * 100

    cmin = color_range[0]
    cmax = color_range[1]

    if cmin is None:
        cmin = np.min(matrix)

    if cmax is None:
        cmax = np.max(matrix)

    # -----------------------------
    # Color normalization
    # -----------------------------
    if color_scale == "log":

        # LogNorm cannot contain zeros
        positive = matrix[matrix > 0]
        if cmin <= 0:
            cmin = positive.min()

        norm = colors.LogNorm(vmin=cmin, vmax=cmax)

    elif color_scale == "sqrt":

        norm = colors.PowerNorm(
            gamma=0.5,
            vmin=cmin,
            vmax=cmax
        )

    else:

        norm = None

    c = ax.contourf(
        x,
        y,
        matrix,
        cmap=cmap,
        norm=norm,
        levels=500
    )

    ax.axis([x.min(), x.max(), y.min(), y.max()])

    if axis_labels is None:
        ax.set_xlabel('$^1D$ time [min]')
        ax.set_ylabel('$^2D$ time [s]')
    else:
        ax.set_xlabel('$^1D-$' + axis_labels[0] + ' [min]')
        ax.set_ylabel('$^2D-$' + axis_labels[1] + ' [s]')

    if cbar:
        if relative:
            cb = fig.colorbar(c, ax=ax, aspect=30, ticks=range(0, 100, 10))
        else:
            cb = fig.colorbar(c, ax=ax, aspect=30)

        cb.formatter.set_powerlimits((0, 3))
        cb.formatter.set_useMathText(True)

    return c

def plot2d_scale(fig, ax, rt, intensity, mod_time,
                 shift=0,
                 solvent_cut=None,
                 color_range=(None, None),
                 color_scale="linear",
                 cmap=seb_map,
                 axis_labels=None,
                 cbar=True,
                 relative=True):

    sample_rate = calc_sample_rate(rt)

    if solvent_cut is not None:
        intensity = _solvent_cutoff(rt, intensity, solvent_cut)

    if shift != 0:
        rt, intensity = _shift_2drt(
            rt,
            intensity,
            sample_rate,
            shift,
            mod_time
        )

    if rt is not None:

        matrix = intensity_matrix(
            rt,
            intensity,
            mod_time,
            sample_rate
        )

        plot = _plot2d_scale(
            fig,
            ax,
            matrix,
            rt,
            mod_time,
            color_range=color_range,
            color_scale=color_scale,
            cmap=cmap,
            relative=relative,
            axis_labels=axis_labels,
            cbar=cbar
        )

        return plot
