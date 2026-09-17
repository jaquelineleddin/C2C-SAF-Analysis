import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import olefile as ole
from pyteomics import mzml
import os
import struct

from .helperFunctions import _integrateScans, scan_annotation, _mzCluster, _combine_mz


def load_file(filepath):
    """

    Parameters
    ----------
    filepath : string
        Path of the file that contains retention time and MS scans. Supported formats are Shimadzu GC .qgd and mzML.

    Returns
    -------
        Object of the class defined by file format
    """

    if os.path.isfile(filepath):
        # get extension from file name
        ext = os.path.splitext(filepath)[1]

        if ext.lower() == ".mzml":
            # mzML file
            return mzMLFile(filepath)

        elif ext.lower() == ".qgd":
            # Shimadzu GC data
            return QGDFile(filepath)

        else:
            print("Unsupported file format. Supported formats are qgd and mzML")
    else:
        print("File does not exist.")


class MS1Scan:

    """
    Class to store MS1 data.
    """

    def __init__(self, rt, mz, intensity):

        self.scanNumber = None
        self.rt = rt
        self.mz = mz
        self.intensity = intensity

    def printContent(self):

        print('scan number: {}' .format(self.scanNumber))
        print('retention time: %.3f' % self.rt)
        print('m/z: {}' .format(self.mz))
        print('intensity: {}'.format(self.intensity))

    def plotScan(self, ax, color='blue', linewidth=1,
                 annotation=True, threshold=0, annotation_dict=None):
        """
        Create plot of MS1 scan with optional annotation of m/z values.

        Parameters
        ----------
        ax : matplotlib.pyplot.ax
            Axis object of matplotlib.pyplot created by fig, ax = plt.subplots()
        color : str (default 'blue')
            Color of lines. Use 'red', 'green'... or '#rrggbb' for hex-color
        linewidth : float (default 1)
            Width of lines
        annotation : bool (default True)
            Annotate m/z value for lines with high intensity
        threshold : float (default 0)
            Plot only m/z values with intensity higher than threshold
        annotation_dict : dict (optional)
            Dictionary for signal annotation with keys:
                'mz range' as tupel, 'mz diff' as float, 'threshold' for minimum intensity for annotation

        """

        df = pd.DataFrame({"mz": self.mz, "intensity": self.intensity})

        if threshold != 0:
            df = df[df['intensity'] >= threshold]
            df.reset_index(drop=True, inplace=True)

        ax.vlines(x=df['mz'].values, ymin=0, ymax=df['intensity'].values, colors=color, lw=linewidth)
        ax.set_xlabel('m/z')
        ax.set_ylabel('counts')
        ax.set_title('{} min'.format(round(self.rt, ndigits=3)))
        ax.ticklabel_format(style='sci', useMathText=True, axis='y', scilimits=(-1, 3))

        if annotation:

            if annotation_dict is None:
                idx, max_inty = scan_annotation(df['mz'].values, df['intensity'].values,
                                                threshold=0.05 * df['intensity'].max())

            else:
                idx, max_inty = scan_annotation(df['mz'].values, df['intensity'].values,
                                                threshold=annotation_dict['threshold'],
                                                min_mz_diff=annotation_dict['mz diff'], mz_range=annotation_dict['mz range'])

            ax.set_ylim(None, max_inty * 1.3)

            for i in idx:
                ax.text(df.loc[i, 'mz'], df.loc[i, 'intensity'] + max_inty / 20,
                        str(round(df.loc[i, 'mz'], ndigits=3)),
                        rotation=90, ha='center', va='bottom')

    def combineMZ(self, mz_tol_fixed=0.01, mz_tol_ppm=5):

        # create data frame with mz and intensity and get cluster ID
        df = _mzCluster(self.mz, self.intensity, mz_tol_fixed, mz_tol_ppm)

        # calculate sum of intensities in each cluster and take m/z from maximum as new m/z for cluster
        clust_grp = df.groupby(by='mz Cluster ID')

        new_inty = np.zeros(len(clust_grp))
        new_mz = np.zeros(len(clust_grp))

        for i, grp in enumerate(clust_grp):
            # sub_df = df_clust[df_clust['mz Cluster ID'] == id]
            sub_df = grp[1]

            new_inty[i] = sub_df['intensity'].sum()
            new_mz[i] = sub_df.loc[sub_df['intensity'].idxmax(), 'mz']

        return new_mz, new_inty


class MSFile:

    """
    Class to store retention times, total ion count, base peak count and a list of all MS1 spectra.
    """

    def __init__(self, filepath):

        self.filepath = filepath
        self.rt = None
        self.tic = None
        self.bpc = None
        self.MS1Data = None
        self.cutoffVal = None

    def getMS1ScanIndex(self, rt, rt_tol=0.05):
        """
        Find index of MS1 scans of given retention time.

        Parameters
        ----------
        rt : float
            Retention time in minutes
        rt_tol : float
            Tolerance of retention time. All indices of scans which retention times
            are in the range of rt +- tolerance will be returned.

        Returns
        -------
        List of indices in MS1Data
        """

        idx_list = np.where(abs(self.rt - rt) <= rt_tol)[0]
        return idx_list

    def getEIC(self, mz, mz_tolerance, fixed=True, plot=False):

        """
        Get the extracted ion counts of all MS1 scans.

        Parameters
        ----------
        mz : float
            m/z value to extract
        mz_tolerance : float
            m/z tolerance in ppm (if fixed is False) and Da (if fixed is True)
        fixed : bool, optional
            Set to True to use fixed tolerance in Da or False to use relative m/z tolerance in ppm
        plot : bool, optional
            Plot EIC

        Returns
        -------
        Array of extracted ion counts

        """

        if self.MS1Data is None:
            print('no MS data available!')

        else:

            if fixed:
                mz_min = mz - mz_tolerance
                mz_max = mz + mz_tolerance
            else:
                mz_min = round(mz - mz * mz_tolerance * 1e-6, ndigits=5)
                mz_max = round(mz + mz * mz_tolerance * 1e-6, ndigits=5)

            eic = np.zeros(len(self.MS1Data))

            for i, scan in enumerate(self.MS1Data):

                # get index of all m/z in given range
                mz_idx = [i for i, mz in enumerate(scan.mz) if (mz <= mz_max) and (mz >= mz_min)]

                if len(mz_idx) != 0:
                    # get intensity of m/z candidates
                    inty_array = [scan.intensity[i] for i in mz_idx]
                    eic[i] = max(inty_array)

            if plot:
                fig, ax = plt.subplots()
                ax.plot(self.rt, eic)
                ax.set_xlabel("retention time [min]")
                ax.set_ylabel("EIC")
                ax.set_title(f"m/z {mz_min} - {mz_max}")

                plt.show()

            return eic

    def getBPC(self, plot=False):
        """
        Calculates the base peak counts of intensities in all MS1 scans.

        Parameters
        ----------
        plot : bool, optional
            Create plot of BPC.

        """
        if self.MS1Data is not None:
            self.bpc = np.array([0.0] * len(self.MS1Data))

            for i, scan in enumerate(self.MS1Data):
                self.bpc[i] = max(scan.intensity) if len(scan.intensity) != 0 else 0

            if plot:
                fig, ax = plt.subplots()
                ax.plot(self.rt, self.bpc)
                ax.set_xlabel("retention time [min]")
                ax.set_ylabel("BPC")

                plt.show()

        else:
            print("No MS1 data available! Please read MS1 data.")

    def getBPCX(self, exclude_mz=None, ppm=10, smooth_sigma=2, plot=False):
        """
        Calculates the base peak chromatogram while excluding specified
        background m/z values, with optional Gaussian smoothing.

        Parameters
        ----------
        exclude_mz : array-like, optional
            List of m/z values to exclude from the BPC calculation.
        ppm : float, optional
            Mass tolerance for excluding background ions, in ppm.
        smooth_sigma : float or None, optional
            Standard deviation of the Gaussian kernel in number of data points.
            Set to None or 0 to disable smoothing.
        plot : bool, optional
            Create plot of BPCX.

        """
        if self.MS1Data is not None:

            if exclude_mz is None:
                exclude_mz = []

            exclude_mz = np.asarray(exclude_mz, dtype=float)

            self.bpcx = np.zeros(len(self.MS1Data))

            for i, scan in enumerate(self.MS1Data):

                if len(scan.intensity) == 0:
                    continue

                mz = np.asarray(scan.mz)
                intensity = np.asarray(scan.intensity)

                # Keep all peaks initially
                keep = np.ones(len(mz), dtype=bool)

                # Exclude background m/z values
                for bg_mz in exclude_mz:
                    tolerance = bg_mz * ppm * 1e-6

                    keep &= ~(
                            (mz >= bg_mz - tolerance) &
                            (mz <= bg_mz + tolerance)
                    )

                filtered_intensity = intensity[keep]

                self.bpcx[i] = (
                    np.max(filtered_intensity)
                    if len(filtered_intensity) > 0
                    else 0
                )

            # Gaussian smoothing
            if smooth_sigma is not None and smooth_sigma > 0:
                from scipy.ndimage import gaussian_filter1d
                self.bpcx_smoothed = gaussian_filter1d(
                    self.bpcx,
                    sigma=smooth_sigma
                )
            else:
                self.bpcx_smoothed = self.bpcx.copy()

            if plot:
                fig, ax = plt.subplots()

                ax.plot(self.rt, self.bpcx, alpha=0.4, label="BPCX")
                ax.plot(
                    self.rt,
                    self.bpcx_smoothed,
                    linewidth=2,
                    label="BPCX Gaussian smoothed"
                )

                ax.set_xlabel("retention time [min]")
                ax.set_ylabel("Intensity")
                ax.legend()

                plt.show()

        else:
            print("No MS1 data available! Please read MS1 data.")

    def integrate(self, idx=None, rtRange=None, plot=False):
        """
        Calculates the sum of intensities for every m/z value in defined MS1 scans. Either idx or rtRange must be given.
        Only possible for low resolution MS with low number of m/z values per scan (max 500).
        If number is higher, it will print a warning message.

        Parameters
        ----------
        idx : array
            Array of indices of MS1Data, so reference scans to integrate.
        rtRange : tuple
            Tupel (min, max) for setting retention time range.
        plot : bool
            Plot scan after integration

        Returns
        -------
        Dataframe with m/z values and summed intensities (absolute and relative).

        """

        # check, if MS1Data is available
        if self.MS1Data is None:
            print("No MS1 data available. Please parse all scans with readMSData method!")
            return None

        else:

            if idx is None:
                if rtRange is not None:
                    # find index of all times in retention time range
                    idx = [i for i in range(len(self.MS1Data)) if rtRange[0] < self.MS1Data[i].rt < rtRange[1]]

                else:
                    print("No index list or retention time range is set!"
                          "Please define an index list or a retention time range.")

            if len(idx) > 1:

                num_of_mz = [len(self.MS1Data[i].mz) for i in idx]

                if max(num_of_mz) > 500:
                    print("Maximum number %i of m/z values in scan is higher than recommended. "
                          "Integration might take more time." % max(num_of_mz))

                sum_intensity, sum_mz = _integrateScans(self.MS1Data[idx[0]].mz, self.MS1Data[idx[0]].intensity,
                                                        self.MS1Data[idx[1]].mz, self.MS1Data[idx[1]].intensity)

                for i, index in enumerate(idx[2:]):

                    sum_intensity, sum_mz = _integrateScans(sum_mz, sum_intensity,
                                                            self.MS1Data[index].mz, self.MS1Data[index].intensity)

            else:
                sum_mz = self.MS1Data[idx[0]].mz
                sum_intensity = self.MS1Data[idx[0]].intensity

            df = pd.DataFrame(
                {'mz': sum_mz, 'intensity': sum_intensity, 'rel. intensity': sum_intensity / max(sum_intensity) * 100})
            df.sort_values(by='mz', inplace=True)
            df.reset_index(drop=True, inplace=True)

            if plot:
                fig, ax = plt.subplots()
                ax.vlines(df['mz'], ymin=0, ymax=df['rel. intensity'], linewidth=0.5)
                ax.set_xlabel("m/z")
                ax.set_ylabel("relative intensity")

                plt.show()

            return df

    def intyCutoff(self, cutoff=500):
        """
        Removes all m/z in MS1 scans, which intensity is lower than the value of cut-off.
        Method will not work, if cutoff is lower than an older cut-off. Method will calculate BPC and TIC again.

        Parameters
        ----------
        cutoff : float (default 500)
            threshold for counts in MS1 scans

        Returns
        -------
        Overwrites MS1Data, BPC and TIC!

        """

        # check, if there is already an older cut-off value and new cut-off is lower
        if self.cutoffVal is not None and cutoff < self.cutoffVal:
            # print a warning, because MS scans are overwritten
            print('Old value {} is higher than new threshold {}.\n'
                  'MS1 scans will be overwritten, so that m/z values are already removed.\n'
                  'Read MS data again with readMSData(), than use the new threshold.'
                  .format(self.cutoffVal, cutoff))
        else:

            self.cutoffVal = cutoff

            for scan in self.MS1Data:

                scan.mz = scan.mz[scan.intensity > cutoff]
                scan.intensity = scan.intensity[scan.intensity > cutoff]

            # check, if scans are empty
            empty_check = sum([len(scan.intensity) == 0 for scan in self.MS1Data])

            if empty_check > 0:
                print('Warning: %i of %i scans are empty after intensity cut-off!' % (empty_check, len(self.MS1Data)))

            print("Refreshing TIC and BPC...")
            self.calcTIC()
            self.getBPC()

    def calcTIC(self):
        """
        Calculates TIC from MS1 scans with respect to the intensity cut-off of readMSData method.
        It overwrites self.tic!
        """

        if self.MS1Data is not None:
            self.tic = np.array([sum(scan.intensity) for scan in self.MS1Data])
        else:
            print("No MS1 data available. Please parse all scans with readMSData method!")

    def simplify_integrate(self, idx, mz_tol_fixed=0.01, mz_tol_ppm=5, plot=False):

        # check, if MS1Data is available
        if self.MS1Data is None:
            print("No MS1 data available. Please parse all scans with readMSData method!")
            return None

        else:

            # first cluster mz to process QToF data with higher resolution than Quadrupol data
            mz_list = [np.nan] * len(idx)
            inty_list = [np.nan] * len(idx)

            for i, ind in enumerate(idx):

                print("ind =", ind, "len(MS1Data) =", len(self.MS1Data))
                mz, inty = self.MS1Data[ind].combineMZ(mz_tol_fixed, mz_tol_ppm)

                mz_list[i] = mz
                inty_list[i] = inty

            # integrate processed MS scans
            if len(idx) > 1:

                sum_intensity, sum_mz = _integrateScans(mz_list[0], inty_list[0],
                                                        mz_list[1], inty_list[1])

                for i in range(2, len(mz_list)):
                    sum_intensity, sum_mz = _integrateScans(sum_mz, sum_intensity,
                                                            mz_list[i], inty_list[i])

            else:
                sum_mz = mz_list[0]
                sum_intensity = inty_list[0]

            df = pd.DataFrame(
                {'mz': sum_mz, 'intensity': sum_intensity, 'rel. intensity': sum_intensity / max(sum_intensity) * 100})
            df.sort_values(by='mz', inplace=True)
            df.reset_index(drop=True, inplace=True)

            if plot:
                fig, ax = plt.subplots()
                ax.vlines(df['mz'], ymin=0, ymax=df['rel. intensity'], linewidth=0.5)
                ax.set_xlabel("m/z")
                ax.set_ylabel("relative intensity")

                plt.show()

            return df


class QGDFile(MSFile):
    """
    Class for Shimadzu qgd files to store MS1 scans
    """

    def __init__(self, filepath):
        super().__init__(filepath)
        self.readMSData()

    def printContent(self):
        """
        Print all items that are stored in OLE file
        """
        with ole.OleFileIO(self.filepath) as f:
            for item in f.listdir():
                print(item)

    def readRT(self):
        """
        Parse retention times
        """
        with ole.OleFileIO(self.filepath) as f:

            if ['GCMS Raw Data', 'Retention Time'] in f.listdir():
                rt_stream = f.openstream(['GCMS Raw Data', 'Retention Time'])
                rt_bytes = rt_stream.read()

                # Retention Time
                n_points = int(len(rt_bytes) / 4)
                self.rt = np.zeros(n_points)

                for i in range(n_points):
                    self.rt[i] = round(
                        int.from_bytes(rt_bytes[4 * i:(4 * (i + 1))], byteorder='little', signed=False) / 60000,
                        ndigits=4)

            else:
                print("Cannot find Retention Time in GCMS Raw Data!")

    def readTIC(self):
        """
        Parse total ion counts from data file (TIC with no intensity cut-off)
        """
        with ole.OleFileIO(self.filepath) as f:

            if ['GCMS Raw Data', 'TIC Data'] in f.listdir():
                tic_stream = f.openstream(['GCMS Raw Data', 'TIC Data'])
                tic_bytes = tic_stream.read()

                n_points = int(len(tic_bytes) / 8)
                self.tic = np.zeros(n_points)

                for i in range(n_points):
                    self.tic[i] = int.from_bytes(tic_bytes[8 * i:(8 * (i + 1))], byteorder='little', signed=False)
            else:
                print("Cannot find TIC in GCMS Raw Data!")

    def readMSData(self):
        """
        Parse all MS1 spectra in qgd file. If rt is None, it will also parse it.
        """

        with ole.OleFileIO(self.filepath) as f:

            if ['GCMS Raw Data', 'MS Raw Data'] in f.listdir():

                ms_stream = f.openstream(['GCMS Raw Data', 'MS Raw Data'])
                self.MS1Data = []

                while True:

                    scan_byte = ms_stream.read(4)

                    if len(scan_byte) == 0:
                        break

                    # header
                    scanNumber = int.from_bytes(scan_byte, byteorder='little')
                    rt = round(int.from_bytes(ms_stream.read(4), byteorder='little') / 60000, ndigits=4)

                    if scanNumber % 10000 == 0:
                        print(scanNumber)

                    # skip
                    ms_stream.read(12)

                    n_bytes = int.from_bytes(ms_stream.read(2), byteorder='little')

                    # error in binary file
                    if n_bytes == 1:
                        n_bytes = 4

                    nval = int.from_bytes(ms_stream.read(2), byteorder='little')

                    # skip
                    ms_stream.read(8)

                    # spectra
                    mz = np.zeros(nval)
                    inty = np.zeros(nval)

                    for i in range(nval):
                        mz[i] = int.from_bytes(ms_stream.read(2), byteorder='little') / 20
                        inty[i] = int.from_bytes(ms_stream.read(n_bytes), byteorder='little')

                    temp = MS1Scan(rt=rt, mz=mz, intensity=inty)
                    temp.scanNumber = scanNumber

                    self.MS1Data.append(temp)

        # if rt is None, parse also RT
        if self.rt is None:
            self.readRT()

        # check, if number of RT is same as number of ms scans. If not, ms scans are not parsed right.
        if len(self.rt) != len(self.MS1Data):
            print('Warning: Number of time points %i not equal to number of MS scans %i!'
                  %(len(self.rt), len(self.MS1Data)))
        else:
            print('Number of MS1 scans: %i' % len(self.MS1Data))


class mzMLFile(MSFile):

    """
    Class to store spectra from mzML file. Additional to MS1 scans it can store MS2 scans,
    if they are available.
    """

    def __init__(self, filepath):
        super().__init__(filepath)

        self.readMSData()

    def readMSData(self):

        """
        Parse all scans in file and save them as class MS1Scan,
        contains retention time and arrays of m/z and intensity.
        """

        with mzml.MzML(self.filepath) as spectra:

            num = len(spectra)

            self.MS1Data = [0.0] * num
            self.rt = [np.nan] * num

            for i, spec in enumerate(spectra):

                if i % 10000 == 0:
                    print(i)

                rt = spec['scanList']['scan'][0]['scan start time']

                # Check if the scan is MS1 and the scan is not empty
                if spec['ms level'] == 1 and len(spec['m/z array']) != 0:

                    self.rt[i] = rt
                    mz = spec['m/z array']
                    intensity = spec['intensity array']

                    # remove mz where intensity is zero
                    mz = mz[intensity != 0]
                    intensity = intensity[intensity != 0]

                    temp = MS1Scan(mz=mz, intensity=intensity, rt=rt)
                    temp.scanNumber = i+1

                    self.MS1Data[i] = temp

            # drop nan
            self.MS1Data = [s for s in self.MS1Data if s != 0]
            self.rt = np.array([t for t in self.rt if not np.isnan(t)])

            print("Number of MS1 scans: %i" % len(self.MS1Data))

class GCDFile:

    def __init__(self, filepath):
        self.filepath = filepath

    def readRTParam(self):

        with ole.OleFileIO(self.filepath) as f:

            if ['GC Raw Data-1', 'Status'] in f.listdir():
                rt_param_stream = f.openstream(['GC Raw Data-1', 'Status'])

                time_interval_bytes = rt_param_stream.read(4)
                time_interval = int.from_bytes(time_interval_bytes, byteorder='little', signed=False)

                start_time_bytes = rt_param_stream.read(4)
                start_time = round(int.from_bytes(start_time_bytes, byteorder='little', signed=False) / 60000,
                                   ndigits=5)

                time_points_bytes = rt_param_stream.read(4)
                time_points = int.from_bytes(time_points_bytes, byteorder='little', signed=False)

                return time_interval, start_time, time_points
            else:
                print("Cannot find parameters for retention time in gcd file!")

    def readFID(self):

        # get rt param (interal in ms, start in minutes)
        time_interval, start_time, time_points = self.readRTParam()
        rt = np.linspace(start=start_time, stop=start_time + time_points * time_interval / 60000, num=time_points)

        with ole.OleFileIO(self.filepath) as f:

            if ['GC Raw Data-1', 'Intensity Data'] in f.listdir():

                fid_stream = f.openstream(['GC Raw Data-1', 'Intensity Data'])
                fid_bytes = fid_stream.read()
                n_points = int(len(fid_bytes) / 8)
                fid = np.zeros(n_points)

                # FID intensity is saved as signed float64 (double)
                for i in range(n_points):
                    fid[i] = struct.unpack('d', fid_bytes[8 * i:(8 * (i + 1))])[0]

                fid_df = pd.DataFrame({'RT': np.round(rt, decimals=5), 'FID': np.round(fid, decimals=0)})

                return fid_df

            else:
                print("Cannot find FID intensity in file!")