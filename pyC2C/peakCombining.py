import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


from .helperFunctions import get_2d_rt, _ms2str


class ChromAnalyzer:

    def __init__(self, rt, intensity, msfile):

        self.chromatogram = pd.DataFrame({"rt": rt, "intensity": intensity})
        self.msfile = msfile

        self.picked_signals = None
        self.clustered_signals = None
        self.combined_signals = None
        self.results = None

        self.index_dict = dict()

    def pickSignals(self, ax, prominence=None, height=None, ylabel='counts', solvent_cut=None, marker_offset = 100):
        """
        Pick signals from a chromatogram and mark them in a plot. Returns a data frame with retention time, height
        and peak index for further integration

        Parameters
        ----------
        ax : matplotlib.pyplot.ax
            axis created by plt.subplots()
        prominence : float (optional)
            parameter for scipy.find_peaks. If None, it uses parameter height multiplied by 2.
        height : float (optional)
            minimum peak height. If None, it uses median of intensities in chromatogram multiplied by 1.5
        ylabel : str (default 'counts')
            Label of y-axis in plot
        solvent_cut : float (optional)
            retention time for solvent cut. Only peak after this time will be picked. Use None for no solvent cut.
        marker_offset : float (default 100)
            y-Offest for markers (use ax.get_ylim()[1] * 0.1 to scale offset with upper limits of y-axis)

    """
        rt_feat = self.chromatogram['rt'].values
        inty_feat = self.chromatogram['intensity'].values

        if height is None:
            height = np.median(inty_feat[inty_feat != 0]) * 1.5

        if prominence is None:
            prominence = height * 2

        peaks, info = find_peaks(inty_feat, prominence=prominence, height=height)

        p_rt = rt_feat[peaks]
        p_inty = inty_feat[peaks]

        self.picked_signals = pd.DataFrame({'Time': p_rt, 'Height': p_inty, 'Peak index': peaks})

        if solvent_cut is not None:
            self.picked_signals = self.picked_signals[self.picked_signals['Time'] >= solvent_cut]
            self.picked_signals.reset_index(drop=True, inplace=True)

        ax.plot(self.chromatogram['rt'], self.chromatogram['intensity'], linewidth=1)
        ax.set_title('Prominence %.1e  Height %.1e' % (prominence, height))
        ax.scatter(self.picked_signals['Time'], self.picked_signals['Height'] + marker_offset,
                   color='black', marker='v')
        ax.set_xlabel('retention time (min)')
        ax.ticklabel_format(style='sci', useMathText=True, scilimits=(0, 3))
        ax.set_ylabel(ylabel)

    def changeIndexToScan(self, rt_tol=0.001):
        """
        If signals were pick in FID, the peak index in data frame belongs to FID time points.
        This method changes the peak index to time points of MS data, which are needed for MS scan integration.

        Parameters
        ----------
        rt_tol : float (default 0.0005)
            Retention time tolerance for finding time points of FID in MS data.
            If there are more than one time point in given range,
            method will take MS scan index of the nearest time point.
        """

        new_index = [0] * len(self.picked_signals)

        for i in range(len(self.picked_signals)):

            fid_rt = self.picked_signals.iloc[i, 0]
            scans = self.msfile.getMS1ScanIndex(fid_rt, rt_tol)

            # ✅ FIX: handle empty scan list
            if len(scans) == 0:
                new_index[i] = -1
                continue

            new_index[i] = min(
                scans,
                key=lambda num: abs(fid_rt - self.msfile.MS1Data[num].rt)
            )

        self.picked_signals['Peak index'] = new_index

    def timeCluster(self, modtime, time_tol):

        """
        For 2D chromatograms to combine same features in different fractions. It calculates the time difference
        to the next feature. If the difference is in a tolerated range defined by the modulation time,
        the feature will be assigned to the same cluster.

        Parameters
        ----------
        modtime : float
            modulation time in seconds
        time_tol : float
            tolerance of modulation time in seconds
        """

        if self.picked_signals is None:
            print("No picked signals available. Did you run method pickSignals?")
            return None

        else:

            temp = self.picked_signals.copy()
            temp.sort_values(by='Time', inplace=True, ignore_index=True)

            rtTolMin = modtime - time_tol
            rtTolMax = modtime + time_tol

            temp['Time Cluster ID'] = 0
            curFeat = 1

            # get retention times
            rt = temp['Time'].values

            # get Cluster index
            idx = temp.index

            # go through all frames of Cluster until every frame has a Time Cluster ID
            while len(idx) > 0:

                # set current Time Cluster ID for first frame
                temp.loc[idx[0], 'Time Cluster ID'] = curFeat

                # get retention time of the current feature
                feat_rt = rt[idx[0]]
                num = len(idx)

                for i in range(1, num):

                    # calculate time difference between last feature and current frame
                    time_diff = (rt[idx[i]] - feat_rt) * 60

                    if time_diff < rtTolMax:

                        # check, if time difference is in tolerance range
                        if time_diff > rtTolMin:
                            # store signal index in array and check, if next signal is in tolerance range, too
                            td_in_range = [time_diff]
                            ind = 1
                            # difference cannot be smaller than rtTolMin, but we need to check, if it is bigger than max
                            while (time_diff < rtTolMax) and (i + ind < num):
                                # calculate time difference to next signal to see, if there are more than one in the
                                # tolerance range
                                time_diff = (rt[idx[i + ind]] - feat_rt) * 60
                                td_in_range.append(time_diff)
                                ind += 1
                            # add the signal that is nearest to modulation time to feature cluster
                            check_with_modtime = [abs(td - modtime) for td in td_in_range]
                            # find index of minimum
                            feat_ind = np.argmin(check_with_modtime)
                            temp.loc[idx[i + feat_ind], 'Time Cluster ID'] = curFeat

                            # update time
                            feat_rt = temp.loc[idx[i + feat_ind], 'Time']

                            # if time difference smaller than minimum of tolerance
                            # the difference to next frame will be calculated
                    else:
                        break

                # update index list
                idx = temp[temp['Time Cluster ID'] == 0].index
                curFeat += 1

            # re-order Time Cluster IDs
            self.clustered_signals = temp.sort_values(by=['Time Cluster ID', 'Time'], ignore_index=True)

    def combineSignals(self, modtime, shift=0):
        """
        Combines features in each time cluster, if intensities forms peak shape.
        Intensities are summed. Retention time is assign to feature of the highest intensity.

        Parameters
        ----------
        modtime : float
            modulation time in seconds
        shift : float (default 0)
            shift of y-axis for calculation of 2D retention times

        """

        self.combined_signals = self.clustered_signals.copy()
        time_group = self.combined_signals.groupby(by='Time Cluster ID')

        for group in time_group:
            new_inty, peak_dict = _peakCombining(inty=group[1]['Height'].values,
                                                peak_index=group[1]['Peak index'].values)

            self.combined_signals.loc[group[1].index, 'Height'] = new_inty

            # use dictionary for peak index of combined signals to integrate MS scans later
            self.index_dict[group[0]] = peak_dict

        # remove rows with zero
        self.combined_signals = self.combined_signals[self.combined_signals['Height'] != 0]
        self.combined_signals.reset_index(inplace=True, drop=True)

        # insert 1d and 2d rt
        n_rows = len(self.combined_signals)
        rt_1d = [0.0] * n_rows
        rt_2d = [0.0] * n_rows

        for i in range(n_rows):
            # subtract time offset
            rt_1d[i], rt_2d[i] = get_2d_rt(self.combined_signals.loc[i, 'Time'], self.chromatogram.loc[0, 'rt'],
                                           modtime, shift=shift)

        self.combined_signals.insert(1, 'RT 1d (min)', rt_1d)
        self.combined_signals.insert(2, 'RT 2d (s)', rt_2d)

    def plotCluster(self, xmin, xmax):
        """
        Plot time cluster in a retention time range

        Parameters
        ----------
        xmin : float
            begin of retention time interval in minutes
        xmax : float
            end of retention time interval in minutes

        """

        # check, if data frames of clustered and combined signals exist
        if any(df is None for df in [self.clustered_signals, self.combined_signals]):
            print("Need data frame of clustered and combined signals. Check, if one of them is None.")

        else:
            cmap = plt.colormaps.get_cmap('tab20')
            fig, ax = plt.subplots(2, figsize=(8, 4), sharex=True)

            # get subset of dfs and group by time cluster
            sub_clust = self.clustered_signals[self.clustered_signals['Time'] <= xmax]
            sub_clust = sub_clust[sub_clust['Time'] >= xmin]

            sub_combined = self.combined_signals[self.combined_signals['Time'] <= xmax]
            sub_combined = sub_combined[sub_combined['Time'] >= xmin]

            # get time cluster id list
            id_list = list(sub_clust['Time Cluster ID'].unique())

            for i, time_id in enumerate(id_list):
                df_clust = sub_clust[sub_clust['Time Cluster ID'] == time_id]
                df_comb = sub_combined[sub_combined['Time Cluster ID'] == time_id]

                ax[0].vlines(x=df_clust['Time'], ymin=0, ymax=df_clust['Height'], label=str(time_id), color=cmap(i))
                ax[1].vlines(x=df_comb['Time'], ymin=0, ymax=df_comb['Height'], label=str(time_id), color=cmap(i))

            ax[0].set_title('Time cluster')
            ax[1].set_title('Combined')

            ax[1].set_xlim(xmin, xmax)
            ax[1].set_xlabel('retention time (min)')
            ax[0].set_ylabel('intensities')
            ax[1].set_ylabel('summed intensities')

            ax[0].legend(ncol=2, loc=(1.05, 0))
            ax[0].ticklabel_format(style='sci', useMathText=True, axis='y', scilimits=(0, 3))
            ax[1].ticklabel_format(style='sci', useMathText=True, axis='y', scilimits=(0, 3))

            plt.tight_layout()
            plt.show()

    def integrateMSScans(self, with_neighbours=True, simplify=False, mz_tol_fixed=0.05, mz_tol_ppm=10):
        """
        Gets all MS scans with peak index dictionary, aligns m/z values and sums intensities to get MS scans for
        each peak in 2D data.

        Parameters
        ----------
        with_neighbours : bool (default True)
            Use also the MS scans before and after the MS scan of feature. Should reduce S/N in combined MS scans,
            because time point directly before and after the maximum are part of picked signal.
        simplify : bool (default False)
            If QToF data, we need to combined mz within given tolerance to get low resolution and to minimize length
            of m/z values for integration
        mz_tol_fixed : float (default 0.05)
            Fixed tolerance in Da for m/z combining of QToF data
        mz_tol_ppm : float (default 10)
            Relative tolerance in ppm for m/z combining of QToF data
        """

        # go through data frame and add integrated scan

        # insert new column for ms scans
        self.combined_signals['MS1'] = '0.0:0.0'

        # group by time cluster id
        cluster = self.combined_signals.groupby(by='Time Cluster ID')

        for n, clust in enumerate(cluster):

            if n % 20 == 0:
                print(n)

            tid, df = clust

            ms_array = []

            for i in range(len(df)):

                peak_idx = df['Peak index'].values[i]

                # default = NO CHANGE in pipeline behavior
                ms_string = "0.0:0.0"

                if peak_idx != -1:

                    idx_list = self.index_dict[tid][peak_idx]

                    if with_neighbours:
                        idx_list = np.concatenate([[p - 1, p, p + 1] for p in idx_list])

                    # clean indices (your logic preserved)
                    idx_list = [
                        int(x)
                        for x in idx_list
                        if isinstance(x, (int, np.integer)) and 0 <= x < len(self.msfile.MS1Data)
                    ]

                    if len(idx_list) > 0:

                        if simplify:
                            scan = self.msfile.simplify_integrate(
                                idx_list,
                                mz_tol_fixed=mz_tol_fixed,
                                mz_tol_ppm=mz_tol_ppm
                            )
                        else:
                            scan = self.msfile.integrate(idx_list)

                        ms_string = _ms2str(scan['mz'], scan['rel. intensity'])

                ms_array.append(ms_string)

            self.combined_signals.loc[df.index, 'MS1'] = ms_array

            # copy data frame of combined signals and remove peak index column (was needed only for ms scan integration)
        self.results = self.combined_signals.copy()
        self.results.drop(columns='Peak index', inplace=True)


def _peakCombining(inty, peak_index=None):

    inty = np.array(inty)

    # save index of combined signals for later integration of MS scans
    peak_dict = dict()

    if len(inty) > 1:

        # index of start in cluster
        featureClusterLowestRTIndex = 0

        # index of end of cluster
        featureClusterHighestRTIndex = inty.size - 1

        # Create array to record the frame category, e.g. PC for peak center and PF for Peak Frame
        mpc = np.empty_like(inty, dtype='|S2')
        mpc.fill('--')

        # Create an array to hold the intensities, each time an intensity is categorised it is remove thus leaving
        # the only uncategorized intensities so the highest can be selected
        intPeakCat = np.copy(inty)

        # Do while there are uncategorized frames within the feature group (i.e. 'Molecule Peak Category' mpc == "--")
        # for none nan intensities. np.count_nonzero works because we have a boolean mask whereby True becomes 1
        # and then this is counted as a non-zero
        while sum(mpc == b'--') > np.count_nonzero(inty == 0):

            # Make any categorized intensities to -1 in intPeakCat
            intPeakCat[np.where(mpc != b'--')[0]] = -1

            # Find the highest intensity uncategorized index and set as the peak centre
            peakCentreIndex = intPeakCat.argmax()

            # set peak start and finish index to be same as peak centre to begin with
            peakLowestRTIndex = peakHighestRTIndex = peakCentreIndex

            # Categorize the peak centre as PC
            mpc[peakCentreIndex] = 'PC'

            # Categorize left side of PC
            while featureClusterLowestRTIndex < peakLowestRTIndex:

                # The frame at (peakLowestRTIndex - 1) intensity is not 0
                if inty[peakLowestRTIndex - 1] == 0:
                    break

                    # Going to the left side, the intensity should fall down
                if inty[peakLowestRTIndex - 1] < inty[peakLowestRTIndex]:

                    peakLowestRTIndex -= 1
                    mpc[peakLowestRTIndex] = 'PF'

                # if not, check for intensity fluctuations, 10 percent is allowed
                elif inty[peakLowestRTIndex - 1] * 0.9 < inty[peakLowestRTIndex]:

                    # check, if next frame exits
                    if featureClusterLowestRTIndex < peakLowestRTIndex - 1:

                        # check, if next frame decreases
                        if inty[peakLowestRTIndex - 2] < inty[peakLowestRTIndex - 1]:
                            peakLowestRTIndex -= 1
                            mpc[peakLowestRTIndex] = 'PF'
                        else:
                            break

                    # no second frame on left side, so categorize both frames
                    else:
                        peakLowestRTIndex -= 1
                        mpc[peakLowestRTIndex] = 'PF'
                        # and break, because it is the end of left side
                        break
                else:
                    break

            # Categorize right side of PC
            while featureClusterHighestRTIndex > peakHighestRTIndex:

                # The frame at (peakHighestRTIndex + 1) intensity is not 0
                if inty[peakHighestRTIndex + 1] == 0:
                    break

                # Going to the right side, the intensity should fall down
                if inty[peakHighestRTIndex + 1] < inty[peakHighestRTIndex]:

                    peakHighestRTIndex += 1
                    mpc[peakHighestRTIndex] = 'PF'

                # if not, check for intensity fluctuations, 10 percent is allowed
                elif inty[peakHighestRTIndex + 1] * 0.90 < inty[peakHighestRTIndex]:

                    # check, if next frame exits
                    if featureClusterHighestRTIndex > peakHighestRTIndex + 1:

                        # check, if next frame decreases
                        if inty[peakHighestRTIndex + 2] < inty[peakHighestRTIndex + 1]:
                            peakHighestRTIndex += 1
                            mpc[peakHighestRTIndex] = 'PF'
                        else:
                            break

                    # no second frame on left side, so categorize both frames
                    else:
                        peakHighestRTIndex += 1
                        mpc[peakHighestRTIndex] = 'PF'
                        # and break, because it is the end of left side
                        break
                else:
                    break

            inty_sum = inty[peakLowestRTIndex:peakHighestRTIndex + 1].sum()

            # The 'PC' intensity is set to be the sum of all frames in the peak
            inty[peakCentreIndex] = inty_sum
            # add scan numbers of combined peaks to dictionary for integration of MS scans later
            peak_dict[peak_index[peakCentreIndex]] = peak_index[peakLowestRTIndex:peakHighestRTIndex + 1]

        # Set all non PC frames to 0 prior to return
        inty[np.where(mpc != b'PC')[0]] = 0

    else:
        # if there is only on feature in time cluster, there is no need to combined intensities.
        # Scan number of this feature is added to peak dictionary.
        peak_dict[peak_index[0]] = peak_index

    return inty, peak_dict







