import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as spd


def _integrateScans(mz1, int1, mz2, int2, mzTol=0.01):

    if len(mz1) == 0:
        return int2, mz2

    elif len(mz2) == 0:
        return int1, mz1

    else:
        # Align two mass spectra
        diff = np.empty((len(mz2), len(mz1)))

        for i, mz in enumerate(mz2):
            diff[i] = np.absolute(mz1 - mz)

        row = len(mz1)
        minDiff = diff.min()
        intTable = np.array([int1, np.zeros(len(int1))])
        mz_array = np.copy(mz1)

        while minDiff < mzTol:
            # find the index of that match
            mc = divmod(np.argmin(diff), row)
            intTable[1, mc[1]] = int2[mc[0]]
            diff[:, mc[1]] = 1
            diff[mc[0]] = 1
            mz2[mc[0]] = -1
            minDiff = diff.min()

        unmatched = int2[mz2 != -1]

        if len(unmatched) != 0:
            t = np.row_stack((np.zeros(len(unmatched)), unmatched))
            intTable = np.column_stack((intTable, t))
            mz_array = np.concatenate([mz_array, mz2[mz2 != -1]])

        sum_array = intTable[0] + intTable[1]

        return sum_array, mz_array


def get_eics(msfile, mz_list, mz_tolerance=10, fixed=False, filepath=None):

    """

    Parameters
    ----------
    msfile : MSFile
        MSFile, that contains MS1Data of mzML or Shimadzu QGD
    mz_list : list
        Array with m/z values, which to extract from MS1 scans
    mz_tolerance : float, optional
        Tolerance of m/z in ppm (if fixed is False) or Da (if fixed is True)
    fixed : bool, optional
        Set to False, to use relative m/z tolerance in ppm or True for fixed tolerance in Da
    filepath : str, optional
        File path to export data frame as csv

    Returns
    -------
    Data frame with retention time and EICs as columns

    """

    eic = msfile.getEIC(mz_list[0], mz_tolerance, fixed)

    if eic is not None:

        if msfile.rt is None:
            print("No retention time. Please load it with readTIC() for QGD or getTIC() for mzML files.")

        else:

            df = pd.DataFrame({"RT (min)": msfile.rt, str(mz_list[0]): eic})

            if len(mz_list) > 1:
                for mz in mz_list[1:]:
                    eic = msfile.getEIC(mz, mz_tolerance, fixed)
                    df.insert(len(df.columns), str(mz), eic)

            if filepath is not None:
                df.to_csv(filepath, sep=",", index=False)

            return df


def calc_sample_rate(rt):

    """
    Calculate sample rate (Hz) as an average of time difference of the first 500 time points.

    Parameters
    ----------
    rt : list
        Array with retention time

    Returns
    -------
    Sample rate (Hz)

    """
    if len(rt) > 500:
        rt = rt[0:500]

    diff = np.empty(len(rt) - 1)

    for i in range(len(rt) - 1):
        diff[i] = rt[i + 1] - rt[i]

    sample_rate = round(1/(np.mean(diff)*60), ndigits=2)

    return sample_rate


def _shift_2drt(rt, intensity, sample_rate, shift, mod_time):

    # check, if shift is in mod_time range:
    if abs(shift) <= mod_time:

        # only negative shifts are allowed. If shift is positiv,
        if shift > 0:
            shift = shift - mod_time


        rt_points = int(sample_rate * -shift)

        # cut off time points at the beginning
        new_rt = rt[rt_points:]
        new_intensity = intensity[rt_points:]

        return new_rt, new_intensity

    else:
        print(f"Shift must be in range of -{mod_time} to {mod_time}!")


def _solvent_cutoff(rt, intensity, rt_cut):

    new_intensity = intensity.copy()
    new_intensity[rt < rt_cut] = 0

    return new_intensity


def intensity_matrix(rt, intensity, mod_time, sample_rate):

    """
    Cut the list of intensities into modulation and reshape to matrix.

    Parameters
    ----------
    rt : list
        Array of retention times
    intensity : list
        Array of intensities
    mod_time : float
        Modulation time in seconds
    sample_rate : float
        Sample rate (Hz)

    Returns
    -------
    Matrix of intensities for 2D plot
    """

    run_time = np.round(rt[-1], decimals=1)
    rt_offset = rt[0]

    dim_x = int(np.floor((run_time - rt_offset) * 60 / mod_time))
    dim_y = int(np.floor(mod_time * sample_rate))

    num_data = dim_x * dim_y

    # if less data points, then fill up with 0
    if num_data > len(intensity):
        intensity = np.append(intensity, np.zeros(num_data - len(intensity)))
    else:
        # cut off left over points
        intensity = intensity[0:num_data]

    raw_2d = np.reshape(intensity,
                        (dim_x, dim_y), order="C")

    return raw_2d


def shift_intensity_matrix(matrix, shift_time, sample_rate):

    """
    Shift 2D time axis as a roll up of intensities of all modulations

    Parameters
    ----------
    matrix : 2D array
        Intensity matrix
    shift_time : float
        Time in seconds to shift the 2D time axis. Negative shifts are also possible.
    sample_rate : float
        Sample rate (Hz)

    Returns
    -------
    2d array of shifted intensities

    """

    shift_idx = int(np.round(sample_rate * shift_time, decimals=0))

    matrix_shifted = [None] * len(matrix)

    for n in range(len(matrix)):
        matrix_shifted[n] = np.roll(matrix[n], shift_idx)

    return matrix_shifted


# helper functions for padding (flow mod GCxGC-QToF)
def _get_modulation(y, current_index, step0, stepmax, test_threshold=1.01):

    """
    Uses autocorrelation to find the start of the next modulation, which is the end of the current modulation.

    Parameters
    ----------
    y : list
        Array of intensities
    current_index : int
        Start index in array of retention times and intensities of current modulation
    step0 : int
        Offset index to get first translation for autocorrelation
    stepmax : int
        Maximum steps of translations for autocorrelation
    test_threshold : float
        To test, if there is a maximum in calculation of autocorrelation, the mean with and without the maximum is calculated.
        The test value is the division of mean with to mean without the maximum. Is must be lager than 1. If quotient is lower than
        the threshold, the modulation will not be corrected and function will take the expected modulation time.

    Returns
    -------
    sum array of autocorrelation, index of maximum and index of the detected end of current modulation

    """

    ac = np.zeros(stepmax)
    translations = range(step0, step0 + stepmax)
    # template is intensity array from current_index to step0. Therefore, the arrays do not overlap
    template = range(current_index, current_index + step0)

    for i in range(stepmax):
        # calculate sum
        ac[i] = sum([y[j] * y[j + translations[i]] for j in template])

    # get index of maximum and check, if it is a maximum
    max_index = np.argmax(ac)
    breakpoint = max_index + current_index + step0

    drop_list = [max_index]

    if max_index != 0:
        drop_list.append(max_index - 1)

    if max_index != len(ac) - 1:
        drop_list.append(max_index + 1)

    # remove maximum from list
    ac_wo_max = np.delete(ac, drop_list)

    # calculate the mean of both
    test_value = np.mean(ac) / np.mean(ac_wo_max)

    if test_value < test_threshold:
        # print("No correlation detected")
        breakpoint = int(stepmax / 2) + current_index + step0

    return ac, max_index, breakpoint  # end of current modulation and start index for next modulation


def _get_inty_matrix(inty, modtime, mod_tol, sample_rate, test_threshold):
    # define start and end of translation
    step0 = int(np.floor((modtime - mod_tol) * sample_rate))
    stepmax = int(np.ceil(2 * mod_tol * sample_rate))

    current_index = 0
    inty_list = []
    breaks = [0]

    while current_index < len(inty) - step0:

        try:
            ac, ac_max, mod_corrected = _get_modulation(inty, current_index, step0, stepmax, test_threshold)
            # plt.plot(ac)
            # plt.show()

        except IndexError:
            break

        breaks.append(mod_corrected)

        # copy current modulation in list
        inty_list.append(inty[current_index:mod_corrected])

        # set replace current index
        current_index = mod_corrected

    # append left over
    inty_list.append(inty[current_index:])

    return inty_list, breaks


def _padding(inty_list, dim_y):
    corrected = [0] * len(inty_list)

    for i, l in enumerate(inty_list):

        if len(l) < dim_y:
            corrected[i] = np.concatenate([l, np.zeros(dim_y - len(l))])

        elif len(l) > dim_y:
            corrected[i] = l[0:dim_y]

        else:
            corrected[i] = l

    return corrected


def intensity_matrix_padding(rt, intensity, mod_time, mod_tol, mod_offset, sample_rate, test_threshold):
    # flow mod starts with an offset of some seconds (offset in minutes!)
    offset_points = int(mod_offset * 60 * sample_rate)
    rt_corrected = rt[offset_points:]

    dim_y = int(np.floor(mod_time * sample_rate))

    inty_list, breaks = _get_inty_matrix(intensity, mod_time, mod_tol, sample_rate, test_threshold)

    corrected_inty = _padding(inty_list, dim_y)

    return rt_corrected, corrected_inty, breaks


def scan_annotation(mz, inty, threshold=20, min_mz_diff=5, mz_range=(None, None)):

    df = pd.DataFrame({"mz": mz, "intensity": inty})
    data = df[df["intensity"] > threshold]

    if mz_range[0] is not None:
        data = data[data['mz'] > mz_range[0]]

    if mz_range[1] is not None:
        data = data[data['mz'] < mz_range[1]]

    max_inty = df['intensity'].max()
    data = data.sort_values(by="intensity", ascending=False)

    text_idx = [data.index[0]]

    for i in range(1, len(data)):

        if all(abs(data.loc[data.index[i], 'mz'] - data.loc[text_idx, 'mz']) > min_mz_diff):
            text_idx.append(data.index[i])

    return text_idx, max_inty


def get_2d_rt(rt, rt_offset, mod_time, shift=0):

    rt_1d_part = np.floor((rt - rt_offset) * 60 / mod_time)
    rt_2d_part = (rt - rt_offset) * 60 / mod_time - rt_1d_part

    rt_1d = round(rt_1d_part * mod_time / 60 + rt_offset, ndigits=2)
    rt_2d = rt_2d_part * mod_time

    if shift != 0:
        rt_2d = rt_2d + shift

        if rt_2d >= mod_time:
            rt_2d = rt_2d - mod_time
        elif rt_2d < 0:
            rt_2d = rt_2d + mod_time

    return rt_1d, round(rt_2d, ndigits=2)


def _ms2str(mz, inty, threshold=5):

    scan_str = ''

    for i in range(len(mz)):

        if inty[i] > threshold:
            scan_str = scan_str + str(round(mz[i], ndigits=3)) + ":" + str(round(inty[i], ndigits=2)) + ' '

    # remove last spacer
    scan_str = scan_str[0:-1]

    return scan_str


def _str2ms(scan_str):

    # get mz and intensity
    peaks = scan_str.split(' ')

    peak_num = len(peaks)

    mz = np.zeros(peak_num)
    inty = np.zeros(peak_num)

    for i in range(peak_num):

        mz[i], inty[i] = peaks[i].split(':')

    return mz, inty


def _dict2str(dictionary, with_tabs=False):

    str_out = ''

    if with_tabs:

        for i, k in enumerate(dictionary.keys()):

            if i % 4 == 0 and i > 1:
                str_out = str_out + f"{k} : {dictionary[k]}\n"
            else:
                str_out = str_out + f"{k} : {dictionary[k]}    "

    else:
        for k in dictionary.keys():
            str_out = str_out + f"{k} : {dictionary[k]}\n"

    # remove last line break or tab
    str_out = str_out[0:-1]

    return str_out


# preprocess raw ms scan, combine similar m/z values
def _calc_tolerance(mz, tol_fixed, tol_ppm):
    return round(2 * (tol_fixed + mz * tol_ppm * 1e-6), ndigits=5)


def _mzCluster(mz_array, intensity_array, mz_tol_fixed=0.01, mz_tol_ppm=5):

    df = pd.DataFrame({'mz': mz_array, 'intensity': intensity_array})

    # sort dataframe ascending by mz and reset index
    df.sort_values(by='mz', inplace=True, ignore_index=True)

    # report in current frame mass difference between current frame and next frame (after sorting by mass)
    df['Mass Difference - to next Frame'] = df['mz'].shift(-1) - df['mz']

    # count total number of records
    totalRecCount = len(df)
    sectBegin = 0
    sectMin = 49

    # first comes coarse clustering by mz values. Starts looking at the mz difference.
    # If it is more than mz tolerance, a new cluster is started.
    currentClusterSectionID = 1

    df['mz Cluster Section ID'] = 0

    while totalRecCount - sectBegin >= sectMin:

        sectEnd = sectBegin + sectMin

        while sectEnd < totalRecCount - 1:

            # Should this be range of max mass in next feature??
            # If the mass difference to the next frame is greater than the average of the range of the largest mass
            # in the current group and the range of the smallest mass in the next group
            # it must be safe to end the mass section
            tolerance = (_calc_tolerance(df['mz'].iloc[sectEnd], mz_tol_fixed, mz_tol_ppm) + _calc_tolerance(
                df['mz'].iloc[sectEnd + 1], mz_tol_fixed, mz_tol_ppm)) / 2.0
            if df['Mass Difference - to next Frame'].iloc[sectEnd] > tolerance:
                break
            sectEnd += 1

        df.loc[sectBegin:sectEnd, 'mz Cluster Section ID'] = currentClusterSectionID
        currentClusterSectionID += 1
        sectBegin = sectEnd + 1

    df.loc[sectBegin:totalRecCount, 'mz Cluster Section ID'] = currentClusterSectionID

    # How many unique cluster sections (could use max in mz Cluster Section ID' column)
    mzClustSectCount = len(df['mz Cluster Section ID'].unique())

    # Within each section comes a hierachical clustering
    df['mz Cluster ID'] = 0

    # variable for keeping track of cluster ids
    currentMaxClusterID = 0

    for clustSect in range(1, mzClustSectCount + 1):

        # Get max error width in current cluster (based on max mass),
        # this will be used as the cut-off for hierachical clustering
        cluster = df[df['mz Cluster Section ID'] == clustSect]
        currentMaxMZError = _calc_tolerance(cluster['mz'].max(), mz_tol_fixed, mz_tol_ppm)

        # Get the masses in the current cluster into the correct format. Each mass is a single item list within a list
        vectorMZ = cluster['mz'].values.reshape((-1, 1))

        # Check if there is only 1 sublist within the list
        if len(vectorMZ) == 1:

            # If 1 item then give it next cluster id and move to next mass section
            df.loc[df['mz Cluster Section ID'] == clustSect, 'mz Cluster ID'] = currentMaxClusterID + 1
            currentMaxClusterID += 1
        else:
            # perform hierachical clustering

            # calculate distance between every mass in cluster section
            mzDistMatrix = spd.pdist(vectorMZ)

            # calculate linkage
            mzLinkage = hier.complete(mzDistMatrix)

            # return cluster id at each position to a numpy array, note the addition of the last recorded cluster name
            # prior to this (clustering starts at each time, hence currentMaxClusterID starting at 0
            mzClusters = hier.fcluster(mzLinkage, currentMaxMZError, 'distance') + currentMaxClusterID

            # Append these cluster ids to the data frame
            df.loc[df['mz Cluster Section ID'] == clustSect, 'mz Cluster ID'] = mzClusters

            # Increment currentMaxClusterID by number of unique clusters in current mass section
            currentMaxClusterID += len(set(mzClusters))

    # Remove columns that are not needed anymore
    df.drop('Mass Difference - to next Frame', inplace=True, axis=1)
    df.drop('mz Cluster Section ID', inplace=True, axis=1)

    # Cluster ids not in order, a bit of code to keep the ids in order with mz
    mc = df['mz Cluster ID'].to_numpy(copy=True)
    CurClus = 1
    lenTF = len(df)
    for i in range(lenTF - 1):
        if mc[i + 1] == mc[i]:
            mc[i] = CurClus
        else:
            mc[i] = CurClus
            CurClus += 1
    mc[lenTF - 1] = CurClus

    return df


def _combine_mz(df_clust):
    # combine mz values
    # clust_list = df_clust['mz Cluster ID'].unique()
    clust_grp = df_clust.groupby(by='mz Cluster ID')

    # inty_sum = [0] * len(clust_list)
    # mz_max = [0] * len(clust_list)
    inty_sum = np.zeros(len(clust_grp))
    mz_max = np.zeros(len(clust_grp))

    for i, grp in enumerate(clust_grp):
        # sub_df = df_clust[df_clust['mz Cluster ID'] == id]
        sub_df = grp[1]

        inty_sum[i] = sub_df['intensity'].sum()
        mz_max[i] = sub_df.loc[sub_df['intensity'].idxmax(), 'mz']

    return np.array(mz_max), np.array(inty_sum)


