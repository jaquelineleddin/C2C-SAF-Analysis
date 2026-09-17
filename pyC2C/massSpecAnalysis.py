from itertools import product
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as spd


class MassSpectrumAnalyzer:
    # Monoisotopic masses
    element_masses = {
        'C': 12.000000,
        'H': 1.007825,
        'N': 14.003074,
        'O': 15.994915,
        'S': 31.972071,
        'P': 30.973762,
        '13C': 13.003355,
        'Cl': 34.968853,
        '37Cl': 36.965903
    }

    element_max = {
        'C': 40,
        'H': 80,
        'N': 0,
        'O': 0,
        'S': 0,
        'P': 0,
        '13C': 0,
        'Cl': 0,
        '37Cl': 0
    }

    # Adduct definitions -- include adduct formula for display
    adducts = {
        '[M+H]+': (1.007276, 1, 'H'),
        '[M-H]-': (-1.007276, -1, 'H'),
        '[M-H]+': (-1.008374, 1, 'H'),
        '[M+Na]+': (22.989218, 1, 'Na'),
        '[M+K]+': (38.963158, 1, 'K'),
        '[M+NH4]+': (18.033823, 1, 'NH4'),
        '[M+NO]+': (29.997989, 1, 'NO'),
        '[M+NO2]-': (45.993453, -1, 'NO2'),
        '[M+NO3]-': (61.988368, -1, 'NO3'),
        '[M]+': (-0.000549, 1, ''),
        '[M]-': (0.000549, -1, ''),
        '[M+H3O]+': (19.017841, 1, 'H3O'),
        '[M+OH]-': (17.003289, -1, 'OH'),
        '[M+O2]-': (31.990379, -1, 'O2'),
        '[M+HO2]-': (32.998204, -1, 'HO2'),
        '[M+O-H]-': (14.987639, -1, 'O-H'),

    }

    def __init__(self):
        self.print_elementMax()
        print('\n')
        self.print_adducts()

    def print_elementMax(self):
        print("Maximum numbers of elements: ")
        for el in self.element_max.keys():
            print("%s : %i" % (el, self.element_max[el]))

    def set_elementMax(self, element, new_max):
        if element in self.element_max.keys():
            self.element_max[element] = new_max
        else:
            print(f"Element {element} not in list! Append it with method add_element")

    def add_element(self, element, mass, max_number):
        self.element_masses[element] = mass
        self.element_max[element] = max_number

    def print_elementMasses(self):
        print("Masses of elements:")
        for el in self.element_masses.keys():
            print("%s : %i" % (el, self.element_masses[el]))

    def print_adducts(self, withMass=False):
        print("List of adducts:")
        if withMass:
            for adduct in self.adducts.keys():
                print("%s : %.6f, %i, %s" % (adduct, self.adducts[adduct][0],
                                             self.adducts[adduct][1], self.adducts[adduct][2]))
        else:
            for adduct in self.adducts.keys():
                print(adduct)

    def add_adducts(self, adduct_name, mz_change, charge, added_elements):
        self.adducts[adduct_name] = (mz_change, charge, added_elements)

    def generate_formulas(self, target_mz, ppm_tolerance=3.0, adduct='M+H'):
        if adduct not in self.adducts:
            print("Unknown adduct. Please use one of:", ', '.join(self.adducts.keys()))
            return []

        adduct_mass, charge, adduct_formula = self.adducts[adduct]
        results = []
        ranges = [range(self.element_max[el] + 1) for el in self.element_masses]
        elements = list(self.element_masses.keys())

        for counts in product(*ranges):
            if sum(counts) == 0:
                continue
            neutral_formula = ''.join(f"{el}{c}" if c > 0 else '' for el, c in zip(elements, counts))
            mass = sum(self.element_masses[el] * count for el, count in zip(elements, counts))

            # dbe
            dbe = 0
            for el, count in zip(elements, counts):

                if el in ('C', '13C'):
                    dbe += 2 * count
                elif el == 'N':
                    dbe += count
                elif el in ('Cl', '37Cl', 'H'):
                    dbe -= count

            dbe = (dbe + 2) / 2

            observed_mz = (mass + adduct_mass) / abs(charge) if charge != 0 else mass
            # Ion formula: neutral + adduct (for adducts such as '-H', you could expand subtraction logic if needed)
            if adduct.startswith('[M-') and adduct_formula:
                # For M-H, subtract H from formula string
                ion_formula = neutral_formula + "-" + adduct_formula
            elif adduct_formula:
                ion_formula = neutral_formula + "+" + adduct_formula
            else:
                ion_formula = neutral_formula

            ppm = (observed_mz - target_mz) / target_mz * 1e6
            if abs(ppm) <= ppm_tolerance:
                results.append((neutral_formula, ion_formula, observed_mz, ppm, dbe))
        return sorted(results, key=lambda x: abs(x[3]))


# Kendrick mass defect
def kendrick_mass_conversion(mz):
    kendrick_mass = mz * 14.00000 / 14.01565
    knm = np.round(kendrick_mass, decimals=0) # round to next nearest integer
    kmd = knm - kendrick_mass  # kmd is between -0.5 and 0.5

    return np.round(kendrick_mass, decimals=5), knm, np.round(kmd, decimals=5)


# preprocess raw ms scan, combine similar m/z values
def _calc_tolerance(mz, tol_fixed, tol_ppm):
    return round(2 * (tol_fixed + mz * tol_ppm * 1e-6), ndigits=5)


def mzCluster(df_in, mz_tol_fixed=0.01, mz_tol_ppm=5):
    df = df_in.copy()

    # sort dataframe ascending by mz and reset index
    df.sort_values(by='mz', inplace=True, ignore_index=True)

    # report in current frame mass difference between current frame and next frame (after sorting by mass)
    df['Mass Difference - to next Frame'] = df['mz'].shift(-1) - df['mz']

    # count total number of records
    totalRecCount = len(df)
    sectBegin = 0

    # first comes coarse clustering by mz values. Starts looking at the mz difference.
    # If it is more than mz tolerance, a new cluster is started.
    currentClusterSectionID = 1

    df['mz Cluster Section ID'] = 0

    while totalRecCount - sectBegin >= 1:

        sectEnd = sectBegin

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
    mc = df['mz Cluster ID'].values
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


def combine_mz(df_clust):
    # combine mz values
    clust_list = df_clust['mz Cluster ID'].unique()

    inty_sum = [0] * len(clust_list)
    mz_max = [0] * len(clust_list)

    for i, id in enumerate(clust_list):
        sub_df = df_clust[df_clust['mz Cluster ID'] == id]

        inty_sum[i] = sub_df['intensity'].sum()
        mz_max[i] = sub_df.loc[sub_df['intensity'].idxmax(), 'mz']

    df_combined = pd.DataFrame({'mz': mz_max, 'intensity': inty_sum})

    return df_combined


def ch2_cluster(mz_list, inty_list, mz_tol=0.01):
    ch2 = 14.01565
    mz_max = ch2 + mz_tol
    mz_min = ch2 - mz_tol

    df = pd.DataFrame({'mz': mz_list, 'intensity': inty_list, 'struct ID': 0})
    # sort data frame by ascending mz
    df.sort_values(by='mz', inplace=True, ignore_index=True)

    cur_id = 1
    idx_list = df[df['struct ID'] == 0].index

    # start from begin
    while len(idx_list) > 0:

        # take first mz without id (i is counter of idx_list)
        temp_mz = df.loc[idx_list[0], 'mz']
        df.at[idx_list[0], 'struct ID'] = cur_id
        num = len(idx_list)

        # loop through mz array
        for i in range(1, num):

            # mz difference to next signal
            diff = df.loc[idx_list[i], 'mz'] - temp_mz

            # if difference bigger than mz_max, no candidates for temp_mz
            if diff > mz_max:
                break
            else:
                # check, if we found a candidate
                if diff > mz_min:

                    candidates = [diff]
                    ind = 1
                    # difference cannot be smaller than mz_min, but we need to check, if it is bigger than mz_max
                    while (diff < mz_max) and (i + ind < num):
                        # calculate time difference to next signal to see, if there are more than one candidates
                        diff = df.loc[idx_list[i + ind], 'mz'] - temp_mz
                        candidates.append(diff)
                        ind += 1
                    # add mz with the smallest difference to ch2 to fine structure
                    check_with_ch2 = [abs(mz_diff - ch2) for mz_diff in candidates]
                    # find index of minimum
                    mz_ind = np.argmin(check_with_ch2)
                    df.loc[idx_list[i + mz_ind], 'struct ID'] = cur_id

                    # update temp_mz
                    temp_mz = df.loc[idx_list[i + mz_ind], 'mz']

                # if mz difference smaller than minimum of tolerance
                # the difference to next frame will be calculated
        # update index list
        idx_list = df[df['struct ID'] == 0].index
        cur_id += 1

    # sort ids
    df.sort_values(by='struct ID', inplace=True, ignore_index=True)

    return df

