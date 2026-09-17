import numpy as np


class CNum:

    def __init__(self):

        self.cnum_ranges = dict()

    def create_cnumItem(self, dbe, cnum_list, mz_list, c_ranges):

        cnum_item = dict()

        for i in range(len(cnum_list)):

            start_point = c_ranges[i][0]
            end_point = c_ranges[i][1]

            slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
            intercept = start_point[1] - slope * start_point[0]

            cnum_item[cnum_list[i]] = dict({'rt1': [start_point[0], end_point[0]],
                                            'rt2': [start_point[1], end_point[1]],
                                            'slope': round(slope, ndigits=3),
                                            'intercept': round(intercept, ndigits=3),
                                            'm/z': mz_list[i]})

        self.cnum_ranges[dbe] = cnum_item

    def get_cnum(self, results):

        results_dbe = results['DBE']
        results_rt1 = results['RT 1d (min)']
        results_rt2 = results['RT 2d (s)']

        # get possible dbes in dictionary
        key_list = list(self.cnum_ranges.keys())
        # create empty array for cnum
        cnum = [-1] * len(results_dbe)

        # loop through dbe array of results data frame
        for i in range(len(results_dbe)):

            temp_dbe = str(results_dbe[i])

            # check, if dbe is in dictionary
            if temp_dbe in key_list:
                # get ranges for each carbon number
                ranges = self.cnum_ranges[temp_dbe]

                # get rt1 und rt2
                temp_rt1 = results_rt1[i]
                temp_rt2 = results_rt2[i]

                # loop through ranges to find possible lines to check
                candidates_c = []
                candidates_rt2_diff = []

                for c in ranges.keys():

                    if ranges[c]['rt1'][0] <= temp_rt1 <= ranges[c]['rt1'][1]:

                        # calculate difference of rt2
                        candidates_rt2_diff.append(abs(temp_rt2 - (ranges[c]['slope'] * temp_rt1 + ranges[c]['intercept'])))
                        candidates_c.append(c)

                if len(candidates_c) == 1:
                    cnum[i] = candidates_c[0]

                elif len(candidates_c) > 1:
                    cnum[i] = candidates_c[np.argmin(candidates_rt2_diff)]

        return cnum



