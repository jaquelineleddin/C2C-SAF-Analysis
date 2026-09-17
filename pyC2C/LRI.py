class LinearRetentionIndex:

    def __init__(self):
        # key is carbon number and item is retention time of n-paraffin
        self.lri_dict = dict({7: 7.0,
                              8: 9.13,
                              9: 12.67,
                              10: 17.40,
                              11: 22.67,
                              12: 27.93,
                              13: 32.87,
                              14: 37.60,
                              15: 41.93})

    def print_LRIList(self):
        for cn in self.lri_dict.keys():
            print(f"{cn} : {self.lri_dict[cn]}")

    def add_to_LRIList(self, carbon_number, retention_time, print_list=True):
        self.lri_dict[carbon_number] = retention_time
        self.lri_dict = dict(sorted(self.lri_dict.items()))

        if print_list:
            self.print_LRIList()

    def calcLRI(self, rt_sample):

        # check, if rt is in range of dictionary
        cn_start = list(self.lri_dict.keys())[0]
        cn_end = list(self.lri_dict.keys())[-1]

        rt_start = self.lri_dict[cn_start]
        rt_end = self.lri_dict[cn_end]

        if rt_start < rt_sample < rt_end:

            # loop through dictionary to find range for calculation
            for k in self.lri_dict.keys():

                if rt_sample < self.lri_dict[k]:

                    rt_end = self.lri_dict[k]
                    cn_end = k
                    break
                else:
                    rt_start = self.lri_dict[k]
                    cn_start = k

            # calc lri
            lri = int(round( 100 * (cn_start + (rt_sample - rt_start) / (rt_end - rt_start) * (cn_end - cn_start)), ndigits=0))

        elif rt_start == rt_sample:
            lri = cn_start * 100

        elif rt_end == rt_sample:
            lri = cn_end * 100

        else:
            lri = -1

        return lri

    def getLRI_list(self, rt1_array):

        # create empty list for LRI
        lri_list = [0] * len(rt1_array)

        # loop through data frame and add LRI to list
        for i in range(len(rt1_array)):

            rt = rt1_array[i]
            lri_list[i] = self.calcLRI(rt)

        return lri_list
