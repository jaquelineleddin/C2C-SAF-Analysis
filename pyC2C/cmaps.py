from matplotlib.colors import LinearSegmentedColormap

blues = LinearSegmentedColormap.from_list('blues', ['#ffffff00', '#4361ee', '#3a0ca3'])
reds = LinearSegmentedColormap.from_list('reds', ['#ffffff00', '#c1121f', '#780000'])
greens = LinearSegmentedColormap.from_list('greens', ['#ffffff00', '#6ede8a', '#208b3a'])
oranges = LinearSegmentedColormap.from_list('oranges', ['#ffffff00', '#e8aa14', '#d36135'])
purples = LinearSegmentedColormap.from_list('purples', ['#ffffff00', '#9d4edd', '#5a189a'])

seb_map = LinearSegmentedColormap.from_list('seb map',
                                            ['#ffffff', '#457B9D', '#355e80', '#2e5375', '#1d3557',
                                             '#213e55', '#264b52', '#32664d', '#39ae29', '#40ff00',
                                             '#ffff00', '#ff0000', '#c00000'])

lc_map = LinearSegmentedColormap.from_list('lc map', ['#ffffffff', '#063FA1', '#128a82', '#71c582',
                                                      '#d5ff74', '#ffc61d', '#ff6224'])

