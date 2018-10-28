import time

import pandas
import seaborn
from matplotlib import pyplot

# CSV_FILE = '/tmp/daft_2018-10-19.csv'
CSV_FILE = '/tmp/daft_dublin_city_rooms_2018-10-27.csv'


def box_plot():

    data_frame = pandas.read_csv(CSV_FILE)

    data_frame = data_frame[data_frame['distance_to_ucd'] <= 4.0]
    data_frame = data_frame[data_frame['owner_occupied'] == 'no']
    data_frame = data_frame[~data_frame['room_type'].isin(['shared', 'twin'])]
    data_frame = data_frame[data_frame['gender'] != 'female']

    print(data_frame.describe())
    print(data_frame['price'])
    seaborn.boxplot(x='property_type', y='price', data=data_frame)
    # seaborn.boxplot(x='owner_occupied', y='price', data=data_frame)
    pyplot.savefig('/tmp/daft_box_plot.pdf')
    # pyplot.savefig('/tmp/daft_limerick_box_plot.pdf')
    pyplot.cla()
    pyplot.clf()

    seaborn.stripplot(x='property_type', y='price', data=data_frame,
                      jitter=True)
    pyplot.savefig('/tmp/daft_limerick_price.pdf')
    pyplot.cla()
    pyplot.clf()


def main():
    box_plot()


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)