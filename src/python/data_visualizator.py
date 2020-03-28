import time

from datetime import datetime
import pandas
import seaborn
from matplotlib import pyplot

# CSV_FILE = '/tmp/daft_2018-10-19.csv'
from data_loader import filter_data

DATA_FOLDER = '/Users/fpena/Projects/daft/data/'
CSV_FILE = '%sdaft_dublin_city_rooms_2018-10-27.csv' % DATA_FOLDER


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


def price_average():
    csv_file = '%sdaft_dublin_city_rooms_2018-11-%02d.csv'
    days = range(12, 16)
    daily_mean = []

    for day in days:
        data_frame = pandas.read_csv(csv_file % (DATA_FOLDER, day))
        daily_mean.append(data_frame['price'].mean())
        print(data_frame['price'].mean())

    current_data_frame = pandas.read_csv('%sdaft_dublin_city_rooms_2018-11-15.csv' % DATA_FOLDER)
    current_data_frame['datetime'] = current_data_frame.apply(date_time_conversion, axis=1)
    # print(current_data_frame.groupby('published_date')['price'].mean())
    print(current_data_frame['datetime'])

    pyplot.plot(days, daily_mean)
    pyplot.savefig('/tmp/daily_average.pdf')
    pyplot.cla()
    pyplot.clf()

    current_data_frame = filter_data(current_data_frame)
    current_data_frame = current_data_frame[current_data_frame['datetime'] > '2018-10-21']
    current_data_frame.groupby('datetime')['price'].mean().plot()
    pyplot.savefig('/tmp/date_average.pdf')
    pyplot.cla()
    pyplot.clf()


def date_time_conversion(row):
    # print(row)
    date = row['published_date']
    return datetime.strptime(date, "%Y-%m-%d")


def main():
    # box_plot()
    price_average()
    # date_time_conversion('2017-02-14')


start = time.time()
main()
end = time.time()
total_time = end - start
print("Total time = %f seconds" % total_time)