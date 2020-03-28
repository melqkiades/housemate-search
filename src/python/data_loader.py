import math
import pandas
import time

import re
import requests
import seaborn
from daftlistings import Daft, RentType, University, Request
from geopy import Nominatim
from geopy.distance import distance
from geopy.exc import GeocoderQuotaExceeded
from matplotlib import pyplot
from requests import ConnectionError
from sklearn.preprocessing.data import MinMaxScaler
from tqdm import tqdm

from etl_utils import ETLUtils

DATA_FOLDER = '/Users/fpena/Projects/daft/data/'

RENT_TYPE_MAP = {
    'rooms': RentType.ROOMS_TO_SHARE,
    'places': RentType.ANY
}
RENT_TYPE = 'rooms'
# RENT_TYPE = 'places'
LOCATION =  'Ireland'

# CSV_FILE = '/tmp/daft_donnybrook.csv'
# CSV_LOCATION_FILE = '/tmp/daft_location_donnybrook.csv'
# CSV_FILE = '/tmp/daft_%s.csv' % time.strftime("%Y-%m-%d")
CSV_FILE = '%sdaft_%s_%s_%s.csv' % (DATA_FOLDER,
    LOCATION.lower().replace(' ', '_'), RENT_TYPE, time.strftime("%Y-%m-%d"))
# CSV_FILE = '/tmp/daft_donnybrook_%s.csv' % time.strftime("%Y-%m-%d")
# CSV_LOCATION_FILE = '/tmp/daft_location.csv'

TIME_PRICE_PER_HOUR = 20




def get_distance(location1, location2):
    dist = distance((location1.latitude, location1.longitude),
                    (location2.latitude, location2.longitude)).kilometers

    return dist


def get_location(address):
    # response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address=457+Collins+Avenue+West,+Dublin,+Ireland')
    #
    # resp_json_payload = response.json()
    #
    # print(resp_json_payload['results'][0]['geometry']['location'])

    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    # location = geolocator.geocode("175 5th Avenue NYC")
    # location = geolocator.geocode("457 Collins Avenue West, Dublin, Ireland")
    # location = geolocator.geocode("O'Brien Centre for Science, Dublin")
    location = geolocator.geocode(address)
    if not location:
        # print(location)
        pass
    else:
        pass
        # print(location.address)

    return location
    # print(location.address)
    # print((location.latitude, location.longitude))
    # print(location.raw)

    # loc1 = (41.49008, -71.312796)
    # loc2 = (41.499498, -81.695391)
    # dist = distance(loc1, loc2).kilometers
    # print dist

    # UCD Coordinates 53.30841565, -6.22380666284744


def fix_address(address):
    location = get_location(address)
    if location is not None:
        return location

    new_address = re.sub(r"Dublin [0-9]+", "Dublin", address)
    location = get_location(new_address)
    if location is not None:
        return location

    new_address = address.split(',', 1)[-1]
    location = get_location(new_address)
    if location is not None:
        return location
    return None
    # Remove the number from Dublin XX
    # Remove the text before the first comma


def get_listings():
    print('%s: Getting daft listings for %s' % (time.strftime("%Y/%m/%d-%H:%M:%S"), RENT_TYPE))

    daft = Daft()
    # daft.set_county("Dublin City")
    daft.set_county(LOCATION)
    daft.set_listing_type(RENT_TYPE_MAP[RENT_TYPE])
    # daft.set_min_price(0)
    # daft.set_max_price(900)
    # daft.set_address("Donnybrook")
    # daft.set_address("Ballsbridge")
    # daft.set_university(University.UCD)

    offset = 0
    all_listings = []

    while True:
        daft.set_offset(offset)
        listings = daft.search()
        all_listings.extend(listings)
        if not listings:
            break

        offset += 20

    print(len(all_listings))

    # except GeocoderQuotaExceeded:

    # geolocator = Nominatim(user_agent="specify_your_app_name_here")
    # ucd_location = geolocator.geocode("O'Brien Centre for Science, Dublin")
    # i = 0
    # for listing in all_listings:
    #     address = listing.formalised_address
    #     print(i)
    #     listing.location = fix_address(address)
    #     if listing.location:
    #         listing.latitude = listing.location.latitude
    #         listing.longitude = listing.location.longitude
    #         listing.distance_to_ucd = get_distance(ucd_location, listing.location)
    #     else:
    #         listing.latitude = None
    #         listing.longitude = None
    #         listing.distance_to_ucd = None
    #     i += 1

    return all_listings


def export_listings(listings):
    print("%s: Exporting listings" % (time.strftime("%Y/%m/%d-%H:%M:%S")))

    records = []
    ucd_coordinates = (53.30841565, -6.22380666284744)
    
    for listing in tqdm(listings):
        record = {}
        # 'search_type': listing.search_type,
        # 'agent_id': listing.agent_id,
        try_attempts = 0
        while try_attempts >= 0:
            try:
                record['id'] = listing.id.encode('utf-8').strip() if listing.id else None
                try_attempts = -1
            except ConnectionError:
                try_attempts += 1
                time.sleep(3)
                print('Retrying. Attempt number %d' % try_attempts)
            except KeyError:
                print('KeyError found. Continuing with the next record')
                continue

        # record['price'] = listing.price.encode('utf-8').strip() if listing.price else None


        # if listing.price_change is not None:
        #     record['price_change'] = listing.price_change,
        # 'viewings': listing.upcoming_viewings,
        # 'facilities': listing.facilities,
        # 'overviews': listing.overviews,
        record['formalised_address'] =\
            listing.formalised_address.encode('utf-8').strip() if listing.formalised_address else None
        record['address_line_1'] =\
            listing.address_line_1.encode('utf-8').strip() if listing.address_line_1 else None
        record['county'] = listing.county.encode('utf-8').strip() if listing.county else None
        # 'listing_image': listing.images,
        # 'listing_hires_image': listing.hires_images,
        # 'agent': listing.agent,
        # 'agent_url': listing.agent_url,
        # 'contact_number': listing.contact_number,
        record['daft_link'] = listing.daft_link.encode('utf-8').strip() if listing.daft_link else None
        record['latitude'] = obtain_field(listing, 'latitude')
        record['longitude'] = obtain_field(listing, 'longitude')
        record['distance_to_ucd'] = distance(
            ucd_coordinates, (record['latitude'], record['longitude'])).kilometers if record['latitude'] != 'nan' else float('nan')
        # record['city_center_distance'] = listing.city_center_distance
        record['owner_occupied'] = obtain_field(listing, 'owner_occupied').replace('"', '')
        record['property_type'] = obtain_field(listing, 'property_type').replace('"', '')
        record['room_type'] = obtain_field(listing, 'room_type').replace('"', '')
        record['people_living_there'] = obtain_closing_field(listing, 'people_living_there')
        record['published_date'] = obtain_field(listing, 'published_date').replace('"', '')
        record['property_category'] = obtain_field(listing, 'property_category').replace('"', '')
        record['area'] = obtain_field(listing, 'area').replace('"', '')
        record['ensuite_only'] = obtain_field(listing, 'ensuite_only').replace('"', '')
        record['acc_couples'] = obtain_field(listing, 'acc_couples').replace('"', '')
        record['available_for'] = obtain_field(listing, 'available_for').replace('"', '')
        record['gender'] = obtain_field(listing, 'gender').replace('"', '')
        record['seller_type'] = obtain_field(listing, 'seller_type').replace('"', '')

        # price = float(obtain_field(listing, 'price').replace('"', ''))
        # price_frequency = obtain_field(listing, 'price_frequency').replace('"', '')
        # record['price'] = transform_price_to_monthly(price, price_frequency)
        try:
            record['price'] = transform_string_price(listing.price)
        except ValueError as e:
            print(listing)
            print('ValueError found. Continuing with the next record')
            continue
        # record['price'] = listing.price.encode('utf-8')
        # 'shortcode': listing.shortcode,
        # record['date_insert_update'] =\
        #     listing.date_insert_update.encode('utf-8').strip() if listing.date_insert_update else None
        # 'views': listing.views,
        # 'description': listing.description,
        record['dwelling_type'] =\
            listing.dwelling_type.encode('utf-8').strip() if listing.dwelling_type else None
        # record['posted_since'] =\
        #     listing.posted_since.encode('utf-8').strip() if listing.posted_since else None
        record['num_bedrooms'] = listing.bedrooms
        record['num_bathrooms'] = listing.bathrooms
        record['agent'] = listing.agent.encode('utf-8').strip() if listing.agent else None
        record['agent_url'] = listing.agent_url
        record['contact_number'] = str(listing.contact_number)
        record['facilities'] = listing.facilities
        # 'commercial_area_size': listing.commercial_area_size
        record['contact_name'] = obtain_name(listing)
        # print(record)

        records.append(record)

    headers = sorted(records[0].keys())

    ETLUtils.save_csv_file(CSV_FILE, records, headers)

    # for record in records:
    #     print(record)
    #     ETLUtils.write_row_to_csv('/tmp/places.csv', record)


def calculate_scores(data_frame):
    data_frame = filter_data(data_frame)
    min_max_scaler = MinMaxScaler()
    data_frame['cycle_time'] = data_frame['distance_to_ucd'] * 6
    data_frame['price_score'] = 1 - min_max_scaler.fit_transform(
        data_frame[['price']])
    data_frame['cycle_time_score'] = 1 - min_max_scaler.fit_transform(
        data_frame[['cycle_time']])
    data_frame['money'] = \
        data_frame['price'] + data_frame['cycle_time'] * 22 * TIME_PRICE_PER_HOUR / 60
    data_frame['money_score'] = 1 - min_max_scaler.fit_transform(data_frame[['money']])
    data_frame['score'] =\
        data_frame['price_score'] + data_frame['cycle_time_score']
    data_frame['score'] = min_max_scaler.fit_transform(data_frame[['score']])
    data_frame['money_rank'] = data_frame['money'].rank(ascending=False) / (
        len(data_frame))

    pandas.options.display.max_colwidth = 200

    return data_frame


def print_ranking(data_frame):

    # print(data_frame[['formalised_address', 'score', 'price_score', 'price', 'daft_link']].sort_values('price_score').to_string())
    print(data_frame[[
        'formalised_address', 'price', 'cycle_time', 'money', 'money_rank',
        'daft_link', 'published_date']].sort_values('money').to_string())


def obtain_field(listing, coordinate_type):
    # pattern = re.compile(r'"%s":([\-]?[0-9]+\.[0-9]+),' % coordinate_type,
    #                               re.MULTILINE | re.DOTALL)
    pattern = re.compile(r'"%s":(.*?),' % coordinate_type,
                         re.MULTILINE | re.DOTALL)

    script = listing._ad_page_content.find("script", text=pattern)
    if script:
        match = pattern.search(script.text)
        if match:
            coordinate = match.group(1)
            return coordinate
    return 'nan'


def obtain_name(listing):
    div_container = listing._ad_page_content.find('div', {"id": "smi-negotiator-photo"})
    if div_container is None or div_container.find('h2') is None:
        return ''
    name = div_container.find('h2').next_element.encode('utf-8').strip()

    return name


# def transform_price_to_monthly(price, price_frequency):
#
#     if price_frequency == 'monthly':
#         return price
#     if price_frequency == 'weekly':
#         return price / 7 * 365 / 12
#     raise ValueError('Unregonized price frequency \'%s\'' % price_frequency)


def transform_string_price(string_price):

    # processed_price = string_price.replace('From ', '').split(' per ')
    processed_price = re.split(" per ", string_price.replace('From ', ''), flags=re.IGNORECASE)
    price = float(re.sub('[^0-9]', '', processed_price[0]))
    price_frequency = processed_price[1]

    if price_frequency == 'month':
        return price
    if price_frequency == 'week':
        return price / 7 * 365 / 12

    print('Unrecognized price frequency \'%s\'' % price_frequency)
    raise ValueError('Unrecognized price frequency \'%s\'' % price_frequency)


def obtain_closing_field(listing, coordinate_type):
    # pattern = re.compile(r'"%s":([\-]?[0-9]+\.[0-9]+),' % coordinate_type,
    #                               re.MULTILINE | re.DOTALL)
    pattern = re.compile(r'"%s":(.*?)}' % coordinate_type,
                         re.MULTILINE | re.DOTALL)

    script = listing._ad_page_content.find("script", text=pattern)
    if script:
        match = pattern.search(script.text)
        if match:
            coordinate = match.group(1)
            return coordinate
    return None


def count_nones(listings):

    count = 0

    for listing in listings:
        if listing.location is None:
            count += 1

    print('None count: %d' % count)
    return count


def get_new_entries():

    csv_file_1 = '%sdaft_dublin_city_rooms_2018-10-31.csv' % DATA_FOLDER
    csv_file_2 = '%sdaft_dublin_city_rooms_2018-10-30.csv' % DATA_FOLDER
    # csv_file_2 = '/tmp/daft_2018-10-19.csv'

    data_frame_1 = pandas.read_csv(csv_file_1)
    data_frame_2 = pandas.read_csv(csv_file_2)

    data_frame_1 = filter_data(data_frame_1)
    data_frame_1 = calculate_scores(data_frame_1)

    # new_entries_df =\
    #     pandas.concat([data_frame_1, data_frame_2]).drop_duplicates(keep=False)

    # new_entries_df = data_frame_1[~data_frame_2['id'].isin(data_frame_1['id'])]
    new_entries_df = data_frame_1[~data_frame_1['id'].isin(data_frame_2['id'])]

    # new_entries_df = filter_data(new_entries_df)

    # data_frame = pandas.DataFrame()

    # print(new_entries_df.to_string())
    print(new_entries_df[['formalised_address', 'price', 'cycle_time', 'money',
        'money_rank', 'daft_link', 'published_date']].sort_values('money').to_string())
    print('Total entries today: %d' % len(new_entries_df))
    print('Today\'s average: %f' % (new_entries_df['price'].mean()))


def filter_data(data_frame):

    data_frame = data_frame[data_frame['distance_to_ucd'] <= 4.0]
    # data_frame = data_frame[data_frame['dwelling_type'] == 'House to Rent']
    data_frame = data_frame[data_frame['owner_occupied'] == 'no']
    data_frame = data_frame[
        ~data_frame['room_type'].isin(['shared', 'twin'])]
    data_frame = data_frame[data_frame['gender'] != 'female']

    return data_frame


def process_data():

    file_path = '/Users/fpena/Stuff/House Search/Dublin/viewings-ucd.csv'
    data_frame = pandas.read_csv(file_path)
    print(data_frame.columns.values.tolist())
    print(data_frame.head())
    print(data_frame.describe())
    print(data_frame['Price'])

    price_scaler = MinMaxScaler()
    data_frame['Price Score'] = 1 - price_scaler.fit_transform(data_frame[['Price']])
    data_frame['Cycle Time Score'] = 1 - price_scaler.fit_transform(data_frame[['Cycle Time']])
    data_frame['Score'] = 0.5 * (data_frame['Price Score'] + data_frame['Cycle Time Score'])
    data_frame['Rank'] = data_frame['Score'].rank(ascending=True) / (len(data_frame))

    cycle_hour_cost = 30
    working_days_per_month = 22
    data_frame['Money Score'] =\
        data_frame['Price'] + data_frame['Cycle Time'] / 60 * cycle_hour_cost * working_days_per_month
    data_frame.rename(columns={'Cycle Time': 'Cycle'}, inplace=True)
    # print(data_frame['Price Score'])
    # print(data_frame[['Score', 'Rank']])
    # with pandas.option_context('display.max_rows', 500, 'display.max_columns', 10):
    #   print(data_frame[['Address', 'Price', 'Cycle', 'Rank', 'Score']].sort_values('Rank', ascending=False))
    # print(data_frame[['Address', 'Price', 'Cycle', 'Rank', 'Score', 'Money Score']].to_string())
    print(data_frame[['Address', 'Price', 'Cycle', 'Rank', 'Score', 'Money Score']].sort_values('Rank', ascending=False).to_string())

    # seaborn.(x='Price', y='Cycle Time', data_frame=data_frame)
    data_frame.plot.scatter(x='Price', y='Cycle')
    pyplot.savefig('/tmp/daft_scatter.pdf')
    pyplot.cla()
    pyplot.clf()

    data_frame.plot.scatter(x='Price Score', y='Cycle Time Score')
    pyplot.savefig('/tmp/daft_scatter_norm.pdf')
    pyplot.cla()
    pyplot.clf()

    seaborn.stripplot(x='Accommodation Type', y='Price', data=data_frame, jitter=True)
    pyplot.savefig('/tmp/daft_price.pdf')
    pyplot.cla()
    pyplot.clf()

    data_frame.plot.scatter(x='Housemates', y='Price')
    pyplot.savefig('/tmp/daft_scatter_price_housemates.pdf')
    pyplot.cla()
    pyplot.clf()

    data_frame.to_csv('/tmp/daft-houses-processed.csv')


def test():

    # url = 'http://www.daft.ie/dublin/house-share/dundrum/dundrum-road-dundrum-dublin-1045781/'
    # ad_page_content = Request(debug=False).get(url)
    # print(ad_page_content)

    string = '240 per week'
    print(transform_string_price(string))


def main():
    # process_data()
    # get_listings()
    # count_nones()
    # get_location("Hunters Crescent, Ballycullen, Dublin")
    listings = get_listings()
    export_listings(listings)
    # update_listings_location()
    # get_new_entries()
    # csv_file = '/tmp/daft_2018-10-19.csv'

    # csv_file = '/tmp/daft_dublin_city_rooms_2018-06-22.csv'
    # csv_file = '/Users/fpena/Projects/daft/data/daft_ireland_rooms_2019-07-28.csv'
    # csv_file = '/Users/fpena/Projects/daft/data/daft_ireland_places_2019-06-29.csv'
    # data_frame = pandas.read_csv(csv_file)
    # print_ranking(calculate_scores(data_frame))


# TODO: Get the new ads each time after running the query
# TODO: Optimize the get_listings() function by only taking the new IDs
# TODO: Read if description contains the word linkedin to include the profile


# Done
# TODO: Extract ID from daft_link
# TODO: Convert price string to monthly rent
# TODO: Extract distance to city center
# TODO: Obtain the price correctly when there is more than 1 room being advertised
# TODO: Separate the visualization part from the data preprocessing part
# TODO: Generate the scores and the rankings
# TODO: Include the distance to the city center
# TODO: Obtain the name of the contact person


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
