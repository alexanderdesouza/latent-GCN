from __future__ import print_function
import pandas as pd
import numpy as np
import sys, os, argparse
from bs4 import BeautifulSoup


def preprocess(args):
    # loading the flights
    df = pd.read_csv(args.path + args.filename)

    #getting all the unique origins
    origins = df['Origin'].unique()
    print('total origins:', origins.shape[0])

    #getting all the unique destinations
    destinations = df['Dest'].unique()
    print('total destinations:', destinations.shape[0])

    #creating an exclude list for airports that only had one edge
    exclude_list = [o for o in destinations if o not in origins]

    #list of all airports
    total_airports = np.array(list(set(np.append(origins, destinations))))
    print('total airports:', total_airports.shape[0])


    with open(args.path + 'wikipedia_airport_data.txt', 'rb') as wad:
        data = wad.read().replace('\n', '')

        soup = BeautifulSoup(data, 'html.parser')

        headers = soup.find_all('th')
        columns = []
        for h in headers:
            columns += [h.text]

        airport_roles = {}

        rows = soup.find_all('tr')
        for r in rows:
            tds = r.find_all('td')
            if len(tds) == 0 or tds[2].text == '' :
                continue
            iata = tds[2].text
            role = tds[5].text.split(' ')[0]
            roles = ['P-N', 'P-S', 'P-M', 'P-L']
            if role in roles:
                role = min(args.max_labels - 1, roles.index(role))
            else:
                continue
            airport_roles[iata] = role

    df_airports_features = pd.read_csv(args.path + 'Airports_selection.csv')

    global graph
    graph = {}
    global airports
    airports = {}
    global counter
    counter = 0

    def get_id(airport):
        global airports
        global counter
        airport_id = airports.get(airport, -1)
        if airport_id == -1:
            airport_id = counter
            airports[airport] = airport_id
            counter += 1
        return airport_id


    if args.weight_mode == 'identity':
        #unique flights
        # get the different flights, unique and all
        flights_df = df[['Origin','Dest']]
        unique_flights_df = flights_df.drop_duplicates(['Origin', 'Dest'])
        del df
        for origin_airport, destination_airport in unique_flights_df.as_matrix():
            graph[origin_airport] = graph.get(origin_airport, {})
            graph[origin_airport][destination_airport] = graph[origin_airport].get(destination_airport, np.array([0])) + 1
    elif args.weight_mode == 'sum':
        #total amount of flights
        flights_df = df[['Origin','Dest']]
        del df
        for origin_airport, destination_airport in flights_df.as_matrix():
            graph[origin_airport] = graph.get(origin_airport, {})
            graph[origin_airport][destination_airport] = graph[origin_airport].get(destination_airport, np.array([0])) + 1

    elif args.weight_mode == 'edge_features':
        #get edge features
        #Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,UniqueCarrier,FlightNum,TailNum,ActualElapsedTime,CRSElapsedTime,
        #AirTime,ArrDelay,DepDelay,Origin,Dest,Distance,TaxiIn,TaxiOut,Cancelled,CancellationCode,Diverted,CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay

        del df['Year']
        # del df['Month']
        del df['DayofMonth']
        del df['DayOfWeek']
        del df['UniqueCarrier']
        del df['FlightNum']
        del df['TailNum']
        del df['CancellationCode']
        df_g = df.groupby(['Origin', 'Dest'])
        df_g_means = df_g.mean().mean()
        # print(df_g_means)
        print('grouped by origin, destination')
        del df
        for origin_airport, destination_airport in df_g.groups:
            group = df_g.get_group((origin_airport, destination_airport))
            features = np.array([])
            for month in range(1,13):
                group_month = group.loc[group['Month'] == month]
                group_month = group_month.fillna(0.0)
                del group_month['Origin']
                del group_month['Dest']
                del group_month['Month']
                group_mean = group_month.mean()
                group_mean = group_mean.fillna(0.0)
                flights = 0 + len(group_month)
                features = np.append(features, np.array([flights]))
                features = np.append(features, group_mean.as_matrix())
            if len(features) != 240:
                print(len(features))
                print(features)
                sys.exit()
            graph[origin_airport] = graph.get(origin_airport, {})
            graph[origin_airport][destination_airport] = features
    else:
        sys.exit('weight_mode not known, please select "identity", "sum" or "edge_features"')

    exclude_list += list(set(graph.keys()).difference(set(df_airports_features['LocationID'].unique())))
    exclude_list += list(set(graph.keys()).difference(set(airport_roles.keys())))

    def get_airport_info(airport_roles, df_airports_features, iata):
        """
        df_airports_features.columns
        [u'LocationID', u'Lon', u'Lat', u'FacilityTy', u'StateAbbv',
           u'StateName', u'County', u'City', u'FullName', u'FacilityUs',
           u'Elevation', u'Acres', u'hasNOTAMSe', u'ARFFClass', u'ARFFIndx',
           u'ARFFServ', u'isInternat', u'hasCustoms', u'isMilCivJo', u'hasMilLand',
           u'hasAirFram', u'hasPowerPl', u'BottledO2', u'BulkO2', u'hasTower',
           u'SingleEngC', u'MultiEngCo', u'JetEngCoun', u'HeloCount',
           u'GlidersCou', u'MilitaryCr', u'Ultralight', u'Commercial', u'Commuter',
           u'AirTaxi', u'Local', u'Itinerant', u'Military', u'Enplanemen',
           u'Passengers', u'Arrivals', u'Departures']
        """
        features = []

        row = df_airports_features.loc[df_airports_features['LocationID'] == iata]

        #different features
        features += [row['Lat'].as_matrix()[0]]
        features += [row['Lon'].as_matrix()[0]]
        features += [row['Elevation'].as_matrix()[0]]
        features += [row['Acres'].as_matrix()[0]]
        features += [1 if row['hasNOTAMSe'].as_matrix()[0] == 'Y' else 0]
        features += [1 if row['hasCustoms'].as_matrix()[0] == 'Y' else 0]
        features += [1 if row['isMilCivJo'].as_matrix()[0] == 'Y' else 0]
        features += [1 if row['hasMilLand'].as_matrix()[0] == 'Y' else 0]

        feature = row['hasAirFram'].as_matrix()[0]
        if feature == 'Major':
            features += [0, 1]
        elif feature == 'Minor':
            features += [1, 0]
        else: # 'None' en ''
            features += [0, 0]

        feature = row['hasPowerPl'].as_matrix()[0]
        if feature == 'Major':
            features += [0, 1]
        elif feature == 'Minor':
            features += [1, 0]
        else: # 'None' en ''
            features += [0, 0]

        feature = row['BottledO2'].as_matrix()[0]
        if feature == 'High':
            features += [0, 0, 1]
        elif feature == 'High/Low':
            features += [0, 1, 0]
        elif feature == 'Low':
            features += [1, 0, 0]
        else: # 'None' en ''
            features += [0, 0, 0]

        feature = row['BulkO2'].as_matrix()[0]
        if feature == 'High':
            features += [0, 0, 1]
        elif feature == 'High/Low':
            features += [0, 1, 0]
        elif feature == 'Low':
            features += [1, 0, 0]
        else: # 'None' en ''
            features += [0, 0, 0]

        features += [1 if row['hasTower'].as_matrix()[0] == 'Y' else 0]
        features += [row['SingleEngC'].as_matrix()[0]]
        features += [row['MultiEngCo'].as_matrix()[0]]
        features += [row['JetEngCoun'].as_matrix()[0]]
        features += [row['HeloCount'].as_matrix()[0]]
        features += [row['GlidersCou'].as_matrix()[0]]
        features += [row['MilitaryCr'].as_matrix()[0]]
        features += [row['Ultralight'].as_matrix()[0]]
        # features += [row['Commercial'].as_matrix()[0]]
        features += [row['Commuter'].as_matrix()[0]]
        # features += [row['AirTaxi'].as_matrix()[0]]
        features += [row['Local'].as_matrix()[0]]
        features += [row['Itinerant'].as_matrix()[0]]
        features += [row['Military'].as_matrix()[0]]
        # features += [row['Enplanemen'].as_matrix()[0]]
        # features += [row['Passengers'].as_matrix()[0]]
        # features += [row['Arrivals'].as_matrix()[0]]
        # features += [row['Departures'].as_matrix()[0]]

        #label as last column
        features += [airport_roles[iata]]

        return features

    print('exclude list:', exclude_list)
    print('total airports after exclusion:', len([item for item in graph.keys() if item not in exclude_list]))
    print([item for item in graph.keys() if item not in exclude_list])

    with open(args.path + args.edges_file, 'w') as t:
        with open(args.path + args.nodes_file, 'w') as f:
            idx = 0
            total = len([item for item in graph.keys() if item not in exclude_list])
            for origin, destinations in graph.items():
                if origin in exclude_list:
                    # print('skipped origin: ' + origin)
                    continue
                info = get_airport_info(airport_roles, df_airports_features, origin)
                origin_id = get_id(origin)

                for destination, features in destinations.items():
                    if destination in exclude_list:
                        # print('skipped destination: ' + destination)
                        continue
                    else:
                        destination_id = get_id(destination)
                        t.write(str(origin_id) + '\t' + str(destination_id) + '\t' + '\t'.join([str(feature) for feature in features]) + '\n')

                if args.add_i_matrix == 1:
                    identiy_matrix = np.zeros(total)
                    identiy_matrix[idx] = 1
                    f.write(str(origin_id) + '\t' + '\t'.join([str(int(i)) for i in identiy_matrix]) + '\t' + '\t'.join([str(i) for i in info]) + '\n')
                else:
                    f.write(str(origin_id) + '\t'+ '\t'.join([str(i) for i in info]) + '\n')
                idx += 1

def main():
    parser = argparse.ArgumentParser(description='Preprocess RITA dataset')
    parser.add_argument('--filename',       dest='filename',        type=str, default='2008.csv',     help='Source filename for the RITA flights')
    parser.add_argument('--add_identity',   dest='add_i_matrix',    type=int, default=0,              help='Add identity matrix to features')
    parser.add_argument('--edges_file',     dest='edges_file',      type=str, default='rita_tts_hard_lstm.cites',   help='Destination file for the edges')
    parser.add_argument('--nodes_file',     dest='nodes_file',      type=str, default='rita_tts_hard_lstm.content', help='Destination file for node content/features')
    parser.add_argument('--weight_mode',    dest='weight_mode',     type=str, default='identity',     help='Weight mode for the edges, if sum is selected the amount of edges between two nodes will be input sum or identity')
    parser.add_argument('--max_labels',     dest='max_labels',      type=int, default=100000,         help='Total amount of labels, everything above max_labels will be set to max_label min(max_label, label)')
    args = parser.parse_args()
    args.path = 'data/rita_tts_hard_lstm/'

    # args.edges_file = 'wm' + args.weight_mode + '_ml' + str(args.max_labels) + '_aim' + str(args.add_i_matrix) + args.edges_file
    # args.nodes_file = 'wm' + args.weight_mode + '_ml' + str(args.max_labels) + '_aim' + str(args.add_i_matrix) + args.nodes_file

    for arg, value in args.__dict__.items():
        print(arg, value)
    preprocess(args)


if __name__ == "__main__":
    main()
