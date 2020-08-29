import json
import urllib
import pandas as pd
import numpy as np
import datetime
import pickle


class ESIOS(object):
    
    def __init__(self, token):
        """
        Class constructor
        :param token: string given by the ESIOS. You can get yours by sending an email to Consultas esios <consultasios@ree.es>
        """
        if token is None:
            print('The token is unique: You can get yours by sending an email to Consultas esios <consultasios@ree.es>')
        self.token = token

        # standard format of a date for a query        
        self.dateformat = '%Y-%m-%dT%H:%M:%S'
        
        # dictionary of available series
        self.__offer_indicators_list = list()
        self.__analysis_indicators_list = list()
        self.__indicators_name__ = dict()
        self.available_series = dict()

        print('Analyzing indicators...')
        self.available_series = self.get_indicators()

    def __get_headers__(self):
        """
        Prepares the CURL headers
        :return:
        """
        # Prepare the arguments of the call
        headers = dict()
        headers['Accept'] = 'application/json; application/vnd.esios-api-v1+json'
        headers['Content-Type'] = 'application/json'
        headers['Host'] = 'api.esios.ree.es'
        headers['Authorization'] = 'Token token=\"' + self.token + '\"'
        headers['Cookie'] = ''
        return headers
        
    def get_indicators(self):
        """
        Get the indicators and their name.
        The indicators are the indices assigned to the available data series
        :return:
        """
        fname = 'indicators.pickle'

        import os
        if os.path.exists(fname):
            # read the existing indicators file
            with open(fname, "rb") as input_file:
                all_indicators, self.__indicators_name__, self.__offer_indicators_list, self.__analysis_indicators_list = pickle.load(input_file)
        else:
            # create the indicators file querying the info to ESIOS
            """
            curl "https://api.esios.ree.es/offer_indicators" -X GET
            -H "Accept: application/json; application/vnd.esios-api-v1+json"
            -H "Content-Type: application/json"
            -H "Host: api.esios.ree.es"
            -H "Authorization: Token token=""
            -H "Cookie: "
            """
            all_indicators = dict()
            self.__indicators_name__ = dict()

            # This is how the URL is built
            url = 'https://api.esios.ree.es/offer_indicators'

            # Perform the call
            req = urllib.request.Request(url, headers=self.__get_headers__())
            with urllib.request.urlopen(req) as response:
                try:
                    json_data = response.read().decode('utf-8')
                except:
                    json_data = response.readall().decode('utf-8')

                result = json.loads(json_data)

            # fill the dictionary
            indicators = dict()
            self.__offer_indicators_list = list()
            for entry in result['indicators']:
                name = entry['name']
                id_ = entry['id']
                indicators[name] = id_
                self.__indicators_name__[id_] = name
                self.__offer_indicators_list.append([name, id_])

            all_indicators[u'indicadores de curvas de oferta'] = indicators

            """
            curl "https://api.esios.ree.es/indicators" -X GET
            -H "Accept: application/json; application/vnd.esios-api-v1+json"
            -H "Content-Type: application/json" -H "Host: api.esios.ree.es"
            -H "Authorization: Token token="
            -H "Cookie: "
            """
            url = 'https://api.esios.ree.es/indicators'

            req = urllib.request.Request(url, headers=self.__get_headers__())
            with urllib.request.urlopen(req) as response:
                try:
                    json_data = response.read().decode('utf-8')
                except:
                    json_data = response.readall().decode('utf-8')
                result = json.loads(json_data)

            # continue filling the dictionary
            indicators = dict()
            self.__analysis_indicators_list = list()
            for entry in result['indicators']:
                name = entry['name']
                id_ = entry['id']
                indicators[name] = id_
                self.__indicators_name__[id_] = name
                self.__analysis_indicators_list.append([name, id_])

            all_indicators[u'indicadores de análisis '] = indicators

            # save the indictators
            with open(fname, "wb") as output_file:
                dta = [all_indicators, self.__indicators_name__, self.__offer_indicators_list, self.__analysis_indicators_list]
                pickle.dump(dta, output_file)
        
        return all_indicators
        
    def get_names(self, indicators_list):
        """
        Get a list of names of the given indicator indices
        :param indicators_list:
        :return:
        """
        names = list()
        for i in indicators_list:
            names.append(self.__indicators_name__[i])
        
        return np.array(names, dtype=np.object)
        
    def save_indicators_table(self, fname='indicadores.xlsx'):
        """
        Saves the list of indicators in an excel file for easy consultation
        :param fname:
        :return:
        """
        data = self.__offer_indicators_list + self.__analysis_indicators_list
        
        df = pd.DataFrame(data=data, columns=['Nombre', 'Indicador'])
        
        df.to_excel(fname)

    def __get_query_json__(self, indicator, start, end):
        """
        Get a JSON series
        :param indicator: series indicator
        :param start: Start date
        :param end: End date
        :return:
        """
        url = 'https://api.esios.ree.es/indicators/' + indicator + '?start_date=' + start + '&end_date=' + end


        # Perform the call
        req = urllib.request.Request(url, headers=self.__get_headers__())
        with urllib.request.urlopen(req) as response:
            try:
                json_data = response.read().decode('utf-8')
            except:
                json_data = response.readall().decode('utf-8')
            result = json.loads(json_data)
            
        return result

    def get_data(self, indicator, start, end):
        """

        :param indicator: Series indicator
        :param start: Start date
        :param end: End date
        :return:
        """
        # check types: Pass to string for the url
        if type(start) is datetime.datetime:
            start = start.strftime(self.dateformat)
        
        if type(end) is datetime.datetime:
            end = end.strftime(self.dateformat)
        
        if type(indicator) is int:
            indicator = str(indicator)
            
        # get the JSON data
        result = self.__get_query_json__(indicator, start, end)

        # transform the data
        d = result['indicator']['values']  # dictionary of values
        
        if len(d) > 0:
            hdr = list(d[0].keys())  # headers    
            data = np.empty((len(d), len(hdr)), dtype=np.object)
            
            for i in range(len(d)):  # iterate the data entries
                for j in range(len(hdr)):  # iterate the headers
                    h = hdr[j]
                    val = d[i][h]
                    data[i, j] = val
                    
            df = pd.DataFrame(data=data, columns=hdr)  # make the DataFrame
            
            df['datetime_utc'] = pd.to_datetime(df['datetime'])  # convert to datetime
            
            df = df.set_index('datetime')  # Set the index column
            
#             del df.index.name  # to avoid the index name bullshit
            
            return df
        else:
            return None
    
    def get_multiple_series(self, indicators, start, end, country):
        """
        Get multiple series data
        :param indicators: List of indicators
        :param start: Start date
        :param end: End date
        :return:
        """
        df = None
        df_list = list()
        names = list()
        variable = 1
        geo_id = {'Spain':[3,8741], 'France':[2, 8741], 'Portugal':[1, 8741]}
        
        for i in range(len(indicators)):
            name = self.__indicators_name__[indicators[i]]
            names.append(name)
            print('Downloading ' + name)
            if i == 0:
                # Assign the first indicator
                df = self.get_data(indicators[i], start, end)
                df = df[df['geo_id'].isin(geo_id[country])]  # select by geography
                df.rename(columns={'value': name}, inplace=True)
                df_list.append(df)
            else:
                # merge the newer indicators
                df_new = self.get_data(indicators[i], start, end)
                if df_new is not None:
                    df_new = df_new[df_new['geo_id'].isin(geo_id[country])]  # select by geography
                    df_new.rename(columns={'value': name}, inplace=True)
                    df = df.join(df_new[name])
                df_list.append(df_new)
                
        return df, df_list, names

    def multiple_dfs(self, df_list, sheets, file_name, spaces):
        writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
        col = 0
        for dataframe in df_list:
            dataframe.to_excel(writer,sheet_name=sheets,startrow=0 , startcol=col)
            col = col + 20
        writer.save()
        
    def simple_df(self, df_list, sheets, filename):
        writer = pd.ExcelWriter(filename)
        df_list.to_excel(writer,sheets)
        writer.save()



