import torch
import numpy as np
import random
import os
import pickle
from torch.utils.data import Dataset
import pandas as pd


def feature_add(data, weather, poi):
    weather_expanded = np.expand_dims(weather, axis=1)  # shape: (1, T, F_w)
    data = np.expand_dims(data, axis=-1)  # shape: (1, T, F_w)
    weather_broadcast = np.repeat(weather_expanded, data.shape[1], axis=1)  # shape: (N, T, F_w)
    print(data.shape, weather_broadcast.shape, poi.shape)
    output = np.concatenate([data, weather_broadcast, poi], axis=-1)  # shape: (N, T, F_s + F_w)
    return output


def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    holiday_init = 1
    for index in range(data.shape[0]):
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1 * (interval // 5)
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if day_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None:
        k = 1
    else:
        for j in holiday_list :
            holiday_data[j-1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data

def data_type_init(DATASET, args):
    if DATASET == 'METR_LA' or DATASET == 'SZ_TAXI' or DATASET == 'PEMS07M':
        data_type = 'speed'
    elif DATASET == 'PEMS08' or DATASET == 'PEMS04' or DATASET == 'PEMS03' or DATASET == 'PEMS07':
        data_type = 'flow'
    elif DATASET == 'NYC_BIKE' or DATASET == 'NYC_TAXI' or DATASET == 'CHI_TAXI' or DATASET == 'CHI_BIKE':
        data_type = 'demand'
    elif DATASET == 'Electricity':
        data_type = 'MTS'
    elif DATASET == 'NYC_CRIME' or DATASET == 'CHI_CRIME':
        data_type = 'crime'
    elif DATASET == 'BEIJING_SUBWAY':
        data_type = 'people flow'
    elif DATASET == 'chengdu_didi' or 'shenzhen_didi':
        data_type = 'index'
    else:
        raise ValueError

    args.data_type = data_type

# load dataset
def load_st_dataset(dataset, args):
    # 5 / 1 / 2017 - 8 / 31 / 2017 Monday
    if dataset == 'PEMS07':
        data_path = os.path.join('../data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        week_start = 1
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    elif dataset == 'NYC_Theft':
        data_path = os.path.join('../data/NYC_Theft/Crime_NYC_2016_2017.pkl')
        weather_path = os.path.join('../data/NYC_Theft/NYC.csv')
        Poi_path = os.path.join('../data/NYC_Theft/Poi_New York City_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]
        return data

    elif dataset == 'NYC_Assault':
        data_path = os.path.join('../data/NYC_Theft/Assault_NYC_2016_2017.pkl')
        weather_path = os.path.join('../data/NYC_Theft/NYC.csv')
        Poi_path = os.path.join('../data/NYC_Theft/Poi_New York City_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]
        return data

    elif dataset == 'CHI_Theft':
        data_path = os.path.join('../data/CHI_Theft/Crime_CHI_2016_2017.pkl')
        weather_path = os.path.join('../data/CHI_Theft/CHI.csv')
        Poi_path = os.path.join('../data/CHI_Theft/Poi_Chicago_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]
        return data
    elif dataset == 'CHI_Assault':
        data_path = os.path.join('../data/CHI_Theft/Assault_CHI_2016_2017.pkl')
        weather_path = os.path.join('../data/CHI_Theft/CHI.csv')
        Poi_path = os.path.join('../data/CHI_Theft/Poi_Chicago_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]
        return data
    elif dataset == 'DET_Theft':
        data_path = os.path.join('../data/DET_Theft/Crime_DET_2016_2017.pkl')
        weather_path = os.path.join('../data/DET_Theft/DET.csv')
        Poi_path = os.path.join('../data/DET_Theft/Poi_Detroit_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data
    elif dataset == 'DET_Assault':
        data_path = os.path.join('../data/DET_Theft/Assault_DET_2016_2017.pkl')
        weather_path = os.path.join('../data/DET_Theft/DET.csv')
        Poi_path = os.path.join('../data/DET_Theft/Poi_Detroit_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data
    elif dataset == 'PHI_Theft':
        data_path = os.path.join('../data/PHI_Theft/Crime_PHI_2016_2017.pkl')
        weather_path = os.path.join('../data/PHI_Theft/PHI.csv')
        Poi_path = os.path.join('../data/PHI_Theft/Poi_Philadelphia_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data
    elif dataset == 'PHI_Assault':
        data_path = os.path.join('../data/PHI_Theft/Assault_PHI_2016_2017.pkl')
        weather_path = os.path.join('../data/PHI_Theft/PHI.csv')
        Poi_path = os.path.join('../data/PHI_Theft/Poi_Philadelphia_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data == 0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data

    elif dataset == 'LOS_Theft':
        data_path = os.path.join('../data/LOS_Theft/Crime_Los Angeles_2016_2017.pkl')
        weather_path = os.path.join('../data/LOS_Theft/LOS.csv')
        Poi_path = os.path.join('../data/LOS_Theft/Poi_Los Angeles_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data
    elif dataset == 'LOS_Assault':
        data_path = os.path.join('../data/LOS_Theft/Assault_Los Angeles_2016_2017.pkl')
        weather_path = os.path.join('../data/LOS_Theft/LOS.csv')
        Poi_path = os.path.join('../data/LOS_Theft/Poi_Los Angeles_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data == 0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data


    elif dataset == 'BOS_Theft':
        data_path = os.path.join('../data/BOS_Theft/Crime_Boston_2016_2017.pkl')
        weather_path = os.path.join('../data/BOS_Theft/BOS.csv')
        Poi_path = os.path.join('../data/BOS_Theft/Poi_Boston_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data
    elif dataset == 'BOS_Theft':
        data_path = os.path.join('../data/BOS_Theft/Assault_Boston_2016_2017.pkl')
        weather_path = os.path.join('../data/BOS_Theft/BOS.csv')
        Poi_path = os.path.join('../data/BOS_Theft/Poi_Boston_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data == 0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data

    elif dataset == 'NEW_Theft':
        data_path = os.path.join('../data/NEW_Theft/Crime_New Orleans_2016_2017.pkl')
        weather_path = os.path.join('../data/NEW_Theft/NEW.csv')
        Poi_path = os.path.join('../data/NEW_Theft/Poi_New Orleans_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data==0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data
    elif dataset == 'NEW_Assault':
        data_path = os.path.join('../data/NEW_Theft/Assault_New Orleans_2016_2017.pkl')
        weather_path = os.path.join('../data/NEW_Theft/NEW.csv')
        Poi_path = os.path.join('../data/NEW_Theft/Poi_New Orleans_2016_2017.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        data = data.transpose(1, 0)
        with open(Poi_path, 'rb') as file:
            data_poi = pickle.load(file)
        weather_data = pd.read_csv(weather_path).iloc[:, 1:].values
        print(data.dtype, data[data == 0].shape)
        data = feature_add(data, weather_data, data_poi)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
              data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
        args.num_nodes = data.shape[1]

        return data
    else:
        raise ValueError

    args.num_nodes = data.shape[1]

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        # holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    elif len(data.shape) > 2:
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    else:
        raise ValueError

    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
    return data


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data


def split_data_by_ratio_eval(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    end = None if test_ratio == 1.0 else -365 + int(test_ratio * 365)

    test_data = data[-365 + int(val_ratio * 365):end]  #
    # test_data = data[-365: ]  #
    # val_data = data[-365 + int(0.3 * 365): -365 + int(0.3 * 365) + int(0.6 * 365)]
    val_data = data[-365 - int(0.2 * 365): -365]
    train_data = data[int(0.2 * 365): -365]  # -365   -365 - int(0.2 * 365)

    return train_data, val_data, test_data


def split_data_by_ratio_test(data, val_ratio,test_ratio):
    end = None if test_ratio == 1.0 else -365 + int(test_ratio * 365)
    test_data = data[-365 + int(val_ratio * 365):end]  #
    train_data = data[-365 - int(0.2 * 365): -300] # -365
    return train_data, test_data


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

def normalize_dataset(data, data_type, input_base_dim, normalize_type="meanstd"):
    if normalize_type == 'maxmin':
        data_ori = data[:, :, 0:input_base_dim]
        data_day = data[:, :, input_base_dim:input_base_dim+1]
        data_week = data[:, :, input_base_dim+1:input_base_dim+2]

        max_data = data_ori.max()
        min_data = data_ori.min()
        mean_day = data_day.mean()
        std_week = data_week.std()

        std_day = data_day.std()
        mean_week = data_week.mean()
        scaler_data = StandardScaler(min_data, max_data-min_data)
        scaler_day = StandardScaler(mean_day, std_day)
        scaler_week = StandardScaler(mean_week, std_week)
    else:
        data_ori = data[:, :, 0:input_base_dim]
        data_day = data[:, :, input_base_dim:input_base_dim+7]
        data_week = data[:, :, input_base_dim+7:input_base_dim+10]
        # data_holiday = data[:, :, 3:4]

        mean_data = data_ori.mean()
        std_data = data_ori.std()
        mean_day = data_day.mean()
        std_day = data_day.std()
        mean_week = data_week.mean()
        std_week = data_week.std()

        scaler_data = StandardScaler(mean_data, std_data)
        scaler_day = StandardScaler(mean_day, std_day)
        scaler_week = StandardScaler(mean_week, std_week)
    print('Normalize the dataset by Standard Normalization')
    return scaler_data, scaler_day, scaler_week, None

def define_dataloder(stage, val_ratio, test_ratio, args):
    x_trn_dict, y_trn_dict = {}, {}
    x_val_dict, y_val_dict = {}, {}
    x_tst_dict, y_tst_dict = {}, {}
    scaler_dict = {}

    datause_keys = args.dataset_use
    datatst_keys = args.dataset_test

    data_inlist = []
    if stage == 'eval':
        data_inlist.append(datatst_keys)
    # elif stage == 'test':
    #     data_inlist.append(datatst_keys)
    else:
        data_inlist = datause_keys

    for dataset_name in data_inlist:
        print(data_inlist, dataset_name, args.val_ratio, args.test_ratio)
        # print(sss)
        data = load_st_dataset(dataset_name, args)
        if args.mode == 'pretrain':
            data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
        else:
            data_train, data_val, data_test = split_data_by_ratio_eval(data, val_ratio, test_ratio)
            #####

            #########
        scaler_data, scaler_day, scaler_week, scaler_holiday = normalize_dataset(data_train, args.data_type, args.input_base_dim)
        print(data_train.shape, scaler_data.mean, scaler_data.std)
        x_tra, y_tra = Add_Window_Horizon(data_train, args.his, args.pred)
        x_val, y_val = Add_Window_Horizon(data_val, args.his, args.pred)
        x_test, y_test = Add_Window_Horizon(data_test, args.his, args.pred)

        if args.real_value == False:
            x_tra[..., :args.input_base_dim] = scaler_data.transform(x_tra[:, :, :, :args.input_base_dim])
            y_tra[..., :args.input_base_dim] = scaler_data.transform(y_tra[:, :, :, :args.input_base_dim])
            x_val[..., :args.input_base_dim] = scaler_data.transform(x_val[:, :, :, :args.input_base_dim])
            y_val[..., :args.input_base_dim] = scaler_data.transform(y_val[:, :, :, :args.input_base_dim])
            x_test[..., :args.input_base_dim] = scaler_data.transform(x_test[:, :, :, :args.input_base_dim])
            y_test[..., :args.input_base_dim] = scaler_data.transform(y_test[:, :, :, :args.input_base_dim])
        x_tra, y_tra = torch.FloatTensor(x_tra), torch.FloatTensor(y_tra)
        x_val, y_val = torch.FloatTensor(x_val), torch.FloatTensor(y_val)
        x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)

        x_trn_dict[dataset_name], y_trn_dict[dataset_name] = x_tra, y_tra
        x_val_dict[dataset_name], y_val_dict[dataset_name] = x_val, y_val
        x_tst_dict[dataset_name], y_tst_dict[dataset_name] = x_test, y_test
        scaler_dict[dataset_name] = scaler_data

    return x_trn_dict, y_trn_dict, x_val_dict, y_val_dict, x_tst_dict, y_tst_dict, scaler_dict


def define_dataloder_test(stage, val_ratio, test_ratio, args):
    x_tst_dict, y_tst_dict = {}, {}
    scaler_dict = {}

    datause_keys = args.dataset_use
    datatst_keys = args.dataset_test

    data_inlist = []
    if stage == 'eval':
        data_inlist.append(datatst_keys)
    # elif stage == 'test':
    #     data_inlist.append(datatst_keys)
    else:
        data_inlist = datause_keys

    for dataset_name in data_inlist:
        print(data_inlist, dataset_name, args.val_ratio, args.test_ratio)
        # print(sss)
        data = load_st_dataset(dataset_name, args)
        data_train, data_test = split_data_by_ratio_test(data, val_ratio, test_ratio)

        scaler_data, scaler_day, scaler_week, scaler_holiday = normalize_dataset(data_test, args.data_type, args.input_base_dim)
        print(data_train.shape, scaler_data.mean, scaler_data.std)

        x_test, y_test = Add_Window_Horizon(data_test, args.his, args.pred)

        if args.real_value == False:
            x_test[..., :args.input_base_dim] = scaler_data.transform(x_test[:, :, :, :args.input_base_dim])
            y_test[..., :args.input_base_dim] = scaler_data.transform(y_test[:, :, :, :args.input_base_dim])

        x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)

        x_tst_dict[dataset_name], y_tst_dict[dataset_name] = x_test, y_test
        scaler_dict[dataset_name] = scaler_data

    return x_tst_dict, y_tst_dict, scaler_dict


def get_pretrain_task_batch(args, x_list, y_list):

    select_dataset = random.choice(args.dataset_use)
    print(args.dataset_use, select_dataset, 'xxxx')
    batch_size = args.batch_size
    len_dataset = x_list[select_dataset].shape[0]

    batch_list_x = []
    batch_list_y = []
    permutation = np.random.permutation(len_dataset)
    for index in range(0, len_dataset, batch_size):
        start = index
        end = min(index + batch_size, len_dataset)
        indices = permutation[start:end]
        x_data = x_list[select_dataset][indices.copy()]
        y_data = y_list[select_dataset][indices.copy()]
        batch_list_x.append(x_data)
        batch_list_y.append(y_data)
    train_len = len(batch_list_x)
    return batch_list_x, batch_list_y, select_dataset, train_len

def get_val_tst_dataloader(X, Y, args, shuffle):
    X, Y = X[args.dataset_test], Y[args.dataset_test]
    data = torch.utils.data.TensorDataset(X, Y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, drop_last=False)
    return data_loader

