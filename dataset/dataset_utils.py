from folktables import ACSDataSource, ACSIncome
import torch
from torch import clone
from torch.utils.data import Dataset, DataLoader, Subset
import random
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
logging.basicConfig(level=logging.DEBUG, format='%(filename)s - %(funcName)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
from folktables import ACSDataSource,  ACSIncome, ACSPublicCoverage, ACSMobility
from folktables import BasicProblem
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .folktable_utils import add_indicators, add_indicators_pubcov,add_indicators_mobility, preprocess


rac1p_vals = ['white','black','am_ind','alaska','am_alaska','asian','hawaiian','other','two_or_more']
relp_vals = ['reference', 'husband/wife','biologicalson','adoptedson','stepson','brother','father','grandchild','parentinlaw','soninlaw','other','roomer',
    'housemate','unmarried','foster','nonrelative','institutionalized','noninstitutionalized']
SCHL_vals = ['SCHL', 'schl_at_least_bachelor', 'schl_at_least_high_school_or_ged', 'schl_postgrad']
COW_vals = ['cow_employee_profit', 'cow_employee_nonprofit', 'cow_localgovernment', 'cow_stategovernment', 'cow_federalgovernment', 'cow_selfemployed_own',\
            'cow_selfemployed_incorporated', 'cow_family_business', 'cow_unemployed']
OCCP_vals = ['MGR1', 'MGR2', 'MGR3', 'MGR4', 'MGR5', 'BUS1', 'BUS2', 'BUS3', 'FIN1', 'FIN2', 
             'CMM1', 'CMM2', 'CMM3', 'ENG1', 'ENG2', 'ENG3', 'SCI1', 'SCI2', 'SCI3', 'SCI4', 
             'CMS1', 'LGL1', 'EDU1', 'EDU2', 'EDU3', 'EDU4', 'ENT1', 'ENT2', 'ENT3', 'ENT4', 
             'MED1', 'MED2', 'MED3', 'MED4', 'MED5', 'MED6', 'HLS1', 'PRT1', 'PRT2', 'PRT3',  
             'EAT1', 'EAT2', 'CLN1', 'PRS1', 'PRS2', 'PRS3', 'PRS4', 'SAL1', 'SAL2', 'SAL3', 
             'OFF1', 'OFF2', 'OFF3', 'OFF4', 'OFF5', 'OFF6', 'OFF7', 'OFF8', 'OFF9', 'OFF10', 
             'FFF1', 'FFF2', 'CON1', 'CON2', 'CON3', 'CON4', 'CON5', 'CON6', 'EXT1', 'EXT2', 
             'RPR1', 'RPR2', 'RPR3', 'RPR4', 'RPR5', 'RPR6', 'RPR7', 'PRD1', 'PRD2', 'PRD3', 
             'PRD4', 'PRD5', 'PRD6', 'PRD7', 'PRD8', 'PRD9', 'PRD10', 'PRD11', 'PRD12', 'PRD13', 
             'TRN1', 'TRN2', 'TRN3', 'TRN4', 'TRN5', 'TRN6', 'TRN7', 'TRN8', 'MIL1', "no1"]
Big_OCCP_vals = ['MGR', 'BUS', 'FIN', 'CMM', 'ENG', 'SCI', 'CMS', 'EDU', 'ENT', 'MED', 'HLS', 'PRT', 'EAT', 'CLN', 'PRS',\
                'SAL', 'OFF', 'FFF', 'CON', 'EXT', 'RPR', 'PRD', 'TRN', 'MIL', 'no']
Large_OCCP_vals = ['Lg', 'Semi_Lg', 'Non_Lg']
CIT_vals = ['us', 'pr', 'abroad', 'citizen', 'not']
ESR_vals = ['employed', 'partial_employed', 'unemployed', 'armed', 'partial_armed', 'no']


METHOD_NONE_DUMMIES = []
METHOD_NEED_PREPROCESS = ['svm', 'chi_doro', 'cvar_doro']

def adult_filter(data):
    """Mimic the filters in place for Adult data.
    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PINCP'] > 100]
    df = df[df['WKHP'] > 0]
    df = df[df['PWGTP'] >= 1]
    return df
def travel_time_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PWGTP'] >= 1]
    df = df[df['ESR'] == 1]
    return 

def employment_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['AGEP'] < 90]
    df = df[df['PWGTP'] >= 1]
    return 

def public_coverage_filter(data):
    """
    Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
    """
    df = data
    df = df[df['AGEP'] < 65]
    df = df[df['PINCP'] <= 30000]
    return df

def mobility_filter(data):
    df = data 
    df = df[df["AGEP"] > 18]
    df = df[df["AGEP"] < 35]
    return df

def get_USAccident(state, need_preprocess=True, root_dir='data/accident/US_Accidents_Dec21_updated.csv'):
    if not state in ['CA', 'TX', 'FL', 'OR', 'MN', 'VA', 'SC', 'NY', 'PA', 'NC', 'TN', 'MI', 'MO']:
        raise NotImplementedError(f"{state} is not supported in this dataset!")

    # raw_X = preprocess(root_dir)
    # 从本地读取数据
    raw_X = pd.read_csv("data/accident/US_Accidents_Dec21_updated_rawx.csv")
    # 保存raw_x到本地
    data = raw_X[raw_X["State"]==state]

    y_sample = data["Severity"]
    X_sample = data.drop(["Severity", "State", "Start_Lat", "Start_Lng"], axis=1).values

    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(X_sample)
        X_sample = scaler.transform(X_sample)

    # print(X_sample.shape, yq_sample.shape)
    return X_sample, y_sample.to_numpy().astype('int'), None

def get_taxi(city, need_preprocess,root_dir='./datasets/taxi/nyc_clean.csv'):

    remove_col_nyc = ['id', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count']
    remove_col_other = ['id', 'dist_meters', 'wait_sec', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

    try:
        df = pd.read_csv(root_dir)
    except:
        raise FileNotFoundError('File does not exist: {}'.format(root_dir))
        
    df = df[(df.trip_duration < 5900)]
    df = df[(df.pickup_longitude > -110)]
    df = df[(df.pickup_latitude < 50)]
    df.drop(['store_and_fwd_flag'], axis=1, inplace=True)
    df.drop(['vendor_id'], axis=1, inplace=True)
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df.drop(['dropoff_datetime'], axis=1, inplace=True) 
    df['month'] = df.pickup_datetime.dt.month
    df['week'] = df.pickup_datetime.dt.isocalendar().week
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    df['minute'] = df.pickup_datetime.dt.minute
    df['minute_oftheday'] = df['hour'] * 60 + df['minute']
    df.drop(['minute'], axis=1, inplace=True)


    df.drop(['pickup_datetime'], axis=1, inplace=True)

    def ft_haversine_distance(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371 #km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    df['distance'] = ft_haversine_distance(df['pickup_latitude'].values,
                                                    df['pickup_longitude'].values, 
                                                    df['dropoff_latitude'].values,
                                                    df['dropoff_longitude'].values)

    def ft_degree(lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371 #km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    df['direction'] = ft_degree(df['pickup_latitude'].values,
                                    df['pickup_longitude'].values,
                                    df['dropoff_latitude'].values,
                                    df['dropoff_longitude'].values)


    df = df[(df.distance < 200)]
    df['speed'] = df.distance / df.trip_duration
    df = df[(df.speed < 30)]

    df.drop(['speed'], axis=1, inplace=True)

    try:
        df = df.sample(n = 10000)
        test = test.sample(n = 10000)
    except:
        pass

    y_sample = df["trip_duration"].apply(lambda x: 0 if x < 900 else 1)
    df.drop(["trip_duration"], axis=1, inplace=True)
    if city == 'nyc':
        df.drop(remove_col_nyc, axis=1, inplace=True)
    else:
        df.drop(remove_col_other, axis = 1, inplace = True)

    X_sample = df.to_numpy()

    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(X_sample)
        X_sample = scaler.transform(X_sample)

    return X_sample, y_sample.to_numpy().astype('int'), None

def get_ACSIncome(state, year=2018, need_preprocess=True, root_dir = './datasets/acs'):
    task = ACSIncome
    data_source = ACSDataSource(root_dir=root_dir, survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=True)
    
    source_data = adult_filter(source_data)
    
    new_features = ['SEX', 'AGEP', 'WKHP',  'married', 'widowed','divorced','separated','never']+['relp_'+x for x in relp_vals]\
                    +['race_'+x for x in rac1p_vals] + COW_vals +['big_occp_'+x for x in Big_OCCP_vals]\
                    +SCHL_vals+['large_occp_'+x for x in Large_OCCP_vals]
    
    new_task = BasicProblem(new_features, task._target, task._target_transform,
                task._group, task._group_transform,
                preprocess = add_indicators, postprocess = task._postprocess)

    source_X_raw, source_y_raw, _ = new_task.df_to_numpy(source_data)
    
    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(source_X_raw[:,2:])
        source_X_raw[:,2:] = scaler.transform(source_X_raw[:,2:])
        
    return source_X_raw, source_y_raw.astype("int"), new_features

def get_ACSIncome_aug(state, year=2018, need_preprocess=True, root_dir = './datasets/acs'):
    """
    Add features to mitigate the concept drifts
    """
    task = ACSIncome
    data_source = ACSDataSource(root_dir=root_dir, survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=True)
    source_data = adult_filter(source_data)
    
    new_features = ['SEX', 'AGEP', 'WKHP',  'married', 'widowed','divorced','separated','never']+['relp_'+x for x in relp_vals]\
                    +['race_'+x for x in rac1p_vals] + COW_vals +['big_occp_'+x for x in Big_OCCP_vals]\
                    +SCHL_vals+['large_occp_'+x for x in Large_OCCP_vals]+['ENG']
    
    new_task = BasicProblem(new_features, task._target, task._target_transform,
                task._group, task._group_transform,
                preprocess = add_indicators, postprocess = task._postprocess)

    source_X_raw, source_y_raw, _ = new_task.df_to_numpy(source_data)
    
    index = np.where(source_X_raw[:,-1]>2)[0]
    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(source_X_raw[:,1:])
        source_X_raw[:,1:] = scaler.transform(source_X_raw[:,1:])

    return source_X_raw, source_y_raw.astype("int"), new_features

def get_ACSPubCov(state, year=2018, need_preprocess=True, root_dir = './datasets/acs'):
    task = ACSPublicCoverage
    data_source = ACSDataSource(root_dir=root_dir, survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=False)
    
    source_data = public_coverage_filter(source_data)
    rac1p_vals = ['white','black','am_ind','alaska','am_alaska','asian','hawaiian','other','two_or_more']
    
    new_features = ['SEX', 'AGEP', 'DIS', 'ESP', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE',
                    'DREM', 'PINCP', 'FER',  'married', 'widowed','divorced','separated','never']+['race_'+x for x in rac1p_vals]+SCHL_vals+['CIT_'+x for x in CIT_vals]\
                    +['ESR_'+x for x in ESR_vals]
    
    new_task = BasicProblem(new_features, task._target, task._target_transform,
                task._group, task._group_transform,
                preprocess = add_indicators_pubcov, postprocess = task._postprocess)

    source_X_raw, source_y_raw, _ = new_task.df_to_numpy(source_data)
    
    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(source_X_raw[:,2:])
        source_X_raw[:,2:] = scaler.transform(source_X_raw[:,2:])
        
    return source_X_raw, source_y_raw.astype("int"), new_features

def get_ACSMobility(state, year=2018, need_preprocess=True, root_dir = './datasets/acs'):
    task = ACSMobility
    data_source = ACSDataSource(root_dir=root_dir, survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=False)
    
    source_data = mobility_filter(source_data)
    rac1p_vals = ['white','black','am_ind','alaska','am_alaska','asian','hawaiian','other','two_or_more']
    relp_vals = ['reference', 'husband/wife','biologicalson','adoptedson','stepson','brother','father','grandchild','parentinlaw','soninlaw','other','roomer',
        'housemate','unmarried','foster','nonrelative','institutionalized','noninstitutionalized']
    new_features = ['SEX', 'AGEP', 'SCHL', 'DIS', 'ESP','CIT', 'MIL', 'ANC', 'NATIVITY',  'DEAR','DEYE','DREM',
                    'GCL',  'WKHP', 'JWMNP', 'PINCP', 'married', 'widowed','divorced','separated','never']\
                    +['race_'+x for x in rac1p_vals]+['relp_'+x for x in relp_vals]+COW_vals+['ESR_'+x for x in ESR_vals]
    
    new_task = BasicProblem(new_features, task._target, task._target_transform,
                task._group, task._group_transform,
                preprocess = add_indicators_mobility, postprocess = task._postprocess)

    source_X_raw, source_y_raw, _ = new_task.df_to_numpy(source_data)
    
    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(source_X_raw[:,2:])
        source_X_raw[:,2:] = scaler.transform(source_X_raw[:,2:])
        
    return source_X_raw, source_y_raw.astype("int"), new_features

def select_dataset_class(dataset_name):
    dataset_classes = {
        "acs": ACSRawDatasets,
       
    }
    if dataset_name in dataset_classes:
        return dataset_classes[dataset_name]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available datasets: {list(dataset_classes.keys())}")
class ACSRawDatasets:
    def __init__(self,task, src_region, tgt_region,state_names ):
        self.src_state_name = src_region
        self.tgt_state_name = tgt_region
        self.task = task
        
        get_task = {
            "income_noisy": get_ACSIncome,
            "income": get_ACSIncome,
            "pubcov": get_ACSPubCov,
            "mobility": get_ACSMobility,
            "accident":get_USAccident,
        }[self.task]
        self.all_data = dict() # key: state_name, value: (tensor(x),tensor(y))
        # self.state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
        #       'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
        #       'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
        #       'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
        #       'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
        self.state_list = state_names
        if self.task == 'accident':
            root_dir = 'data/accident/US_Accidents_Dec21_updated.csv'
        else:
            root_dir = 'data'
        for state_name in self.state_list:
            x, y, self.feature_names = get_task(state=state_name, root_dir=root_dir)
            # 如果数据集大与20000，随机采样20000个样本
            if x.shape[0] > 20000:
                indices = np.random.choice(x.shape[0], 20000, replace=False)
                x = x[indices]
                y = y[indices]
            self.all_data[state_name] = (torch.from_numpy(x).float(), torch.from_numpy(y).long())
        if self.task == "income_noisy":
            self.scaler_y = StandardScaler()
            # 用src_state_name的y值训练scaler
            self.scaler_y.fit(self.all_data[self.src_state_name][1].reshape(-1,1))
            for state_name in self.state_list:
                x, y = self.all_data[state_name]
                y = self.scaler_y.transform(y.reshape(-1,1))
                self.all_data[state_name] = (x, torch.from_numpy(y).float().to('cuda'))     
    def get_src_state_data(self):
        return self.src_state_name, self.all_data[self.src_state_name]
    def get_tgt_state_data(self):
        return self.tgt_state_name, self.all_data[self.tgt_state_name]
    def get_other_state_data(self):
        for state_name in self.all_data.keys():
            if state_name == self.src_state_name or state_name == self.tgt_state_name:
                continue
            yield state_name, self.all_data[state_name]
    def n_features(self):
        return self.all_data[self.src_state_name][0].shape[1]
    def n_labels(self):
        return len(np.unique(self.all_data[self.src_state_name][1]))
    def other_state_names(self):
        return [state_name for state_name in self.state_list if state_name != self.src_state_name and state_name != self.tgt_state_name]
    def region_dataset(self, state_name):
        return ACSDataset(*self.all_data[state_name], state_name)
    def all_other_data(self):
        all_x = torch.cat([self.all_data[state_name][0] for state_name in self.other_state_names()])
        all_y = torch.cat([self.all_data[state_name][1] for state_name in self.other_state_names()])
        return all_x, all_y
class ACSDataset(Dataset):
    def __init__(self, x, y, state_name=None):
        super(ACSDataset, self).__init__()
        self.state_name = state_name
        
        idx = random.sample(range(x.size(0)), x.size(0))
        self.x = x[idx].to('cuda')
        self.y = y[idx].to('cuda')
        
        self.ucb_next_idx = 0
        self.uniform_next_idx = 0
        self.target_next_idx = 0
        self.mmd_next_idx = 0
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def random_sample(self, n_samples):
        indices = np.random.choice(self.x.size(0), n_samples, replace=True)
        return self.x[indices], self.y[indices]
    def sample(self, n_samples, method):
        if method == 'ucb':
            next_idx = self.ucb_next_idx
            self.ucb_next_idx  = (self.ucb_next_idx + n_samples) 
            
        elif method == 'uniform':
            next_idx = self.uniform_next_idx
            self.uniform_next_idx  = (self.uniform_next_idx + n_samples) 
            
        elif method == 'target':
            next_idx = self.target_next_idx
            self.target_next_idx = (self.target_next_idx + n_samples) 
            
        elif method == 'mmd':
            next_idx = self.mmd_next_idx
            self.mmd_next_idx = (self.mmd_next_idx + n_samples)
        else:
            raise ValueError(f"Unsupported sampling method: {method}. Available methods: ['ucb', 'uniform','target']")
        
        return self.x[next_idx:next_idx+n_samples], self.y[next_idx:next_idx+n_samples]
    def distribution(self):
        return torch.bincount(self.y).float()/self.y.size(0) 

class BertActiveLearningData:
    def __init__(self, all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_genres, device):
        self.all_input_ids = all_input_ids.to(device)
        self.all_attention_mask = all_attention_mask.to(device)
        self.all_token_type_ids = all_token_type_ids.to(device)
        self.all_labels = all_labels.to(device)
        self.all_genres = all_genres.to(device)
        self.device = device
        self.pool_idx_mask = torch.ones(all_input_ids.size(0), dtype=torch.bool)
        self.train_idx_mask = torch.zeros(all_input_ids.size(0), dtype=torch.bool)
        self.fetch_times = [0] * 5
    def remove_from_pool_to_train(self, inds):
        inds = torch.tensor(inds, dtype=torch.long)
        self.pool_idx_mask[inds] = False
        self.train_idx_mask[inds] = True
        fetch_idx = self.all_genres[inds]
        self.fetch_times[fetch_idx] += 1
    def train_dataset(self):
        return (self.all_input_ids[self.train_idx_mask].to(self.device),
            self.all_attention_mask[self.train_idx_mask].to(self.device),
            self.all_token_type_ids[self.train_idx_mask].to(self.device),
            self.all_labels[self.train_idx_mask].to(self.device))
        
    def pool_dataset(self):
        return torch.utils.data.TensorDataset(
            self.all_input_ids[self.pool_idx_mask].to(self.device),
            self.all_attention_mask[self.pool_idx_mask].to(self.device),
            self.all_token_type_ids[self.pool_idx_mask].to(self.device),
            self.all_labels[self.pool_idx_mask].to(self.device),
        )

class ActiveLearningData:
    def __init__(self, x_pool, y_pool, x_source, y_source, device):
        self.x_pool = x_pool.to(device)
        self.y_pool = y_pool.to(device)
        self.x_train = x_source.to(device)
        self.y_train = y_source.to(device)
        self.device = device
        
    def remove_from_pool_to_train(self, inds):
        # inds is a list of indices to remove from the pool
        inds = torch.tensor(inds, dtype=torch.long)
        
        x_ = self.x_pool[inds]
        y_ = self.y_pool[inds]
        # Add selected samples to training set
        self.x_train = torch.cat([self.x_train, self.x_pool[inds]])
        self.y_train = torch.cat([self.y_train, self.y_pool[inds]])
        
        # Create mask to filter out selected indices
        mask = torch.ones(len(self.x_pool), dtype=torch.bool)
        mask[inds] = False
        
        # Update pool by removing selected samples
        self.x_pool = self.x_pool[mask]
        self.y_pool = self.y_pool[mask]
        
        return x_, y_
        
    def train_dataset(self):
        return torch.utils.data.TensorDataset(self.x_train.to(self.device), self.y_train.to(self.device))
    def pool_dataset(self):
        return torch.utils.data.TensorDataset(self.x_pool.to(self.device), self.y_pool.to(self.device))

class RAMBOUtilityData:
    def __init__(self, D):
        ''''
        D is a list, each element is a pair: ((x1, u1), (x2, u2))
        x shape [n_samples, n_features]
        u shape [n_samples]
        '''
        # First, split D into multiple subsets, each subset x' n_samples is the same
        self.D = D
        self.subsets = {}
        for pr in D:
            (x1, u1), (x2, u2) = pr
            if x1.size(0) not in self.subsets:
                self.subsets[x1.size(0)] = []
            self.subsets[x1.size(0)].append(pr)
    def generate_batch(self, batch_size):
        n_samples = random.choice(list(self.subsets.keys()))
        subset = self.subsets[n_samples]
        indices = np.random.choice(len(subset), batch_size, replace=True)
        batch = [subset[i] for i in indices]
        return batch
        
            
        

class RAMBOData:
    def __init__(self, x, y, s0_size, b, t1, ft_n_samples, ft_n_rounds):
        idx = random.sample(range(x.size(0)), x.size(0))
        assert s0_size + b*t1 <= x.size(0)
        self.x_S0 = x[idx[:s0_size]].to('cuda')
        self.y_S0 = y[idx[:s0_size]].to('cuda')
        self.raw_x_S_left = x[idx[s0_size:s0_size+b*t1]].to('cuda') 
        self.x_S_left = torch.split(self.raw_x_S_left, b, dim=0)
        self.raw_y_S_left = y[idx[s0_size:s0_size+b*t1]].to('cuda')
        self.y_S_left = torch.split(self.raw_y_S_left, b, dim=0)
        self.D = []
        self.ft_n_samples = ft_n_samples
        self.ft_n_rounds = ft_n_rounds
    
    def get_S(self):
        # if t == 0:
        #     return self.x_S0, self.y_S0
        
        x_S = torch.cat([self.x_S0, self.raw_x_S_left])
        y_S = torch.cat([self.y_S0, self.raw_y_S_left])
        return x_S, y_S
    
    def utility_samples_aug(self, x_S_prev, x_S_curr, n, acc_prev, acc_curr, set_nn):
        for i in range(n):
            n_samples = random.choice(range(start=self.ft_n_samples, stop=self.ft_n_samples*(self.ft_n_rounds+1), step=self.ft_n_rounds))
            # sample n_samples from x_S_prev
            z1 = x_S_prev[random.sample(range(x_S_prev.size(0)), n_samples)]
            z2 = x_S_prev[random.sample(range(x_S_prev.size(0)), n_samples)]
            emb_z1 = set_nn.embed(z1)
            emb_z2 = set_nn.embed(z2)
            emb_s_prev = set_nn.embed(x_S_prev)
            emb_s_curr = set_nn.embed(x_S_curr)
            dist_1_prev = wasserstein_distance(emb_z1, emb_s_prev)
            dist_2_prev = wasserstein_distance(emb_z2, emb_s_prev)
            dist_1_curr = wasserstein_distance(emb_z1, emb_s_curr)
            dist_2_curr = wasserstein_distance(emb_z2, emb_s_curr)
            alpha_1 = dist_1_curr / (dist_1_curr + dist_1_prev)
            alpha_2 = dist_2_curr / (dist_2_curr + dist_2_prev)
            u_1 = alpha_1 * acc_prev + (1-alpha_1) * acc_curr
            u_2 = alpha_2 * acc_prev + (1-alpha_2) * acc_curr
            self.D.append(((z1, u_1), (z2, u_2)))
    def get_D(self):
        return RAMBOUtilityData(self.D)
    def begin_acquisition(self, x_U, y_U):
        self.x_S = self.x_S0
        self.y_S = self.y_S0
        # self.x_S = torch.cat([self.x_S0, self.raw_x_S_left])
        # self.y_S = torch.cat([self.y_S0, self.raw_y_S_left])
        self.x_U = x_U
        self.y_U = y_U
    def greedy_margin(self, set_nn, pred_model, set_size, M, steps):
        for _ in range(steps):
            _, _ = self.greedy_margin_step(set_nn, pred_model,
                                    set_size, M)
    def greedy_margin_step(self, set_nn, pred_model, set_size, M):
        with torch.no_grad():
            y_pred = pred_model(self.x_U)
            n_subset = M // set_size
            margin = torch.abs(y_pred[:,0] - y_pred[:,1])
            _, indices = torch.topk(margin, M, largest=False)
            indices = indices[:n_subset*set_size]
            indices = indices.view(n_subset, set_size)
            i_tmp_max = 0
            
            for i in range(1, n_subset):
                input1 = self.x_U[indices[i_tmp_max]].unsqueeze(0)
                input2 = self.x_U[indices[i]].unsqueeze(0)
                _, _, prob = set_nn(input1, input2)
                if prob < 0.5:
                    i_tmp_max = i
            self.x_S = torch.cat([self.x_S, self.x_U[indices[i_tmp_max]]])
            self.y_S = torch.cat([self.y_S, self.y_U[indices[i_tmp_max]]])
            mask = torch.ones(len(self.x_U), dtype=torch.bool)
            mask[indices[i_tmp_max]] = False
            self.x_U = self.x_U[mask]
            self.y_U = self.y_U[mask]
            return self.x_S, self.y_S
            
            
            
                
                

                    
                

        
def flipped_datasets(x, y, feature_names):
    # 查找名为'race_white', 'race_black', 'race_am_ind', 'race_alaska', 'race_am_alaska', 'race_asian', 
    # 'race_hawaiian', 'race_other', 'race_two_or_more'  的特征下标
    # race_indexes = []
    # for i, feature_name in enumerate(feature_names):
    #     if feature_name.startswith('race'):
    #         race_indexes.append(i)
    # # 把x的race_indexes随机打乱
    # x_race_flipped = x.clone()  # 复制原始数据以避免修改原始数据
    # race_combinations = x_race_flipped[:, race_indexes]
    # unique_race_combinations, counts = torch.unique(race_combinations, dim=0, return_counts=True)
    # # print
    # for i, (combination, count) in enumerate(zip(unique_race_combinations, counts)):
    #     print(f"Race Combination {i}: {combination}, Count: {count}")
    # # 找到出现次数最少的组合
    # min_count_idx = torch.argmin(counts)  # 找到最小的计数索引
    # min_count_combination = unique_race_combinations[min_count_idx]  # 对应的唯一值
    # x_race_flipped[:, race_indexes] = min_count_combination
    
    # 查找married', 'widowed', 'divorced', 'separated', 'never'
    marital_indexes = []
    for i, feature_name in enumerate(feature_names):
        if feature_name in ['married', 'widowed', 'divorced', 'separated', 'never']:
            marital_indexes.append(i)
    # 查找'big_occp_MGR', 'big_occp_BUS', 'big_occp_FIN', 'big_occp_CMM', 'big_occp_ENG', 'big_occp_SCI', 'big_occp_CMS', 'big_occp_EDU', 'big_occp_ENT', 'big_occp_MED', 'big_occp_HLS', 'big_occp_PRT', 'big_occp_EAT', 'big_occp_CLN', 'big_occp_PRS', 'big_occp_SAL', 'big_occp_OFF', 'big_occp_FFF', 'big_occp_CON', 'big_occp_EXT', 'big_occp_RPR', 'big_occp_PRD', 'big_occp_TRN', 'big_occp_MIL', 'big_occp_no'
    x_marital_flipped = x.clone()
    marital_combinations = x_marital_flipped[:, marital_indexes]
    unique_marital_combinations, counts = torch.unique(marital_combinations, dim=0, return_counts=True)
    for i, (combination, count) in enumerate(zip(unique_marital_combinations, counts)):
        print(f"Marital Combination {i}: {combination}, Count: {count}")
        
    min_count_idx = torch.argmin(counts)
    min_count_combination = unique_marital_combinations[min_count_idx]
    x_marital_flipped[:, marital_indexes] = min_count_combination
    
    
    big_occp_indexes = []
    for i, feature_name in enumerate(feature_names):
        if feature_name.startswith('big_occp'):
            big_occp_indexes.append(i)
    x_big_occp_flipped = x.clone()
    big_occp_combinations = x_big_occp_flipped[:, big_occp_indexes]
    unique_big_occp_combinations, counts = torch.unique(big_occp_combinations, dim=0, return_counts=True)
    for i, (combination, count) in enumerate(zip(unique_big_occp_combinations, counts)):
        print(f"Big Occupation Combination {i}: {combination}, Count: {count}")
        
    min_count_idx = torch.argmin(counts)
    min_count_combination = unique_big_occp_combinations[min_count_idx]
    x_big_occp_flipped[:, big_occp_indexes] = min_count_combination
    
    cow_indexes = []
    for i, feature_name in enumerate(feature_names):
        if feature_name in ['cow_employee_profit', 'cow_employee_nonprofit', 'cow_localgovernment', 'cow_stategovernment', 'cow_federalgovernment', 'cow_selfemployed_own', 'cow_selfemployed_incorporated', 'cow_family_business', 'cow_unemployed']:
            cow_indexes.append(i)
    x_cow_flipped = x.clone()
    cow_combinations = x_cow_flipped[:, cow_indexes]
    unique_cow_combinations, counts = torch.unique(cow_combinations, dim=0, return_counts=True)
    for i, (combination, count) in enumerate(zip(unique_cow_combinations, counts)):
        print(f"Cow Combination {i}: {combination}, Count: {count}")
        
    min_count_idx = torch.argmin(counts)
    min_count_combination = unique_cow_combinations[min_count_idx]
    x_cow_flipped[:, cow_indexes] = min_count_combination
    
    
    relp_indexes = []
    for i, feature_name in enumerate(feature_names):
        if feature_name.startswith('relp'):
            relp_indexes.append(i)
    x_relp_flipped = x.clone()
    relp_combinations = x_relp_flipped[:, relp_indexes]
    unique_relp_combinations, counts = torch.unique(relp_combinations, dim=0, return_counts=True)
    for i, (combination, count) in enumerate(zip(unique_relp_combinations, counts)):
        print(f"Relp Combination {i}: {combination}, Count: {count}")
        
    min_count_idx = torch.argmin(counts)
    min_count_combination = unique_relp_combinations[min_count_idx]
    
    flipped_datasets = []
    flipped_datasets.append(ACSDataset(clone(x).to('cuda'), clone(y), state_name="no_flip"))
    # flipped_datasets.append(ACSDataset(x_race_flipped.to('cuda'), clone(y), state_name="race_flip"))
    flipped_datasets.append(ACSDataset(x_marital_flipped.to('cuda'), clone(y), state_name="marital_flip"))
    flipped_datasets.append(ACSDataset(x_big_occp_flipped.to('cuda'), clone(y), state_name="big_occp_flip"))
    flipped_datasets.append(ACSDataset(x_cow_flipped.to('cuda'), clone(y), state_name="cow_flip"))
    flipped_datasets.append(ACSDataset(x_relp_flipped.to('cuda'), clone(y), state_name="relp_flip"))
    # flipped_datasets.append(ACSDataset(x_sex_flipped.to('cuda'), clone(y), state_name='sex_flip'))
    random.shuffle(flipped_datasets)
    return flipped_datasets

def flipy_datasets(x, y, feature_names):   
    # 把y随机打乱
    
    flipped_datasets = []
    flipped_datasets.append(ACSDataset(clone(x).to('cuda'), clone(y), state_name="no_flip"))
    for i in range(3):
        y_flipped = y.clone()
        permuted_indices = torch.randperm(y_flipped.size(0))
        y_flipped = y_flipped[permuted_indices]
        flipped_datasets.append(ACSDataset(clone(x).to('cuda'), y_flipped, state_name=f"flip_{i}"))
    return flipped_datasets
def shift_datasets(x, y, feature_names):
    shifted_datasets = []
    shifted_datasets.append(ACSDataset(clone(x).to('cuda'), clone(y), state_name="no_shift"))
    for ratio in [1, 2]:
        y_shifted = y.clone()
        y_shifted = y_shifted + ratio
        shifted_datasets.append(ACSDataset(clone(x).to('cuda'), y_shifted.to('cuda'), state_name=f"shift_{ratio}"))
    return shifted_datasets
def white_noise_datasets(x, y, feature_names):
    noise_levels = [0.2, 0.1]
    datasets = []
    datasets.append(ACSDataset(clone(x).to('cuda'), clone(y), state_name="no_noise"))

    for noise_level in noise_levels:
        y_noisy = clone(y).to('cuda')  # 复制 y 以避免修改原始数据
        num_samples = y.size(0)  # 样本数量
        num_noisy_samples = int(num_samples * noise_level)  # 计算污染样本数量（50%）
        noisy_indices = torch.randperm(num_samples)[:num_noisy_samples]
        noise = torch.normal(mean=2, std=1.0, size=(num_noisy_samples, 1)).to('cuda')
        y_noisy[noisy_indices] += noise
        datasets.append(ACSDataset(clone(x).to('cuda'), y_noisy, state_name=f"noise_{noise_level}"))
    return datasets


class InnerRegionDataset(Dataset):
    def __init__(self, acs_dataset:ACSDataset, attribute_idx, value):
        super(InnerRegionDataset, self).__init__()
        self.x = []
        self.y = []
        for i in range(len(acs_dataset)):
            x, y = acs_dataset[i]
            if x[attribute_idx] == value:
                self.x.append(x)
                self.y.append(y)
        self.x = torch.stack(self.x).to('cuda')
        self.y = torch.stack(self.y).to('cuda')
        # 随机打乱
        idx = random.sample(range(self.x.size(0)), self.x.size(0))
        self.x = self.x[idx]
        self.y = self.y[idx]
        
        self.ucb_next_idx = 0
        self.uniform_next_idx = 0
        self.target_next_idx = 0
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def sample(self, n_samples, method):
        if method == 'ucb':
            next_idx = self.ucb_next_idx
            self.ucb_next_idx  += n_samples // len(self.y)
            
        elif method == 'uniform':
            next_idx = self.uniform_next_idx
            self.uniform_next_idx += n_samples // len(self.y)
            
        elif method == 'target':
            next_idx = self.target_next_idx
            self.target_next_idx += n_samples // len(self.y)
        else:
            raise ValueError(f"Unsupported sampling method: {method}. Available methods: ['ucb', 'uniform','target']")
        
        return self.x[next_idx:next_idx+n_samples], self.y[next_idx:next_idx+n_samples]
    
def dataset_split(dataset, ratio=0.8):
    n = len(dataset)
    n_train = int(n*ratio)
    n_valid = n - n_train
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid])

    
    
    return train_dataset, valid_dataset
import torch
from torch.utils.data import Dataset, Subset
import numpy as np

def dataset_split_equal(dataset, val_samples):
    """
    Split a dataset such that each class has the same number of samples.

    Args:
        dataset (Dataset): The dataset to split, assumed to have `targets` attribute.
        samples_per_class (int): The number of samples to select per class.

    Returns:
        Subset: A subset of the dataset with balanced classes.
    """
    # 获取目标标签
    targets = dataset.y.cpu()
    # print("val samples: ", val_samples)
    # 获取所有类别
    classes = torch.unique(targets).tolist()
    selected_indices = []
    samples_per_class = val_samples // len(classes)
    
    # 对每个类别采样固定数量
    for cls in classes:
        indices = (targets == cls).nonzero(as_tuple=True)[0]
        selected_indices.extend(indices[:samples_per_class].tolist())

    # 创建平衡后的数据集
    val_subset = Subset(dataset, selected_indices)


    
    train_indices = list(set(range(len(dataset))) - set(selected_indices))
    train_subset = Subset(dataset, train_indices)

    return train_subset, val_subset


        
def sample_from_dataset(dataset, n_samples):
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=n_samples, shuffle=False)
    return next(iter(loader))        
    # 取前n_samples个样本



if __name__ == "__main__":
    # X, Y, _ = get_USAccident("CA", need_preprocess=True, root_dir='data/accident/US_Accidents_Dec21_updated.csv')
    # print(X.shape, Y.shape)
    # print(X[0], Y[0])
    raw_X = preprocess('data/accident/US_Accidents_Dec21_updated.csv')
    # 保存raw_x到本地
    raw_X.to_csv('data/accident/US_Accidents_Dec21_updated_rawx.csv', index=False)