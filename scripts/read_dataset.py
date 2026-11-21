import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from pickle import dump, load
import os
states_name = np.array(['Q1','Q2','S1','S2','I','X1', 'X2','X3','C2','C1'])
states = ["output_cgm","state_Q2","state_S1","state_S2","state_I","state_X1","state_X2","state_X3","state_Ug","state_Ug2"]

lookback_insulin = False
lookback_hrs_insulin = None

lookback_carbs = False
lookback_hrs_carbs = None

output = 0
ts = 5

inputs = np.array(['input_insulin','input_meal_carbs'])

dict_states = { 'Q1':['X1','X3','Q1','Q2','C2'],
                'Q2':['X1','X2','Q1','Q2'],
                'S1':['S1','input_insulin'],
                'S2':['S1','S2'],
                'I':['S2','I'],
                'X1':['I','X1'],
                'X2':['I','X2'],
                'X3':['I','X3'],
                'C2':['C1','C2'],
                'C1':['C1','input_meal_carbs']}

path_scaler = './scaler/'
os.makedirs(path_scaler, exist_ok=True)

def scaler(x_est,u_id, train=False):
    if train:
        print('INFO: Training scaler...')
        scaler_states = RobustScaler()#MinMaxScaler()
        scaler_inputs = RobustScaler()#MinMaxScaler()

        x_est = scaler_states.fit_transform(x_est)
        u_id = scaler_inputs.fit_transform(u_id)

        # Save the scaler
        dump(scaler_states, open(path_scaler+'scaler_states.pkl', 'wb'))
        dump(scaler_inputs, open(path_scaler+'scaler_inputs.pkl', 'wb'))

        return x_est, u_id
    else:
        print('INFO: Loading scaler...')
        scaler_states = load(open(path_scaler+'scaler_states.pkl', 'rb'))
        scaler_inputs = load(open(path_scaler+'scaler_inputs.pkl', 'rb'))

        x_est = scaler_states.transform(x_est)
        u_id = scaler_inputs.transform(u_id)
        return x_est, u_id

def lookback(u_input,lookback_hrs,ts):
    input = ["input_tm" + str(ts * ii) for ii in np.arange((lookback_hrs * 60 // ts))]
    input.reverse()

    df = pd.DataFrame(0, columns=input, index=np.arange(len(u_input)))

    lookback_steps = int(12 * lookback_hrs)
    u_input_aux = np.concatenate((np.zeros(lookback_steps-1),u_input))

    df.loc[:,input] = np.vstack([u_input_aux[index: index  + lookback_steps ] for index in df.index])

    return df.values

def read_dataset(path,states,inputs,output,scale,ts,lookback_carbs,lookback_hrs_carbs,lookback_insulin,lookback_hrs_insulin,train=True,return_df=False):

    sim_time = 7*24*12
    time_id = np.arange(sim_time).astype(np.float32)*ts

    print(path)
    if type(path)==list:
        file = []
        file_corr = []
        for p in path:
            if 'bolusCorrectionFactor_' in p:
                file_corr+=glob.glob(p + "*.pkl")
            else:
                file+=glob.glob(p + "*.pkl")
    else:
        file = glob.glob(path + "*.pkl")
        file_corr = []

    #file = file[:5]
    #file_corr = file_corr[:5]


    #print('INFO: Reading files: ',file)
    #print('INFO: Reading files with correction boluses: ',file_corr)

    if len(file)>0:
        df_data = pd.concat([pd.read_pickle(f) for f in file], ignore_index=True)
        df_data.dosing_problem_id = df_data.dosing_problem_id.apply(str)
        df_data['state_Ug2'] = 0

        df_data['subject'] = np.repeat([f.split('/')[-1].split('_')[1] for f in file],11*2016)
        df_data['subject_meal'] = np.repeat([f.split('/')[-1].split('_')[1]+f.split('/')[-1].split('_')[2]+f.split('POPULATION')[0].split('/')[-2] for f in file],11*2016)
        
        df_data['simID'] = np.repeat([f.split('/')[1] for f in file],11*2016) # This is highly dependent on the path structure, so it may need to be changed if the path changes

    if len(file_corr)>0:
        df_data_corr = pd.concat([pd.read_pickle(f) for f in file_corr], ignore_index=True)

        df_data_corr['event'] = np.repeat(np.arange(len(df_data_corr)/2016),2016)

        df_data_corr.dosing_problem_id = df_data_corr.dosing_problem_id.apply(str)
        df_data_corr['state_Ug2'] = 0

        df_data_corr['subject'] = np.repeat([f.split('/')[-1].split('_')[1] for f in file_corr],11*2016)
        df_data_corr['subject_meal'] = np.repeat([f.split('/')[-1].split('_')[1]+f.split('/')[-1].split('_')[2]+f.split('POPULATION')[0].split('/')[-2] for f in file_corr],11*2016)

        df_data_corr['simID'] = 'SIM005'#np.repeat([f.split('/')[6] for f in file_corr],11*2016)

        # Filter only yhose sequences with correction boluses
        event_cbolus = df_data_corr.groupby(['event']).sum()['correction_bolus'].reset_index()

        events = event_cbolus.loc[event_cbolus.correction_bolus>0]['event'].values
        df_data_corr = df_data_corr.set_index('event').loc[events].reset_index()

        df_data = pd.concat([df_data,df_data_corr],ignore_index=True)

    x_est = np.array(df_data[states].values).astype(np.float32)
    u_id = np.array(df_data[inputs].values).astype(np.float32)
    simID = np.array(df_data['simID'].values)

    # Robust scaler
    if scale:
        x_est, u_id = scaler(x_est,u_id,train)

    #states_scaled = [s+'_scaled' for s in states]
    #inputs_scaled = [s+'_scaled' for s in inputs]

    #df_data[states] = x_est
    #df_data[inputs] = u_id

    min_values = np.min(x_est,axis=0)
    y_id = np.copy(x_est[:, output]).reshape(-1,sim_time,1)
    x_est = x_est.reshape(-1,sim_time,len(states))
    u_id = u_id.reshape(-1,sim_time,len(inputs))
    simID = simID[::sim_time]

    u_id_lookback = u_id

    if return_df:
        return y_id,x_est,u_id_lookback, time_id, min_values,df_data

    else:
        return y_id,x_est,u_id_lookback, time_id, min_values,simID

def read_file(df_data,states,inputs,output,scale,ts,lookback_carbs,lookback_hrs_carbs,lookback_insulin,lookback_hrs_insulin):

    sim_time = 7*24*12
    scenario = 0
    time_id = np.arange(sim_time).astype(np.float32)*ts

    x_est = np.array(df_data[states].values).astype(np.float32)
    u_id = np.array(df_data[inputs].values).astype(np.float32)

    if scale:
        x_est, u_id = scaler(x_est,u_id,False)

    y_id = np.copy(x_est[:, output]).reshape(-1,sim_time,1)
    x_est = x_est.reshape(-1,sim_time,len(states))
    u_id = u_id.reshape(-1,sim_time,len(inputs))

    if lookback_insulin:
        u_insulin = lookback(u_id[scenario,:,0],lookback_hrs_insulin,ts)
    else:
        u_insulin = u_id[scenario,:,0].reshape(-1,1)

    if lookback_carbs:
        u_carbs = lookback(u_id[scenario,:,1],lookback_hrs_carbs,ts)
    else:
        u_carbs = u_id[scenario,:,1].reshape(-1,1)

    u_id_lookback = np.concatenate((u_insulin,u_carbs),axis=1).reshape(1,u_id.shape[1],-1)

    return y_id,x_est,u_id_lookback, time_id


if True:
    import time
    start_time = time.time()
    #y_id,x_est,u_id,time_id,min_values = read_dataset(dataset_path+'TRAIN/',states,inputs,output,scale,ts,lookback_carbs,lookback_hrs_carbs,lookback_insulin,lookback_hrs_insulin,train=False)
    #y_id_validation,x_est_validation,u_id_validation,time_id_validation,_ = read_dataset(dataset_path+'VALIDATION/',states,inputs,output,scale,ts,lookback_carbs,lookback_hrs_carbs,lookback_insulin,lookback_hrs_insulin,train=False)

    scale = True


    # Read training set
    dataset_path =  ['OHSUSimPopulation/SIM002/POPULATION/TRAIN/'] + \
                    ['OHSUSimPopulation/SIM004/POPULATION/boluslate_'+str(t)+'min/TRAIN/' for t in range(5,50,5)] +\
                    ['OHSUSimPopulation/SIM005/POPULATION/bolusCorrectionFactor_'+str(t)+'/TRAIN/' for t in [28,35,42]]

    print('INFO: Start reading training dataset...')
    y_id,x_est,u_id,time_id,min_values,simID = read_dataset(dataset_path,states,inputs,output,scale,ts,lookback_carbs,lookback_hrs_carbs,lookback_insulin,lookback_hrs_insulin,train=True)

    # Read validation set
    dataset_path =  'OHSUSimPopulation/SIM002/POPULATION/VALIDATION/'

    print('INFO: Start reading validation dataset...')
    y_id_validation,x_est_validation,u_id_validation,time_id_validation,_,simID_val = read_dataset(dataset_path,states,inputs,output,scale,ts,lookback_carbs,lookback_hrs_carbs,lookback_insulin,lookback_hrs_insulin,train=False)

    print('INFO: Finish reading dataset')
    print('INFO: Time to read dataset: ',time.time() - start_time)