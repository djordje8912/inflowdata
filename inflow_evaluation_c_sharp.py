
#evaluation on one input vector
import pandas as pd
import numpy as np  
import torch 
from torch import nn
from bs4 import BeautifulSoup
from scipy.signal import savgol_filter
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#from torch.utils.data.dataset import Dataset
#from torch.utils.data import random_split
#from torch import split
from datetime import datetime
#from matplotlib import pyplot as plt 
from sklearn.preprocessing import StandardScaler   
#from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError
#from sklearn.metrics import mean_squared_error
#from inflow_prediction import LSTM
#from inflow_prediction import split_sequences
import json
import joblib
#from sklearn.metrics import r2_score
#import datetime
device ='cpu'
USE_CUDA=False

print(torch.__version__)
def interpolate(a,abc):
    n=[]
    for i in range(len(a)-1):
        n.append(a[i])
        for j in range(1,abc):
            n.append(a[i]+(a[i+1]-a[i])/8*j)
    return n



class LSTM3(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers,fully_connected_layer_neurons_number,dropout,device,use_cuda=False):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        self.fully_connected_layer_neurons_number=fully_connected_layer_neurons_number
        self.use_cuda=use_cuda
        self.device=device
        self.dropout=dropout
        # LSTM model
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout) # lstm
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout) # lstm
        self.fc_1 =  nn.Linear(hidden_size, fully_connected_layer_neurons_number) # fully connected 
        self.fc_2 = nn.Linear(fully_connected_layer_neurons_number, fully_connected_layer_neurons_number) #  fully connected
        self.fc_3 = nn.Linear(fully_connected_layer_neurons_number, num_classes) # fully connected last layer

        self.fc_after1 = nn.Linear(input_size, fully_connected_layer_neurons_number) # fully connected last layer
        self.fc_after2 = nn.Linear(fully_connected_layer_neurons_number, 1) # fully connected last layer
        self.relu = nn.ReLU()
        #self.layers = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=1)
        self.linear_layer = nn.Linear(in_features=input_size, out_features=1)
        self.m = nn.Mish()
        #self._n_splits = input_size
    def forward(self,x,x2):
         #print(x.shape)
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  #
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # propagate input through LSTM
        if self.use_cuda:
            h_0=h_0.to(self.device)
            c_0=c_0.to(self.device)
            #h_1=h_1.to(self.device)
            #c_1=c_1.to(self.device)
        #print(x.shape,x2.shape)
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        #output, (hn, cn) = self.lstm(torch.unsqueeze(x,1), (h_0, c_0)) # (input, hidden, and internal state)
        #output, (hn, cn) = self.lstm2(hn, (h_1, c_1))  #added 
        #hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        #print(hn)
        out = self.relu(output[:,-1,:])

        #out = self.relu(hn)
       
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # second output
        out = self.relu(out) # relu
        out = self.fc_3(out) # final output
        #print("izlaz {0} ".format(out.shape))
        #out=out.reshape(-1,1)
        #print(out,x2)
        
        out=out.unsqueeze(2)
        #print(out.shape,x2.shape)
        out=torch.cat((x2, out),2)  #.swapaxes(1,2)
        a,b,c =out.shape
        torch.reshape(out, (a*b, c))
        #print(out.shape)
        #B, C = out.shape
        #print(B,C)
        #print(out.view(B, C//self._n_splits, -1))
        #output = self.layers(out.view(B, C//self._n_splits, -1))

        #output=self.linear_layer(torch.reshape(out, (5, 14,1)))
        #output=self.linear_layer(out.view(out.size(0), -1))

        #method 1
        output=self.linear_layer(out)

        out=self.fc_after1(out)
        out = self.m(out)
        output=self.fc_after2(out)
        #output = self.linear_layer(torch.unsqueeze(out,2))
        #print(output.shape)
        return output.squeeze()

class LSTM2(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers,fully_connected_layer_neurons_number,dropout,device,use_cuda=False):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        self.fully_connected_layer_neurons_number=fully_connected_layer_neurons_number
        self.use_cuda=use_cuda
        self.device=device
        self.dropout=dropout
        # LSTM model
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout) # lstm
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout) # lstm
        self.fc_1 =  nn.Linear(hidden_size, fully_connected_layer_neurons_number) # fully connected 
        self.fc_2 = nn.Linear(fully_connected_layer_neurons_number, fully_connected_layer_neurons_number) #  fully connected
        self.fc_3 = nn.Linear(fully_connected_layer_neurons_number, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        # self.layers = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=1)
        self.linear_layer = nn.Linear(in_features=input_size, out_features=1)
        # self._n_splits = input_size
    def forward(self,x,x2):
        
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  #
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #h_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #c_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # propagate input through LSTM
        if self.use_cuda:
            h_0=h_0.to(self.device)
            c_0=c_0.to(self.device)
            #h_1=h_1.to(self.device)
            #c_1=c_1.to(self.device)
        #print(x.shape,h_0.shape)
        #print(torch.unsqueeze(x,1).shape)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # output, (hn,cn) = self.lstm(torch.unsqueeze(x,1), (h_0,c_0))   old
        #output, (hn,cn) = self.lstm(torch.squeeze(x), (h_0,c_0)) 
        #output, (hn, cn) = self.lstm(torch.unsqueeze(x,1), (h_0, c_0)) # (input, hidden, and internal state)
        #output, (hn, cn) = self.lstm2(hn, (h_1, c_1))  #added 
        #hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        #print(hn)
        out = self.relu(hn)
       
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # second output
        out = self.relu(out) # relu
        out = self.fc_3(out) # final output
        #print("izlaz {0} ".format(out.shape))
        # print(out.shape,x2.shape)
        out=out.squeeze(0).unsqueeze(2)
       #print(out.shape,x2.shape)
        out=torch.cat((x2, out),2)  #.swapaxes(1,2)
        a,b,c =out.shape
        torch.reshape(out, (a*b, c))
       #print(out.shape)
       #B, C = out.shape
       #print(B,C)
       #print(out.view(B, C//self._n_splits, -1))
       #output = self.layers(out.view(B, C//self._n_splits, -1))

       #output=self.linear_layer(torch.reshape(out, (5, 14,1)))
       #output=self.linear_layer(out.view(out.size(0), -1))
        output=self.linear_layer(out)

       #output = self.linear_layer(torch.unsqueeze(out,2))
       #print(output.shape)
        return output.squeeze()
        # old
        # out=out.reshape(-1,1)
        # #print(out,x2)
        
        # #print(out.shape,x2.shape)
        # out=torch.cat((x2, out),1)
        # #print(out)
        # B, C = out.shape
        # #print(B,C)
        # #print(out.view(B, C//self._n_splits, -1))
        # #output = self.layers(out.view(B, C//self._n_splits, -1))

        # #output=self.linear_layer(torch.reshape(out, (5, 14,1)))
        # output=self.linear_layer(out.view(out.size(0), -1))
        

        # #output = self.linear_layer(torch.unsqueeze(out,2))
        # #print(output)
        # return output
def one_step_forecast2(model, history,forecast):
      '''
      model: PyTorch model object
      history: a sequence of values representing the latest values of the time 
      series, requirement -> len(history.shape) == 2
    
      outputs a single value which is the prediction of the next value in the
      sequence.
      '''
      model.cpu()
      model.eval()
      with torch.no_grad():

        pre = torch.Tensor(history)
      
        pre2 = torch.Tensor(forecast)
        # print(pre.unsqueeze(0).shape,pre2.unsqueeze(0).shape)
        pred = model(pre.unsqueeze(0),pre2.unsqueeze(0))
      return pred.detach().numpy().reshape(-1)
def one_step_forecast3(model, history,forecast):
      '''
      model: PyTorch model object
      history: a sequence of values representing the latest values of the time 
      series, requirement -> len(history.shape) == 2
    
      outputs a single value which is the prediction of the next value in the
      sequence.
      '''
      model.cpu()
      model.eval()
      with torch.no_grad():

        pre = torch.Tensor(history)
      
        pre2 = torch.Tensor(forecast)
        print(pre.unsqueeze(0).shape,pre2.unsqueeze(0).shape)
        pred = model(pre.unsqueeze(0),pre2.unsqueeze(0))
      return pred.detach().numpy().reshape(-1)

class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers,fully_connected_layer_neurons_number,dropout,device,use_cuda=False):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        self.fully_connected_layer_neurons_number=fully_connected_layer_neurons_number
        self.use_cuda=use_cuda
        self.device=device
        self.dropout=dropout
        # LSTM model
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout) # lstm
        #self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
        #                    num_layers=self.num_layers, batch_first=True, dropout=self.dropout) # lstm
        self.fc_1 =  nn.Linear(hidden_size, fully_connected_layer_neurons_number) # fully connected 
        self.fc_2 = nn.Linear(fully_connected_layer_neurons_number, fully_connected_layer_neurons_number) #  fully connected
        self.fc_3 = nn.Linear(fully_connected_layer_neurons_number, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  #
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        h_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # propagate input through LSTM
       # if self.use_cuda:
        #    h_0=h_0.to(self.device)
        #    c_0=c_0.to(self.device)
        #    h_1=h_1.to(self.device)
         #   c_1=c_1.to(self.device)


        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        #output, (hn, cn) = self.lstm2(hn, (h_1, c_1))  #added 
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
       
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # second output
        out = self.relu(out) # relu
        out = self.fc_3(out) # final output
        return out
def one_step_forecast(model, history):
      '''
      model: PyTorch model object
      history: a sequence of values representing the latest values of the time 
      series, requirement -> len(history.shape) == 2
    
      outputs a single value which is the prediction of the next value in the
      sequence.
      '''
      model.cpu()
      model.eval()
      with torch.no_grad():
        pre = torch.Tensor(history).unsqueeze(0)
        pred = model(pre)
      return pred.detach().numpy().reshape(-1)

def evaluation():
    with open("InflowForecast.dll.config", "r") as file:
    # Read each line in the file, readlines() returns a list of lines
        content = file.readlines()
    # Combine the lines in the list into a string
    content = "".join(content)
    #print(content)
    bs_content = BeautifulSoup(content)
    days_range=bs_content.find("add", {"key": "days_range"}).get('value')
    resolution=bs_content.find("add", {"key": "resolution"}).get('value')
    k1=bs_content.find("add", {"key": "Koman_1D"}).get('value')
    v1=bs_content.find("add", {"key": "VauDejes_1D"}).get('value')
    f1=bs_content.find("add", {"key": "Fierze_1D"}).get('value')
    k3=bs_content.find("add", {"key": "Koman_3D"}).get('value')
    v3=bs_content.find("add", {"key": "VauDejes_3D"}).get('value')
    f3=bs_content.find("add", {"key": "Fierze_3D"}).get('value')
    k6=bs_content.find("add", {"key": "Koman_6D"}).get('value')
    v6=bs_content.find("add", {"key": "VauDejes_6D"}).get('value')
    f6=bs_content.find("add", {"key": "Fierze_6D"}).get('value')
    k14=bs_content.find("add", {"key": "Koman_14D"}).get('value')
    v14=bs_content.find("add", {"key": "VauDejes_14D"}).get('value')
    f14=bs_content.find("add", {"key": "Fierze_14D"}).get('value')
    startOfPrediction=datetime.strptime(bs_content.find("add", {"key": "startOfPrediction"}).get('value'),'%Y-%m-%dT%H:%M')
    locationOfArchiveCsvPrediction=bs_content.find("add", {"key": "locationOfArchiveCsvPrediction"}).get('value')
    archive_forecast=bs_content.find("add", {"key": "archive_forecast"}).get('value')
    # previous_time_forecast=bool(bs_content.find("add", {"key": "previous_time_forecast"}).get('value'))
    
    
    if(days_range=="6"):
        json_name=[f6,k6,v6   ]
    elif(days_range=="1"):
        json_name=[f1,k1,v1   ]
    elif(days_range=="3"):
        json_name=[f3,k3,v3   ]
    else:
        json_name=[f14,k14,v14   ]
    df = pd.read_csv('input.csv')

    df1= pd.DataFrame(df, columns=["Kukes_rain","Kukes_temp","Kukes_humi","Okshtun_rain","Okshtun_temp","Okshtun_humi","Fierze_rain","Fierze_temp","Fierze_humi","Zogaj_temp","Zogaj_rain","Peshkopi_temp","Peshkopi_rain","Inflows"])
        
    df1.reset_index(drop=True)
    # print(df1.head())
    #df2 = df2.drop(df2.columns[0],axis=1)

    X1, y1 = df1, df1.Inflows.values
    #second way
    #y  =df2.Inflows.values
    #X= df2.drop(df2.columns[13],axis=1)


    df2= pd.DataFrame(df, columns=["Koman_rain","Koman_temp","Koman_humi","Dragobi_rain","Dragobi_temp","Dragobi_humi","Theth_rain","Theth_temp","Theth_humi","Puke_temp","Puke_rain","Puke_humi","InflowsTributary"])
        
    df2.reset_index(drop=True)
    #df2 = df2.drop(df2.columns[0],axis=1)
    X2, y2 = df2, df2.InflowsTributary.values

    df3= pd.DataFrame(df, columns=["VauDejes_rain","VauDejes_temp","VauDejes_humi","Koman_rain","Koman_temp","Koman_humi","Puke_temp","Puke_rain","Puke_humi","InflowsTributary2"])
        
    df3.reset_index(drop=True)
    #df2 = df2.drop(df2.columns[0],axis=1)
    X3, y3 = df3, df3.InflowsTributary2.values



    #print(json_name)
    X=[X1,X2,X3]
    #json_name=["20231207091142","20231207100810","20231207105236"]
    output=[]
    for k in range(len(X)):
        #timestamp=json_name[15:]
        # Opening JSON file
        f = open('models/hyperparameters'+json_name[k]+'.json')
        data = json.load(f)


        fully_connected_layer_neurons_number=int(data["fully_connected_layer_neurons_number"])
        input_size = int(data["input_size"])
        hidden_size = int(data["hidden_size"]) # number of features in hidden state
        num_layers =int(data["num_layers"]) # number of stacked lstm layers
        prediction_window = int(data["prediction_window"]) # number of output classes 
        history_window = int(data["history_window"])
        dropout = float(data["dropout"]) # number of output classes 
        train_test_rate= int(data["train_test_rate"]) 
        dataset_name=str(data["dataset_name"])
        predicted_value=str(data["predicted_value"])
        train_test_rate= int(data["train_test_rate"]) 
        algorithm_type= int(data["algorithm_type"]) 
        
        #prediction_window_extended=24*7
        if(predicted_value=="Inflows"):
            prediction_index=0
        elif(predicted_value=="InflowsTributary"):
            prediction_index=1
        else:
            prediction_index=2


        model_name = 'models/model_'+json_name[k]+'.pth'
        
        if(algorithm_type==3):
            abc=int((dataset_name.split('_'))[1].split('h')[0])
            X[k] =X[k].iloc[96-history_window*abc:]
            if(predicted_value=="Inflows"):
                X[k]=pd.read_csv('input.csv',parse_dates=["Date"])
                X[k]['dayofyear'] = X[k]['Date'].dt.dayofyear/366
                X[k]['hour'] = X[k]['Date'].dt.hour/24
                X[k]= pd.DataFrame(X[k], columns=["dayofyear","hour","Kukes_rain","Fierze_rain","Fierze_temp","Fierze_humi","Peshkopi_rain","Inflows"])
                X[k].reset_index(drop=True)
                X[k]['Fierze_rain'] =  X[k]['Fierze_rain'].shift(48)
                X[k]['Kukes_rain'] =  X[k]['Kukes_rain'].shift(48)
                X[k]['Peshkopi_rain'] =  X[k]['Peshkopi_rain'].shift(48)
                X[k] = X[k].dropna()
                X[k]=X[k].groupby(np.arange(len(X[k]))//abc).mean()
            elif(predicted_value=="InflowsTributary"):
                X[k]=pd.read_csv('input.csv',parse_dates=["Date"])
                X[k]['dayofyear'] = X[k]['Date'].dt.dayofyear/366
                X[k]['hour'] = X[k]['Date'].dt.hour/24
                X[k]= pd.DataFrame(X[k], columns=["dayofyear","hour","Koman_rain","Koman_temp","Koman_humi","Puke_rain","InflowsTributary"])  
                X[k].reset_index(drop=True)
                X[k]['Fierze_rain'] =  X[k]['Koman_rain'].shift(48)
                X[k]['Kukes_rain'] =  X[k]['Puke_rain'].shift(48)
                X[k] = X[k].dropna()
                X[k]=X[k].groupby(np.arange(len(X[k]))//abc).mean()
            else:
                X[k]=pd.read_csv('input.csv',parse_dates=["Date"])
                X[k]['dayofyear'] = X[k]['Date'].dt.dayofyear/366
                X[k]['hour'] = X[k]['Date'].dt.hour/24
                X[k]= df2= pd.DataFrame(X[k], columns=["dayofyear","hour","VauDejes_rain","VauDejes_temp","VauDejes_humi","InflowsTributary2"])  
                X[k].reset_index(drop=True)
                X[k]['Fierze_rain'] =  X[k]['VauDejes_rain'].shift(48)
                X[k] = X[k].dropna()
                X[k]=X[k].groupby(np.arange(len(X[k]))//abc).mean()
                pass
            lstm = LSTM3(prediction_window, 
                            input_size, 
                            hidden_size, 
                            num_layers,
                            fully_connected_layer_neurons_number,
                            dropout,
                            device
                    ).to(device)


            lstm.load_state_dict(torch.load(model_name,map_location=torch.device('cpu') ))
            # lstm.load_state_dict(torch.load(model_name))
            lstm = lstm.to(device)

            dataset_name = dataset_name.replace("test", "train")
            scaler_filename1='models/scaler_ss_'+dataset_name+'_'+str(prediction_index)+'.pkl'
            scaler_filename2='models/scaler_mm_'+dataset_name+'_'+str(prediction_index)+'.pkl'
            ss = joblib.load(scaler_filename1) 
            mm = joblib.load(scaler_filename2) 
            #eval_set=np.genfromtxt("input_example.csv", delimiter=',') 
            #print(eval_set)

            input_arr=X[k].to_numpy()
            print(input_arr)

            #print(input_arr)
            
      
            #print("range"+str(len(input_arr)-96))
            for i in range(int((len(input_arr)-history_window)/prediction_window)):
                inp=input_arr[i*prediction_window:i*prediction_window+history_window,:]
                print(inp[-1,-1],inp[-1,2])
                inp2=input_arr[i*prediction_window+history_window:(i+1)*prediction_window+history_window,:]
                
                df_X_ss = ss.transform(inp)
                print(inp)
                f_X_ss = ss.transform(inp2)
                f_X_ss=f_X_ss[:,0:input_arr.shape[1]-1]
                data_predict =one_step_forecast3(lstm,df_X_ss,f_X_ss)
                #print(data_predict)
                #print(mm.inverse_transform(data_predict))
                predict=mm.inverse_transform(data_predict.reshape(-1,1))
                #print(len(predict))
                for j in range(len(predict)):

                    input_arr[i*prediction_window+history_window+j][len(input_arr[0])-1]=predict[j]
                arr=interpolate(predict.reshape(-1),abc)
                if(abc==1):
                    for k in range(len(predict)):
                        df.loc[history_window+i*prediction_window*abc+k,predicted_value] = predict[k]
                else:
                    for k in range(len(arr)):
                        df.loc[history_window+i*prediction_window*abc+k,predicted_value] = arr[k]
        elif(algorithm_type==2):
            lstm = LSTM2(prediction_window, 
                            input_size, 
                            hidden_size, 
                            num_layers,
                            fully_connected_layer_neurons_number,
                            dropout,
                            device
                    ).to(device)


            lstm.load_state_dict(torch.load(model_name,map_location=torch.device('cpu') ))
            # lstm.load_state_dict(torch.load(model_name))
            lstm = lstm.to(device)

            dataset_name = dataset_name.replace("test", "train")
            scaler_filename1='models/scaler_ss_'+dataset_name+'_'+str(prediction_index)+'.pkl'
            scaler_filename2='models/scaler_mm_'+dataset_name+'_'+str(prediction_index)+'.pkl'
            ss = joblib.load(scaler_filename1) 
            mm = joblib.load(scaler_filename2) 
            #eval_set=np.genfromtxt("input_example.csv", delimiter=',') 
            #print(eval_set)

            input_arr=X[k].to_numpy()


            #print(input_arr)

      
            #print("range"+str(len(input_arr)-96))
            for i in range(int((len(input_arr)-96)/prediction_window)):
                inp=input_arr[i*prediction_window:i*prediction_window+96,:]

                inp2=input_arr[i*prediction_window+96:(i+1)*prediction_window+96,:]
                
                df_X_ss = ss.transform(inp)
                f_X_ss = ss.transform(inp2)
                f_X_ss=f_X_ss[:,0:input_arr.shape[1]-1]
                data_predict =one_step_forecast2(lstm,df_X_ss,f_X_ss)
                #print(data_predict)
                #print(mm.inverse_transform(data_predict))
                predict=mm.inverse_transform(data_predict.reshape(-1,1))
                #print(len(predict))
                for j in range(len(predict)):

                    input_arr[i*prediction_window+96+j][len(input_arr[0])-1]=predict[j]
                    df.loc[96+i*prediction_window+j,predicted_value] = predict[j]
        else:
            lstm = LSTM(prediction_window, 
                            input_size, 
                            hidden_size, 
                            num_layers,
                            fully_connected_layer_neurons_number,
                            dropout,
                            device
                            ).to(device)



            lstm.load_state_dict(torch.load(model_name,map_location=torch.device('cpu') ))
            lstm = lstm.to(device)

            dataset_name = dataset_name.replace("test", "train")
            scaler_filename1='models/scaler_ss_'+dataset_name+'_'+str(prediction_index)+'.pkl'
            scaler_filename2='models/scaler_mm_'+dataset_name+'_'+str(prediction_index)+'.pkl'
            ss = joblib.load(scaler_filename1) 
            mm = joblib.load(scaler_filename2) 
            #eval_set=np.genfromtxt("input_example.csv", delimiter=',') 
            #print(eval_set)

            input_arr=X[k].to_numpy()


            #print(input_arr)

      

            for i in range(int((len(input_arr)-96)/prediction_window)):
                inp=input_arr[i*prediction_window:i*prediction_window+96,:]
                #print(inp.shape)
                df_X_ss = ss.transform(inp)
        
                data_predict =one_step_forecast(lstm,df_X_ss)
                
                #print(data_predict)
                #print(mm.inverse_transform(data_predict))
                predict=mm.inverse_transform(data_predict.reshape(-1,1))
                
                for j in range(len(predict)):

                    input_arr[i*prediction_window+96+j][len(input_arr[0])-1]=predict[j]
                    df.loc[96+i*prediction_window+j,predicted_value] = predict[j]
           # input_arr[i+96][len(input_arr[0])-1]=mm.inverse_transform(data_predict.reshape(-1,1))
            #df.loc[96+i,predicted_value] = mm.inverse_transform(data_predict.reshape(-1,1))
        #print(input_arr[96:len(input_arr)-1])
        #output=np.concatenate((output, )
    
        #o=input_arr[96:len(input_arr)-1,len(input_arr[0])-1]
        #output.append(o)
    koef=1
    if(resolution=='240'):
        df=df.iloc[3::4, :]
        koef=0.25
    if(resolution=='15'):
        koef=4
        
        df_copy = pd.DataFrame().reindex_like(df)
        df_copy.reset_index(drop=True)
        df_copy=df_copy.dropna()
        lst15 =  [None] * 34
        lst30 = [None] * 34
        lst45 = [None] * 34
        lst_previous = [None] * 34
        for row in df.itertuples():
            i=0;
            if(row.Index>0):
                
                # diff=pd.to_datetime( df1.loc[index].at["Date"])+ pd.Timedelta(minutes=15)
                # df.loc[len(df.index)] = ['Amy', 89, 93] 
                # print(diff)
                df_copy.loc[len(df_copy)] = lst_previous
                for (columnName, columnData) in df.loc[row.Index].items():
                    if(i==0):
                        
                        lst15[0]=pd.to_datetime( lst_previous[i])+ pd.Timedelta(minutes=15)
                        lst30[0]=pd.to_datetime( lst_previous[i])+ pd.Timedelta(minutes=30)
                        lst45[0]=pd.to_datetime(lst_previous[i])+ pd.Timedelta(minutes=45)
                        
                    else:
                        
                        lst15[i]=lst_previous[i] + (columnData-lst_previous[i])/4
                        lst30[i]=lst_previous[i] + (columnData-lst_previous[i])/2
                        lst45[i]=lst_previous[i] + (columnData-lst_previous[i])*3/4
                        # print(i,lst15[i],lst30[i],lst45[i],lst_previous[i],columnData)
                    lst_previous[i]=columnData
                    
                    i=i+1;
                # print(lst_previous)
                # print(lst15)
                # print(df_copy)
                # df_copy.append(pd.DataFrame([lst15],columns=list(df_copy)),ignore_index=True)
                
                df_copy.loc[len(df_copy)] = lst15
                df_copy.loc[len(df_copy)] = lst30
                df_copy.loc[len(df_copy)] = lst45
              
                
                # df_copy.loc[len(df_copy)]=[lst15]
                # print(lst15)
                # df_copy.loc[len(df.index)] = lst15
                # df_copy.loc[len(df.index)] =lst30
                # df_copy.loc[len(df.index)] = lst45    
                # print(row['Date'], row['VauDejes_temp'])
            else:
                
                for (_, columnData) in df.loc[row.Index].items():
                    # print(columnData)
                    lst_previous[i]=columnData
                    i=i+1;
        
        df=df_copy
    # df=df[96*previous_time_forecast:int(days_range)*24)*koef]
    df=df.head(int((96+int(days_range)*24)*koef))
    
    df['Date']=pd.to_datetime(df['Date'], infer_datetime_format='%Y-%m-%d %H:%M:%S')

        #end zeros  
    if(not df.empty):    
        while(df['Inflows'].iloc[-1]<0.1):
            df=df.drop(df.tail(1).index)

    if(resolution=='1440'):
        df = df.groupby([df['Date'].dt.date]).mean()
        
        df['Date'] = df['Date'].apply(lambda x:x.replace(hour=0,minute=0))  #add 12 hours 
       #  df['Date']=pd.to_datetime(df['Date'], infer_datetime_format='%Y-%m-%d 12:00:00')
    # df['Date']=pd.to_datetime(df['Date'], infer_datetime_format='%d/%m/%Y %H:%M:%S')

    if(resolution=='60'):
        a= savgol_filter(df['Inflows'].values, 7, 1)
        df['Inflows']=pd.Series(a)
        pass
       

    df.to_csv("AllDetails.csv", index=False) 
   
    if(archive_forecast.lower()=='true'):
        df.to_csv(locationOfArchiveCsvPrediction+"\\forecast_"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv", index=False) 
    # print(output)


evaluation()

#data_predict_transform = ss.inverse_transform(input_arr)
