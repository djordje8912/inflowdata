
#evaluation on one input vector
import pandas as pd
import numpy as np  
import torch 
from torch import nn
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#from torch.utils.data.dataset import Dataset
#from torch.utils.data import random_split
#from torch import split
#from datetime import datetime
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
    df = pd.read_csv('C:\\development\\InflowForecast\\InflowForecast\\bin\\Debug\\net6.0-windows\\input.csv')

    df1= pd.DataFrame(df, columns=["Kukes_rain","Kukes_temp","Kukes_humi","Okshtun_rain","Okshtun_temp","Okshtun_humi","Fierze_rain","Fierze_temp","Fierze_humi","Zogaj_temp","Zogaj_rain","Peshkopi_temp","Peshkopi_rain","Inflows"])
        
    df1.reset_index(drop=True)
    print(df1.head())
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




    X=[X1,X2,X3]
    json_name=["20231005060846","20231005004415","20231004234606"]
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
        prediction_window_extended=24*7
        if(predicted_value=="Inflows"):
            prediction_index=0
        elif(predicted_value=="InflowsTributary"):
            prediction_index=1
        else:
            prediction_index=2


        model_name = 'models/model_'+json_name[k]+'.pth'
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

  

        for i in range(len(input_arr)-96):
            inp=input_arr[i:i+96,:]
            #print(inp.shape)
            df_X_ss = ss.transform(inp)
    
            data_predict =one_step_forecast(lstm,df_X_ss)
            #print(data_predict)
            #print(mm.inverse_transform(data_predict))
          
            input_arr[i+96][len(input_arr[0])-1]=mm.inverse_transform(data_predict.reshape(-1,1))
            df.loc[96+i,predicted_value] = mm.inverse_transform(data_predict.reshape(-1,1))
        #print(input_arr[96:len(input_arr)-1])
        #output=np.concatenate((output, )
    
        #o=input_arr[96:len(input_arr)-1,len(input_arr[0])-1]
        #output.append(o)
 
    df.to_csv("AllDetails.csv", index=False) 
    print(output)


evaluation()

#data_predict_transform = ss.inverse_transform(input_arr)
