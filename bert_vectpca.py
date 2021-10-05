!pip install tensorboardX
from google.colab import drive
drive.mount('/content/drive')
from tensorboardX import SummaryWriter 
writer = SummaryWriter(logdir='/content/drive/MyDrive/data/bert_block/log')

import pandas as pd
#train=pd.read_csv("/content/drive/MyDrive/data/bert/write.csv") #パースなし
#train=pd.read_csv("/content/drive/MyDrive/data/bert_parse/write.csv")
#train=pd.read_csv("/content/drive/MyDrive/data/bert_nolicence/write.csv")
#train=pd.read_csv("/content/drive/MyDrive/data/bert_block/write_vecter.csv") #4096次元のベクトル
train=pd.read_csv("/content/drive/MyDrive/data/bert_block/write_vecter_pca.csv")

train

history=pd.read_csv("/content/drive/MyDrive/data/bert_block/history.csv")
train["y"]=history["history_num"]

train
train.corr()

# 必要なカラムを取り出してdatXを作る。
select_cols = ["0","7","8","9","10"]
X =train[select_cols]

#範囲指定して必要なカラムを取り出すとき
#X=train.iloc[:,0:14]

#X["flag"]=[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#X["flag"]=[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
X["flag"]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]

X

# Xにおいてtrainフラグが1のものをtrainXとする。その後、trainフラグを削除
trainX = X[X["flag"]==0]
trainX = trainX.drop(columns="flag")
trainY= train["y"][X["flag"]==0]

# datXにおいてtrainフラグが0のものをtestXとする。その後、trainフラグを削除
testX = X[X["flag"]==1]
testX = testX.drop(columns="flag")
testY= train["y"][X["flag"]==1]

trainX

import math
import torch
import torch.nn as nn

class ML_loss(nn.Module):
  def __inif__(self):
    super().__init__()

  def forward(self,outputs,targets):
    sum=0
    for i in range(len(outputs)):
      sum=sum+(targets[i]*(torch.log(outputs[i,0]))-(outputs[i,0]))   

    loss=(-1)*sum
    return loss

def rmse2(outputs,y):
  sum=0
  for i in range(len(outputs)):
    sum=sum+(outputs[i,0]-y.iloc[i])**2
    
  mse=sum/len(outputs)
  mse=mse.clone().detach().numpy()
  rmse=np.sqrt(mse)
  return rmse

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error


#  定数定義
EPOCH = 50
LEARNING_LATE = 0.01

"""
class EarlyStopping:
    #earlystoppingクラス

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        #引数：最小値の非更新数カウンタ、表示設定、モデル格納path

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        
        #特殊(call)メソッド
        #実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する
"""


#1
class practiceNet(nn.Module):

#2
  def __init__(self):
    super(practiceNet, self).__init__()
    self.fc1 = nn.Linear(5,10)
    self.bn1 = nn.BatchNorm1d(10) #バッチ正規化
    #self.dropout1 = torch.nn.Dropout(p=0.2)
    self.fc2 = nn.Linear(10,1)
    #self.dropout2 = torch.nn.Dropout(p=0.5)
    #self.fc3 = nn.Linear(50,25)
    #self.fc4 = nn.Linear(25, 1)

#3
  def forward(self, x):

    x = F.relu(self.fc1(x))
    #x = self.dropout1(x) 
    x = self.bn1(x) #バッチ正規化を行う
    #x = F.relu(self.fc2(x))
    #x = self.dropout2(x) 
    #x = F.relu(self.fc3(x))
    x = torch.exp(self.fc2(x))
    return x



net=practiceNet()
# optimezer定義
optimizer = optim.Adam(net.parameters(), lr=LEARNING_LATE)
# loss関数定義
criterion = ML_loss()
#criterion = nn.MSELoss()


X_train=[]
for i in range(len(trainX)):
  X_train.append(trainX.iloc[i])

X_test=[]
for i in range(len(testX)):
  X_test.append(testX.iloc[i])

y_train=[]
for i in range(len(trainY)):
  y_train.append(trainY.iloc[i])

x = torch.tensor(X_train,dtype=torch.float)
x_test=torch.tensor(X_test, dtype = torch.float)
y=torch.tensor(y_train,dtype=torch.float)


#torch.utils.tensorboardを使用した場合
import torch.utils.tensorboard
with torch.utils.tensorboard.SummaryWriter(log_dir='/content/drive/MyDrive/data/bert_block/log/model_2') as w:
  w.add_graph(net,x)

writer.close()

#TensorBoard notebook の読み込み（一度でOK）
%load_ext tensorboard
 
 
#TensorBoard起動（表示したいログディレクトリを指定）
%tensorboard --logdir='/content/drive/MyDrive/data/bert_block/log/model_2'

writer = SummaryWriter(logdir='/content/drive/MyDrive/data/bert_block/log/train')

running_loss = 0.0

#★EarlyStoppingクラスのインスタンス化★
#earlystopping = EarlyStopping(patience=1, verbose=True) #検証なのでわざとpatience=1回にしている


for i in range(EPOCH):
  # 勾配初期化
  optimizer.zero_grad()
  output = net(x)
  outputs=net(x_test)
  #print(output)
  # loss算出
  loss = criterion(output, y)
  loss.backward()
  optimizer.step()
  running_loss += loss.item()


  rmse_train=rmse2(output,trainY)
  rmse_test=rmse2(outputs,testY)


  print('epoch: {},'.format(i) + 'loss: {:.10f},'.format(loss)+ "rmse_train: {:.1f},".format(rmse_train)+ "rmse_test: {:.1f}".format(rmse_test))
  writer.add_scalar('training_loss',running_loss,i)
  writer.add_scalar('rmse/train', rmse_train, i)
  writer.add_scalar('rmse/test', rmse_test, i)


  #★毎エポックearlystoppingの判定をさせる★
  #earlystopping(running_loss , net) #callメソッド呼び出し
  #if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
  #  print("Early Stopping!")
  #  break
      
  running_loss = 0.0


# 精度計算
outputs = net(torch.tensor(X_test, dtype = torch.float))
r=rmse2(outputs,testY)

#モデルの保存

model_save_path = "/content/drive/MyDrive/data/bert_block/model1_full.pt"
torch.save(net.state_dict(), model_save_path)
net.load_state_dict(torch.load(model_save_path))
writer.close()
#TensorBoard notebook の読み込み（一度でOK）
%load_ext tensorboard
 
 
#TensorBoard起動（表示したいログディレクトリを指定）
%tensorboard --logdir='/content/drive/MyDrive/data/bert_block/log/train'