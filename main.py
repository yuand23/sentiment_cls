import torch
from transformers import BertTokenizer, AlbertConfig, AlbertModel
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os


# building model network structure
class SentimentClassfier(torch.nn.Module):
    def __init__(self,bert_model,bert_config,num_class):
        super(SentimentClassfier,self).__init__()
        self.bert_model=bert_model
        self.dropout=torch.nn.Dropout(0.2)
        self.fc1=torch.nn.Linear(bert_config.hidden_size,64)
        self.fc2=torch.nn.Linear(64,num_class)
        self.relu=torch.nn.ReLU()
        #self.softmax = torch.nn.LogSoftmax(dim=1) 加不加没有影响torch.nn.CrossEntropyLoss() = LogSoftmax + NLLLoss
    def forward(self,token_ids):
        bert_out=self.bert_model(token_ids)[1] #Sentence vector [batch_size,hidden_size]
        bert_out=self.fc1(bert_out) 
        bert_out=self.relu(bert_out)
        bert_out=self.dropout(bert_out)
        bert_out=self.fc2(bert_out) #[batch_size,num_class]
        bert_out=self.softmax(bert_out)
        return bert_out

# preparation of training data and validation data
def get_train_test_data(pos_file_path,neg_file_path,max_length=50,test_size=0.3):
    data=[]
    label=[]
    pos_df=pd.read_csv(pos_file_path,header=None)
    pos_df.columns=['content']
    for index, row in pos_df.iterrows():
        row=row['content']
        ids=tokenizer.encode(row.strip(),max_length=max_length,padding='max_length',truncation=True)
        data.append(ids)
        label.append(1)
    
    # neu_df=pd.read_csv(neu_file_path,header=None)
    # neu_df.columns=['content']
    # for index, row in neu_df.iterrows():
    #     row=row['content']
    #     ids=tokenizer.encode(row.strip(),max_length=max_length,padding='max_length',truncation=True)
    #     data.append(ids)
    #     label.append(1)

    neg_df=pd.read_csv(neg_file_path,header=None)
    neg_df.columns=['content']
    for index, row in neg_df.iterrows():
        row=row['content']
        ids=tokenizer.encode(row.strip(),max_length=max_length,padding='max_length',truncation=True)
        data.append(ids)
        label.append(0)
    X_train, X_test, y_train, y_test=train_test_split(data,label,test_size=test_size,shuffle=True)
    return (X_train,y_train),(X_test,y_test)

class DataGen(data.Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)           
    def __getitem__(self,index):
        return np.array(self.data[index]),np.array(self.label[index])

if __name__ == '__main__':
    # pretrained = 'voidful/albert_chinese_tiny'  #Use small version of Albert
    # pretrained = './models/albert_chinese_tiny' 
    # pretrained = 'clue/albert_chinese_small'
    pretrained = './models/albert_chinese_small' 
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model=AlbertModel.from_pretrained(pretrained)
    config=AlbertConfig.from_pretrained(pretrained)
    
    print(config.hidden_size,config.embedding_size,config.max_length)

    inputtext = "I'm in a good mood today. I bought a lot of things. I like it very much. I finally have my favorite electronic products. I can study hard this time"
    tokenized_text=tokenizer.encode(inputtext)
    input_ids=torch.tensor(tokenized_text).view(-1,len(tokenized_text))
    outputs=model(input_ids)

    print(outputs[0].shape,outputs[1].shape)
    print("="*10)
    
    # freeze all the parameters
    # print(type(model.children))
    # for child in model.children():
    #     print("--"*10)
    #     print(child)
    #     for param in child.parameters():
    #         print("=="*10)
    #         print(param.requires_grad)
    #         param.requires_grad = False
    # print("*"*10)
    # print(model)
    
    print(type(model.children))
    for param in model.parameters():
        param.requires_grad = False

    print(" * "*10)
    print(model)
    print(" * "*10)

    model.pooler.weight.requires_grad = True
    model.pooler.bias.requires_grad = True
    
    # print out the status of the parameters in each layer
    for cnme,child in model.named_children():
        print("--"*10)
        print(cnme,":",child)
        for nme,param in child.named_parameters():
            print(nme,":",param.requires_grad)
    
    # sys.exit()

    sentiment_cls=SentimentClassfier(model,config,2)
    device=torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

    # print(sentiment_cls)
    # device='cpu'
    # sys.exit()
    print("0-"*10)
    sentiment_cls=sentiment_cls.to(device)
    # sys.exit()
    print("1-"*10)
    pos_file_path="./input/data01/zmxw.txt"
    neg_file_path="./input/data01/fmxw.txt"
    (X_train,y_train),(X_test,y_test)=get_train_test_data(pos_file_path,neg_file_path)
    print(len(X_train),len(X_test),len(y_train),len(y_test),len(X_train[0]))

    train_dataset=DataGen(X_train,y_train)
    test_dataset=DataGen(X_test,y_test)
    print("2-"*10)
    train_dataloader=data.DataLoader(train_dataset,batch_size=32)
    test_dataloader=data.DataLoader(test_dataset,batch_size=32)

    # Define optimizer and loss function
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(sentiment_cls.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-4)
    print("-"*10)
    # Model training and testing
    for epoch in range(100):
        loss_sum=0.0
        accu=0
        # set to training mode
        sentiment_cls.train()
        for step,(token_ids,label) in enumerate(train_dataloader):
            token_ids=token_ids.to(device)
            label=label.to(device)
            out=sentiment_cls(token_ids)
            loss=criterion(out,label)
            optimizer.zero_grad()
            loss.backward() #Back propagation
            optimizer.step() #Gradient update
            loss_sum+=loss.cpu().data.numpy()
            accu+=(out.argmax(1)==label).sum().cpu().data.numpy()
        test_loss_sum=0.0
        test_accu=0
        # set to evaluation mode
        sentiment_cls.eval()
        for step,(token_ids,label) in enumerate(test_dataloader):
            token_ids=token_ids.to(device)
            label=label.to(device)
            with torch.no_grad():
                out=sentiment_cls(token_ids)
                loss=criterion(out,label)
                test_loss_sum+=loss.cpu().data.numpy()
                test_accu+=(out.argmax(1)==label).sum().cpu().data.numpy()
        print("epoch % d,train loss:%f,train acc:%f,test loss:%f,test acc:%f"%(epoch,loss_sum/len(train_dataset),accu/len(train_dataset),test_loss_sum/len(test_dataset),test_accu/len(test_dataset)))
    
