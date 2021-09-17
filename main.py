import torch
from transformers import BertTokenizer, AlbertConfig, AlbertModel
import numpy as np
from torch.utils import data
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os
from sklearn.metrics import classification_report

# building model network structure
class SentimentClassfier(torch.nn.Module):
    def __init__(self, bert_model, bert_config, num_class):
        super(SentimentClassfier,self).__init__()
        self.bert_model=bert_model
        self.dropout=torch.nn.Dropout(0.2)
        self.fc1=torch.nn.Linear(bert_config.hidden_size,64)
        self.fc2=torch.nn.Linear(64,num_class)
        self.relu=torch.nn.ReLU()
        #self.softmax = torch.nn.LogSoftmax(dim=1) # 加不加没有影响torch.nn.CrossEntropyLoss() = LogSoftmax + NLLLoss
    def forward(self, input_ids, attn_masks, token_type_ids):
        bert_out=self.bert_model(input_ids, attn_masks, token_type_ids)[1] #Sentence vector [batch_size,hidden_size]
        bert_out=self.fc1(bert_out) 
        bert_out=self.relu(bert_out)
        bert_out=self.dropout(bert_out)
        bert_out=self.fc2(bert_out) #[batch_size,num_class]
        # bert_out=self.softmax(bert_out)
        return bert_out

# preparation of training data and validation data
def get_train_test_data(pos_file_path, neg_file_path, max_length=50, test_size=0.3):
    LT = torch.LongTensor
    input_ids, attn_masks, token_type_ids, labels = [],[],[],[]

    pos_df=pd.read_csv(pos_file_path, header=None)
    pos_df.columns=['content']
    for index, row in pos_df.iterrows():
        row=row['content']
        tokenized_text=tokenizer(row.strip(), max_length=max_length, padding='max_length', truncation=True)
        input_ids.append(tokenized_text['input_ids'])
        attn_masks.append(tokenized_text['attention_mask'])
        token_type_ids.append(tokenized_text['token_type_ids'])
        labels.append(1)

    neg_df=pd.read_csv(neg_file_path,header=None)
    neg_df.columns=['content']
    for index, row in neg_df.iterrows():
        row=row['content']
        tokenized_text=tokenizer(row.strip(), max_length=max_length, padding='max_length', truncation=True)
        input_ids.append(tokenized_text['input_ids'])
        attn_masks.append(tokenized_text['attention_mask'])
        token_type_ids.append(tokenized_text['token_type_ids'])
        labels.append(0)
    
    input_ids_tensor = LT(input_ids)
    attn_masks_tensor = LT(attn_masks)
    token_type_ids_tensor = LT(token_type_ids)
    labels_tensor = LT(labels)
    dt = TensorDataset(input_ids_tensor, attn_masks_tensor, token_type_ids_tensor, labels_tensor)
    train_dt, val_dt = train_test_split(dt,test_size=test_size,shuffle=True)
    return train_dt, val_dt

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

    # print(outputs[0].shape,outputs[1].shape)
    # print("="*10)
    
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
    
    # print(type(model.children))
    # for param in model.parameters():
    #     param.requires_grad = False

    # print(" * "*10)
    # print(model)
    # print(" * "*10)

    model.pooler.weight.requires_grad = True
    model.pooler.bias.requires_grad = True
    
    # print out the status of the parameters in each layer
    for cnme,child in model.named_children():
        print("--"*10)
        print(cnme,":",child)
        for nme,param in child.named_parameters():
            print(nme,":",param.requires_grad)    

    sentiment_cls=SentimentClassfier(model,config,2)
    device=torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

    print("0-"*10)
    sentiment_cls=sentiment_cls.to(device)
    print("1-"*10)
    pos_file_path="./input/data01/zmxw.txt"
    neg_file_path="./input/data01/fmxw.txt"
    train_dataset,test_dataset = get_train_test_data(pos_file_path, neg_file_path)
    print(len(train_dataset),len(test_dataset))

    train_dataloader = data.DataLoader(train_dataset, batch_size=32)
    val_dataloader = data.DataLoader(test_dataset, batch_size=32)

    # Define optimizer and loss function
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(sentiment_cls.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-4)
    # Model training and testing

    for epoch in range(100):
        train_label,train_pred,val_label,val_pred = [],[],[],[]
        loss_sum=0.0
        accu=0
        # set to training mode
        sentiment_cls.train()
        for step, batch in enumerate(train_dataloader):
            token_ids, attn_mask, segment_mask, labels = batch
            inputs = {'input_ids' : token_ids.to(device), 
                      'attn_masks' : attn_mask.to(device), 
                      'token_type_ids': segment_mask.to(device)} 
            labels=labels.to(device)
            out=sentiment_cls(**inputs)
            loss=criterion(out,labels)
            optimizer.zero_grad()
            loss.backward() #Back propagation
            optimizer.step() #Gradient update
            loss_sum+=loss.cpu().data.numpy()
            accu+=(out.argmax(1)==labels).sum().cpu().data.numpy()
            train_pred.extend(out.argmax(1).cpu()) 
            train_label.extend(labels.cpu()) 
        test_loss_sum=0.0
        test_accu=0
        # set to evaluation mode
        sentiment_cls.eval()
        for step,batch in enumerate(val_dataloader):
            token_ids, attn_mask, segment_mask, labels = batch
            inputs = {'input_ids' : token_ids.to(device), 
                      'attn_masks' : attn_mask.to(device), 
                      'token_type_ids': segment_mask.to(device)} 
            labels=labels.to(device)
            with torch.no_grad():
                out=sentiment_cls(**inputs)
                loss=criterion(out,labels)
                test_loss_sum+=loss.cpu().data.numpy()
                test_accu+=(out.argmax(1)==labels).sum().cpu().data.numpy()
                val_pred.extend(out.argmax(1).cpu()) 
                val_label.extend(labels.cpu()) 
        print("epoch % d,train loss:%f,train acc:%f,val loss:%f,val acc:%f"%(epoch,loss_sum/len(train_dataset),accu/len(train_dataset),test_loss_sum/len(test_dataset),test_accu/len(test_dataset)))
        print(classification_report(train_label, train_pred, labels=[0,1], target_names=['negative','positive']))
        print(classification_report(val_label, val_pred, labels=[0,1], target_names=['negative','positive']))
    
