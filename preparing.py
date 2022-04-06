#from scipy.misc import imread, imresize
import sys
import tensorflow as tf
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from tensorflow.keras import layers,Input,models
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import cv2 as cv

import gc
from tensorflow.keras import backend as bek
fault_mapping = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
waferSizeX=49
waferSizeY=49
def core_select(select):
    bek.clear_session()
    gc.collect()
    if select == 'GPU':
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        tf.config.list_physical_devices('GPU')
    else:
        tf.config.list_physical_devices('CPU')
    return

goodWafer=np.array(
[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
)

#================================================Files Functions
#==================
def savefile(name,xdata,ydata):
#==================   
    np.savez_compressed('./data/%s'%name, a=xdata, b=ydata)
#==================    
def loadfile(name):
#==================   
    loaded = np.load('./data/%s.npz'%name)
    return loaded['a'] , loaded['b']

#================================================Plot Functions
#==================
def acc_plot(title_a,title_b,savename,mhistory):
#==================   
    #transfer to %
    acc_list=[]
    val_acc_list=[]
    loss_list=[]
    val_loss_list=[]
    for i in mhistory.history['acc']:
        acc_list.append(i*100)
    for i in mhistory.history['val_acc']:
        val_acc_list.append(i*100)
    for i in mhistory.history['loss']:
        loss_list.append(i*100)
    for i in mhistory.history['val_loss']:
        val_loss_list.append(i*100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    #fig.suptitle('A tale of 2 subplots')
    ax1.plot(acc_list)
    ax1.plot(val_acc_list)
    ax1.set_title(title_a,fontweight="bold", size=20)
    ax1.set_ylabel('accuracy %',fontsize=12)
    ax1.set_xlabel('epoch',fontsize=12)
    ax1.legend(['train', 'test'], loc='upper left')
    ax1.set_ylim([0,110])
    ax1.autoscalex_on=0
    ax1.set_yticks(list(range(0,110,10)))
    ax1.set_xticks(list(range(0,len(mhistory.history['acc']),1)))
    ax1.set_xticklabels(list(range(1,len(mhistory.history['acc'])+1,1)))
    ax1.grid()
    
    ax2.plot(loss_list)
    ax2.plot(val_loss_list)
    ax2.set_title(title_b,fontweight="bold", size=20)
    ax2.set_ylabel('loss %',fontsize=12)
    ax2.set_xlabel('epoch',fontsize=13)
    ax2.legend(['train', 'test'], loc='upper right')
    ax2.set_ylim([0,110])
    ax2.autoscalex_on=0
    ax2.set_yticks(list(range(0,110,10)))
    ax2.set_xticks(list(range(0,len(mhistory.history['val_acc']),1)))
    ax2.set_xticklabels(list(range(1,len(mhistory.history['val_acc'])+1,1)))
    ax2.grid()
    
    fig.savefig('./report_chart/%s.jpg'%savename, bbox_inches='tight', dpi=150)
    plt.show()
    return

#==================
def failcount_plot(y,y_limit,title):
#==================
    mapping=['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
    y_axis=[]
    for f in range(8):
        y_axis.append(len(y[y==f]))
    x_axis = np.arange(8)
    plt.figure(figsize=(10,3),dpi=80)
    plt.bar(x_axis,y_axis,align='center',color='c',alpha=0.8)
    plt.xticks(x_axis, mapping,rotation=30)
    for a,b in zip(x_axis,y_axis):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
    plt.ylim(0,y_limit)
    plt.xlabel('Fail Mode')
    plt.ylabel('Fail Count')
    plt.title(title)
#    plt.title('Final Type Count for %sX%s'%(waferSizeX,waferSizeY))
    plt.show()
#==================
def failtype_plot(image,each_title):
#==================
    #ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84, 'Near-full': 37}
    fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(10, 10))
    ax = ax.ravel(order='C')
    for i in range(8):
        ax[i].imshow(image[i],cmap=plt.cm.bone)
        ax[i].set_title(each_title[i],fontsize=24)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show() 
#==================
def sub_plot(image):
#==================
    #ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84, 'Near-full': 37}
    fig, ax = plt.subplots(nrows = int(len(image)**0.5), ncols = int(len(image)**0.5), figsize=(10, 10))
    ax = ax.ravel(order='C')
    for i in range(len(image)):
        ax[i].imshow(image[i],cmap=plt.cm.bone)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show() 
#==================
def fail_plot(x,y,XbyY,f):
#==================
    image=x[y==f]
    print(fault_mapping[f])
    image=image[:XbyY]
    sub_plot(image)
#==================
def show_pattern_cmp(org_x,new_x,title,y):
#==================
#Show the fail pattern bewfore/after modified for 8 kind of fail

    cnt = 0
    fig , ax = plt.subplots(nrows = 4, ncols=4, figsize=(10,10))
    ax = ax.ravel(order='C')
    i,j,pic_index=0,0,0
    for i in range(len(fault_mapping)):
        for j in range (len(y)):
            if y[j]==i:
                ax[pic_index].imshow(org_x[j],cmap=plt.cm.bone)
                ax[pic_index].set_title('org %s %s'%(title,fault_mapping[i]))
                ax[pic_index].set_xticks([])
                ax[pic_index].set_yticks([])  
                pic_index+=1
                ax[pic_index].imshow(new_x[j],cmap=plt.cm.bone)
                ax[pic_index].set_title('new %s %s'%(title,fault_mapping[i]))
                ax[pic_index].set_xticks([])
                ax[pic_index].set_yticks([])  
                pic_index+=1 
                break
    plt.tight_layout()
    plt.show()
#==================
def show100map(x,y,fail_name,pic_index):
#==================
    dic = {'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'None':8}
    show_list=np.where(y==dic[fail_name])
    fig,ax=plt.subplots(nrows=10,ncols=10,figsize=(15,15))
    ax=ax.ravel(order='C')
    for index in range(0,100):
        ax[index].imshow(x[show_list[0][pic_index]],cmap=plt.cm.bone)
        ax[index].set_title(show_list[0][pic_index],color='white')
        ax[index].set_xticks([])
        ax[index].set_yticks([])
        pic_index+=1
        if index+1>=len(show_list[0]):
            break
    plt.tight_layout()
    plt.show()
#================== 
#DBSCAN
#================== 
def img_to_coord(img):
    result=[]
    for i in range (waferSizeX):
        for j in range(waferSizeY):

            if img[i][j]==1:
                img[i][j]=0
            if img[i][j]==2:
                temp=[i,j]
                result.append(temp)
    return np.array(result)

def coord_to_img(coord,fill_one):
    if fill_one==True:
        result=np.copy(goodWafer)  #Fill good as 1
    else:
        result=np.zeros((waferSizeX,waferSizeY)) #No fill good as 1
    for i in range(len(coord)):
        result[coord[i][0]][coord[i][1]]=2
    return result

def dbscan_denoize(img,fill_one,max_near_dots,min_near_dots,threashold_dots):
    #eps:兩個樣本之間的最大距離
    #min_samples:將一個點視為核心點的鄰域中的樣本數
    #fill one : Good die fill 1
    #max_near_dots : 鄰近dot 最大點數
    #min_near_dots : 鄰近dot 最小點數
    #threashold_dots: 整張圖,壞 die 點數
    data=img_to_coord(img)
    for neighbor_dots in range (max_near_dots,min_near_dots,-1):
        db = DBSCAN(eps=2, min_samples=neighbor_dots, metric = 'euclidean',algorithm ='auto')
        db.fit(data)
        #print(len(db.components_))
        if len(db.components_) > threashold_dots:
            break
    return(coord_to_img(db.components_,fill_one))

#================== 
#Image Generator
#================== 

    datagen=ImageDataGenerator(featurewise_center=False,  
                            samplewise_center=False, 
                            featurewise_std_normalization=True, 
                            samplewise_std_normalization=True, 
                            zca_whitening=False, 
                            zca_epsilon=1e-06, 
                            rotation_range=90, 
                            width_shift_range=0.0, 
                            height_shift_range=0.0, 
                            brightness_range=None, 
                            shear_range=0.0, 
                            zoom_range=0.0, 
                            channel_shift_range=0.0, 
                            fill_mode='constant', 
                            cval=0.0, 
                            horizontal_flip=True, 
                            vertical_flip=True, 
                            rescale=None, 
                            preprocessing_function=None, 
                            data_format=None, 
                            validation_split=0.0, 
                            dtype=None)

def batchgen(count,gx,gy):
    datagen=ImageDataGenerator(featurewise_center=False,  
                            samplewise_center=False, 
                            featurewise_std_normalization=True, 
                            samplewise_std_normalization=True, 
                            zca_whitening=False, 
                            zca_epsilon=1e-06, 
                            rotation_range=90, 
                            width_shift_range=0.0, 
                            height_shift_range=0.0, 
                            brightness_range=None, 
                            shear_range=0.0, 
                            zoom_range=0.0, 
                            channel_shift_range=0.0, 
                            fill_mode='constant', 
                            cval=0.0, 
                            horizontal_flip=True, 
                            vertical_flip=True, 
                            rescale=None, 
                            preprocessing_function=None, 
                            data_format=None, 
                            validation_split=0.0, 
                            dtype=None)
    batch_count=0
    x_result = np.zeros((1, waferSizeX, waferSizeY, 3))
    y_result = np.zeros((1,))
    for  x_batch,y_batch in datagen.flow(gx,gy,batch_size = 32 ): # Generate 32 pictures for eatch batch
        batch_count+=1
        x_result = np.concatenate((x_result, x_batch), axis=0)
        y_result = np.concatenate((y_result, y_batch), axis=0)
        if (batch_count*32)>(count+32):
            break
    return x_result[1:],y_result[1:] # Remove the first initial one

def img_generator(newx,newy,limit):
    #limit:how many image need to generator
    datagen=ImageDataGenerator(featurewise_center=False,  
                            samplewise_center=False, 
                            featurewise_std_normalization=True, 
                            samplewise_std_normalization=True, 
                            zca_whitening=False, 
                            zca_epsilon=1e-06, 
                            rotation_range=90, 
                            width_shift_range=0.0, 
                            height_shift_range=0.0, 
                            brightness_range=None, 
                            shear_range=0.0, 
                            zoom_range=0.0, 
                            channel_shift_range=0.0, 
                            fill_mode='constant', 
                            cval=0.0, 
                            horizontal_flip=True, 
                            vertical_flip=True, 
                            rescale=None, 
                            preprocessing_function=None, 
                            data_format=None, 
                            validation_split=0.0, 
                            dtype=None)
    datagen.fit(newx) #透过训练数据集来训练(fit)图像数据增强产生器(ImageDataGenerator)的实例
    #设定要"图像数据增强产生器(ImageDataGenerator)"产生的图像批次值(batch size) 
    # "图像数据增强产生器(ImageDataGenerator)"会根据设定回传指定批次量的新生成图像数据
    gen_x=newx
    gen_y=newy
    for f in range (8):
        if len(newy[newy==f])<limit:
            g_x, g_y = batchgen(limit-len(newy[newy==f]),newx[np.where(newy==f)], newy[np.where(newy==f)])
            gen_x=np.concatenate((gen_x, g_x), axis=0)
            gen_y=np.concatenate((gen_y, g_y), axis=0)
    print(gen_x.shape)
    print(gen_y.shape)
    gen_x=gen_x*254
    gen_x=np.ceil(gen_x)
    for f in range (len(fault_mapping)):
        print('%s-%s : %s'%(f,fault_mapping[f],len(gen_y[gen_y==f]))) 

    image=[]
    each_title=[]
    for i in range(len(fault_mapping)):
        for j in range(gen_y.shape[0]):
            if gen_y[j]==i:
                image.append(gen_x[j])
                print(j,end=' ')
                break
    failtype_plot(image,fault_mapping)
    gen_y=to_categorical(gen_y)
    savefile('gen_fail_pattern%s_%s'%(waferSizeX,waferSizeY),gen_x,gen_y)
    print('File Saved: gen_fail_pattern%s_%s'%(waferSizeX,waferSizeY))
    return

#================== 
#AUTOENCODER
#==================
# augment function define


def autoencoder_model(new_x,y,limit):
    # parameter
    epoch=15
    batch_size=128
    # Encoder
    input_shape = (waferSizeX, waferSizeY, 3)
    input_tensor = Input(input_shape)
    encode = layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_tensor)
    latent_vector = layers.MaxPool2D()(encode)
    # Decoder
    decode_layer_1 = layers.Conv2DTranspose(64, (3,3), padding='same', activation='relu')
    decode_layer_2 = layers.UpSampling2D()
    output_tensor = layers.Conv2DTranspose(3, (3,3), padding='same', activation='sigmoid')
    # connect decoder layers
    decode = decode_layer_1(latent_vector)
    decode = decode_layer_2(decode)
    ae = models.Model(input_tensor, output_tensor(decode))
    ae.compile(optimizer = 'Adam',loss = 'mse')
    ae.summary()
    # start train
    ae.fit(new_x, new_x,batch_size=batch_size,epochs=epoch,verbose=0)
    # Make encoder model with part of autoencoder model layers
    encoder = models.Model(input_tensor, latent_vector)
    # Make decoder model with part of autoencoder model layers
    decoder_input = Input((int(waferSizeX/2), int(waferSizeY/2), 64))
    decode = decode_layer_1(decoder_input)
    decode = decode_layer_2(decode)
    decoder = models.Model(decoder_input, output_tensor(decode))
    # Encode original faulty wafer
    encoded_x = encoder.predict(new_x)
    # Add noise to encoded latent faulty wafers vector.
    noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.3, size = (len(encoded_x), int(waferSizeX/2), int(waferSizeY/2), 64))
    # check original faulty wafer data
    #plt.imshow(np.argmax(new_x[3], axis=2))
    # check new noised faulty wafer data
    noised_gen_x = np.argmax(decoder.predict(noised_encoded_x), axis=3)
    # check reconstructed original faulty wafer data
    gen_x = np.argmax(ae.predict(new_x), axis=3)
    # original faulty wafer data
    org_x=np.argmax(new_x,axis=3)
    index=4
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(10, 10))
    ax = ax.ravel(order='C')
    ax[0].imshow(org_x[index],cmap=plt.cm.bone)
    ax[0].set_title('Original',fontsize=12)
    ax[1].imshow(noised_gen_x[index],cmap=plt.cm.bone)
    ax[1].set_title('Noized',fontsize=12)
    ax[2].imshow(gen_x[index],cmap=plt.cm.bone)
    ax[2].set_title('Reconstructed',fontsize=12)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.tight_layout()
    plt.show()
    
    def gen_data(wafer, label):
        # Encode input wafer
        encoded_x = encoder.predict(wafer)  
        # dummy array for collecting noised wafer
        gen_x = np.zeros((1, int(waferSizeX), int(waferSizeY), 3))
        # Make wafer until total # of wafer to 2000
        for i in range((limit//len(wafer))):
            noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), int(waferSizeX/2), int(waferSizeY/2), 64)) 
            noised_gen_x = decoder.predict(noised_encoded_x)
            gen_x = np.concatenate((gen_x, noised_gen_x), axis=0)
        # also make label vector with same length
        #numpy.full(shape, fill_value, dtype=None, order='C')[source]
        #返回一个根据指定shape和type,并用fill_value填充的新数组。
        gen_y = np.full((len(gen_x), 1), label)
        #gen_y=gen_y.reshape(1,-1)
        # return date without 1st dummy data.
        return gen_x[1:], gen_y[1:]
    # Augmentation for all faulty case.
    new_y=np.copy(y)
    faulty_case=np.unique(y)
    for f in faulty_case : 
        if len(y[y==f]) < limit:
            gen_x, gen_y = gen_data(new_x[np.where(y==f)], f)
            new_x = np.concatenate((new_x, gen_x), axis=0)
            gen_y=gen_y.reshape(1,-1)
            new_y = np.concatenate((new_y, gen_y[0]))

    print('After Generate new_x shape : {}, new_y shape : {}'.format(new_x.shape, new_y.shape))
    for f in faulty_case :
        print('{} : {}'.format(f, len(new_y[new_y==f])))

    # one-hot-encoding
    new_y = to_categorical(new_y)
    savefile('autoencoder_fail_pattern%s_%s'%(waferSizeX,waferSizeY),new_x,new_y) 
    print('File Save:autoencoder_fail_pattern%s_%s'%(waferSizeX,waferSizeY))
    return

#================
#FIND FAIL PATTERN
#================
def FindingFailPattern(x,y):
    #Find 8 fail pattern from fail_pattern x,y (Non one hot encoder)
    #return x[0:8],y[0:8]
    index=[]
    for f in range(8):
        for i in range(len(y)):
            if y[i]==f:
                index.append(i)
                break
    test_x=np.zeros((1, waferSizeX, waferSizeY, 3))
    test_y=np.zeros((1,))
    for i in range(8):
        test_x = np.concatenate((test_x,x[index[i]].reshape(-1,waferSizeX,waferSizeY,3)), axis=0)
        test_y = np.concatenate((test_y, y[index[i]].reshape(-1,)), axis=0)
    return test_x[1:],test_y[1:]
#================
#CV2 IMAGE GENERATOR
#================
def rotate_img(img,angle):
    height = img.shape[0] # 定義圖片的高度
    width = img.shape[1] # 定義圖片的寬度
    center = (int(waferSizeX/2), int(waferSizeY/2)) # 定義圖片的中心
    scale = 1.0 # 指定縮放比例
    # 旋轉
    trans = cv.getRotationMatrix2D(center, angle, scale)
    image = cv.warpAffine(img, trans, (width, height))
    image=np.ceil(image)
    return image
def flip_img(img,flip):
    image=cv.flip(img,flip)
    image=np.ceil(image)
    return image
def cv2_batchgen(count,gx,gy):
    batch_count=0
    x_result = np.zeros((1, waferSizeX, waferSizeY, 3))
    y_result = np.zeros((1,))
    
    for i in range (int(count//7)):
    #generate 11 pics a time
    #1.Mirrow Left 2.Mirrow Up 3,Rotate90 4.Mirrow Left
        a1=rotate_img(gx[i],90) #rotate90
        a2=rotate_img(gx[i],180) #rotate90
        a3=rotate_img(gx[i],270) #rotate90
        a4=flip_img(gx[i],0) #flip left
        a5=flip_img(gx[i],1) #flip up
        a6=flip_img(a1,0)
        a7=flip_img(a1,1)
        x_result = np.concatenate((x_result, a1.reshape(-1,waferSizeX,waferSizeY,3)), axis=0) 
        y_result = np.concatenate((y_result, gy[i].reshape(-1,)), axis=0)
        x_result = np.concatenate((x_result, a2.reshape(-1,waferSizeX,waferSizeY,3)), axis=0) 
        y_result = np.concatenate((y_result, gy[i].reshape(-1,)), axis=0)
        x_result = np.concatenate((x_result, a3.reshape(-1,waferSizeX,waferSizeY,3)), axis=0) 
        y_result = np.concatenate((y_result, gy[i].reshape(-1,)), axis=0)
        x_result = np.concatenate((x_result, a4.reshape(-1,waferSizeX,waferSizeY,3)), axis=0) 
        y_result = np.concatenate((y_result, gy[i].reshape(-1,)), axis=0)
        x_result = np.concatenate((x_result, a5.reshape(-1,waferSizeX,waferSizeY,3)), axis=0) 
        y_result = np.concatenate((y_result, gy[i].reshape(-1,)), axis=0)
        x_result = np.concatenate((x_result, a6.reshape(-1,waferSizeX,waferSizeY,3)), axis=0) 
        y_result = np.concatenate((y_result, gy[i].reshape(-1,)), axis=0)
        x_result = np.concatenate((x_result, a7.reshape(-1,waferSizeX,waferSizeY,3)), axis=0) 
        y_result = np.concatenate((y_result, gy[i].reshape(-1,)), axis=0)
    return x_result[1:],y_result[1:]    

def rebuild_wafer(load_x):
    #input: one hot encoder x
    #add good die shape
    #output: one hot encoder x
    x = np.zeros((len(load_x),waferSizeX , waferSizeY))
    for w in range(len(load_x)):
        for i in range(waferSizeX):
            x[w, i]=load_x[w][i].argmax(1) 
    wafer3=np.where(goodWafer==1,3,goodWafer)
    for i in range(len(x)):
        a=x[i].astype('int')&wafer3.astype('int')
        a=a.astype('int')^goodWafer.astype('int')
        x[i]=np.where(a>2,2,a)
    new_x = np.zeros((len(x),waferSizeX , waferSizeY , 3))
    for w in range(len(x)):
        for i in range(waferSizeX):
            for j in range(waferSizeY):
                new_x[w, i, j, int(x[w, i, j])] = 1
    return new_x

def cv2_img_generator(newx,newy,limit,onehot,rebuildx):
    gen_x=np.copy(newx)
    gen_y=np.copy(newy)
    for f in range (8):
        a=(len(newy[newy==f])*(7+1)) #8 fail pattern + one original
        if limit>a:
            limit_count=a
        else:
            limit_count=limit
        if len(newy[newy==f])<=limit:
            g_x, g_y = cv2_batchgen(limit_count-len(newy[newy==f]),newx[np.where(newy==f)], newy[np.where(newy==f)])
            gen_x=np.concatenate((gen_x, g_x), axis=0)
            gen_y=np.concatenate((gen_y, g_y), axis=0)
    for f in range (8):
        print('%s-%s : %s'%(f,fault_mapping[f],len(gen_y[gen_y==f])))
    print('Total: %s'%(len(gen_y)))
    if rebuildx==True:    
        gen_x=rebuild_wafer(gen_x) #Add good die
    # one-hot-encoding
    if onehot==True:
        gen_y = to_categorical(gen_y)
    else:
        x = np.zeros((len(gen_x),waferSizeX , waferSizeY))
        for w in range(len(gen_x)):
            for i in range(waferSizeX):
                x[w, i]=gen_x[w][i].argmax(1) 
        gen_x = x
    return gen_x,gen_y


def drop_blank(load_file,drop_value,waferamount):
    #If wafer map all 0, drop it
    #If wafer map more than limit count(2000) , drop others
    x,y=loadfile('%s%s_%s'%(load_file,waferSizeX,waferSizeY))
    y=y.astype('int')
    print('ORG Wafer Count:%s'%len(y))
    limit=waferamount
    org_count=len(y)
    drop_count=0
    drop_over_limit=0
    org_x=np.copy(x)
    new_x=np.ones((1,waferSizeX,waferSizeY))
    new_y=np.ones((1))
    limit_array=[0,0,0,0,0,0,0,0]
    for i in range (len(y)):
#    for i in range (100):
        zero_count=0
        for j in range(waferSizeX):
            for k in range(waferSizeY):
                if x[i][j][k]==0:
                    zero_count+=1
        if zero_count >= int(((waferSizeX*waferSizeX)*drop_value)): 
            drop_count+=1
        else:
            #temp=temp.reshape(-1,waferSizeX,waferSizeY)
            limit_array[y[i]]+=1
            if limit_array[y[i]] <= limit:
                new_x=np.concatenate((new_x,x[i].reshape(-1,waferSizeX,waferSizeY)),axis=0)
                new_y=np.concatenate((new_y,y[i].reshape(-1,)))
            else:
                drop_over_limit+=1
        if i%1000 == 999:
            print('\r %s/%s'%(i+1,org_count),end=' ')
    new_x=new_x[1:]
    new_y=new_y[1:]
    print('\r %s/%s'%(i+1,org_count))
    print('New Wafer Count:%s'%len(new_y))
    print('Drop Blank Count: %s, drop rate: %.2f%%'%(drop_count,(drop_count/org_count)*100))
    print('Drop Over %s Count: %s, drop rate: %.2f%%'%(limit,drop_over_limit,(drop_over_limit/org_count)*100))
    #savefile('fail_pattern_dbscan_deblank%s_%s'%(waferSizeX,waferSizeY),new_x,new_y)
    for i in range(8):
        print('%s %s %s'%(i,fault_mapping[i],len(new_y[new_y==i])))
    #print('Save to :fail_pattern_dbscan_deblank%s_%s'%(waferSizeX,waferSizeY))
    return new_x,new_y

#=============
def OnehotX(x):
    new_x = np.zeros((len(x),waferSizeX , waferSizeY , 3))
    for w in range(len(x)):
        for i in range(waferSizeX):
            for j in range(waferSizeY):
                if x[w][i][j]<2:
                    new_x[w, i, j, int(x[w, i, j])] = 0
                else:
                    new_x[w, i, j, int(x[w, i, j])] = 1
    return new_x

#=============
def ResNet18(self):
	input = Input(self.shape)
	x = ZeroPadding2D((3,3))(input)
	x = Conv2D(64,(7,7),strides=2)(x)
	x = BatchNormalization(axis = 3)(x)
	x = Activation('relu')(x)
	x = MaxPool2D(pool_size=(3,3),strides = (2,2),padding='same')(x)
	x = self.basic_block(x,64,1,name='shortcut1')
	x = self.basic_block(x,64,1,name='shortcut2')
	x = self.basic_block(x, 128, 2,name='shortcut3')
	x = self.basic_block(x, 128, 1,name='shortcut4')
	x = self.basic_block(x, 256, 2,name='shortcut5')
	x = self.basic_block(x, 256, 1,name='shortcut6')
	x = self.basic_block(x, 512, 2,name='shortcut7')
	x = self.basic_block(x, 512, 1,name='shortcut8')
	size = int(x.shape[1])
	x = AveragePooling2D(pool_size=(size,size))(x)
	x = Flatten()(x)
	x = Dense(8,activation='softmax')(x)
	model = Model(inputs = input,outputs=x)
	model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
	return model

def ResNet50(self):
	input = Input(self.shape)
	x = ZeroPadding2D((3,3))(input)
	x = Conv2D(64,(7,7),strides=(2,2))(x)
	x = BatchNormalization(axis = 3)(x)
	x = Activation('relu')(x)
	x = MaxPool2D(pool_size =(3,3),strides=(2,2),padding='same')(x)
	x = self.convolutional_block(x,[64,64,256],stride=1)
	x = self.identity_block(x,[64,64,256])
	x = self.convolutional_block(x,[128,128,512],stride=1)
	x = self.identity_block(x,[128,128,512])
	x = self.convolutional_block(x,[256,256,1024],stride=2)
	x = self.identity_block(x,[256,256,1024])
	x = self.convolutional_block(x,[512,512,2048],stride=2 )
	x = self.identity_block(x,[512,512,2048])
	size = int(x.shape[1])
	x = AveragePooling2D(pool_size=(size, size))(x)
	x = Flatten()(x)
	x = Dense(8,activation='softmax')(x)
	model = Model(inputs = input,outputs= x)
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	return model


def remove_unwanted_x(model_name,remove_fail,org_x,org_y,remove_percent):
    '''
    model_name:'CNN2' or others
    remove_fail:Unwanted fail type (0-7)
    org_x: test x data which are used for prediction (hot coding)
    org_y: test y data which are used for prediction (non hot coding)
    remove_percent: 1=100% remove unwanted fail
    '''
    model = load_model('%s_Model.h5'%model_name)
    predict_list = model.predict(org_x)
    predict_list = np.argmax(predict_list.round(1),axis=1)
    true_list=org_y
    #true_list=np.argmax(y_test.round(1),axis=1)
    true_index=np.where(true_list==remove_fail)
    true_index=true_index[0]
    predict_fail_index=[]
    for i in true_index:
        if predict_list[i]!=remove_fail:
            predict_fail_index.append(i)
    print('total miss predict:%s'%(len(predict_fail_index)))
    percent=int(len(predict_fail_index)*remove_percent)
    predict_fail_index=predict_fail_index[0:percent]
    wrong_count=[0,0,0,0,0,0,0,0]
    for i in range (len(predict_fail_index)):
        a=predict_list[predict_fail_index[i]]
        wrong_count[a]+=1
    print('=====================')
    total=0
    for i,j in enumerate(wrong_count):
        print('guess %s %s: %s'%(i,fault_mapping[i],j))
        total=total+j
    print('=====================')
    print('total removed %s%%:%s'%(remove_percent*100,total))

    new_x=np.zeros((1, waferSizeX, waferSizeY, 3))
    new_y=np.zeros((1,))
    k=1
    temp=int((len(org_x)//100))
    for i,j in enumerate(org_x):
        if i not in predict_fail_index:
            new_x = np.concatenate((new_x,j.reshape(-1, waferSizeX, waferSizeY, 3)), axis=0)
            new_y = np.concatenate((new_y,org_y[i].reshape(-1,)))
        if i > (k*temp):
            k+=1
            print('\rRebuild Data: %s%%'%(k),end=' ')
    new_x=new_x[1:]       
    new_y=new_y[1:]
    print()
    print('new x shape:%s'%(str(new_x.shape)))
    print('new y shape:%s'%(str(new_y.shape)))
    return new_x,new_y


def confusion_matrix_display(model_name,normalize,org_x,org_y):
    '''
    model_name:'CNN2' or others
    normalize:'true' or None
    org_x: test x data which are used for prediction (hot coding)
    org_y: test y data which are used for prediction (non hot coding)
    '''
    model = load_model('%s_Model.h5'%model_name)
    Y_pred = model.predict_generator(org_x)
    y_pred = np.argmax(Y_pred, axis=1)
    y_test=org_y
    cm=confusion_matrix(y_test, y_pred,normalize=normalize)
    cm=np.round(cm,decimals=2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=fault_mapping[:8])
    disp.plot(xticks_rotation=45,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    print('Classification Report')
    print(classification_report(y_test, y_pred, target_names=fault_mapping[:8]))