import argparse
import json
import os
import numpy as np
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torch.nn.functional as func

from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import random
import pandas as pd
from copy import deepcopy
import argparse
import json
import logging
import os
import sys
#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from timeit import default_timer as timer
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, sampler
import numpy as np
import pandas as pd
import shlex, subprocess
from torchsummary import summary
from PIL import Image
import requests
import pydicom
import glob
from torchvision.models import densenet121

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

nnClassCount = 14 

def find_file_path(filename):
    file_path = None
    for path in glob.glob('**/' + filename, recursive=True):
        if file_path is None:
            file_path = os.path.abspath(path)
            break
    return file_path


''' INFERENCING'''
def get_json(coord,og_img_size):
    # final_coord = [[(coord[x]+16)*(og_img_size[0]/256),(coord[y]+16)*(og_img_size[1]/256)] for x,y in zip(list(range(0,len(coord),2)),list(range(1,len(coord),2)))]
    final_coord = [[coord[x],coord[y]] for x,y in zip(list(range(0,len(coord),2)),list(range(1,len(coord),2)))]
    #final_coord = np.array(final_coord)
    return(final_coord)

def get_coord_dict(cont_new):
    final_pairs = get_pairs(cont_new)
    line_list = []
    for i in range(0,len(final_pairs)):
        coord_cur,coord_after = final_pairs[i][0],final_pairs[i][1]
        z = {'active':True,
        'highlight': True,
        'lines': [{'x': coord_after[0], 'y': coord_after[1]}],
        'x': coord_cur[0],
        'y': coord_cur[1]}
        line_list.append(z)
    return line_list

def get_1D_coord(contours):
    global_list=[]
    for contour_id in range(len(contours)):
        local_list=[]
        for point_idx in range(contours[contour_id].shape[0]):
            if(point_idx==0):
                X_0= contours[contour_id][point_idx][0][0].astype('float')
                Y_0 = contours[contour_id][point_idx][0][1].astype('float')
            X = contours[contour_id][point_idx][0][0].astype('float')
            Y = contours[contour_id][point_idx][0][1].astype('float')
            local_list.append(X)
            local_list.append(Y)
            # If the last point is reached, then append the first point
            if(point_idx == contours[contour_id].shape[0]-1):
                local_list.append(X_0)
                local_list.append(Y_0)
        global_list.append(deepcopy(local_list))
    return(global_list)

def threshold(minimum,maximum,image,binary=True):
    if(binary): # If binary is True, then the image is converted to binary
        image[image<minimum]=0
        image[image>maximum]=0
        image[(image>0)]=1
    else: # If binary is False, then the image is converted to grayscale
        image[image<minimum]=0
        image[image>maximum]=0
    return image

def get_pairs(cont_new):
    pairs=[]
    for i in range(0,cont_new.shape[0]):
        if (i < (cont_new.shape[0]-1)):
            pairs.append((cont_new[i],cont_new[i+1]))
        else:
            pairs.append((cont_new[i],cont_new[0]))
    return(pairs)

def model_fn(model_dir):
    from torchvision.models import densenet121
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    model =  ChexNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = "modele.pth"
    print(">>>>>>>>>>>>>>> model dir >>>>>>>>")
    print(model_dir)
    model_def_path = os.path.join(model_dir, "model.pth")
    if not os.path.isfile(model_def_path):
        print(f"file not found in {model_def_path}")
        raise RuntimeError("Missing the model definition file")
    # model = torch.nn.DataParallel(model)
    modelCheckpoint = torch.load(model_def_path)
    model.load_state_dict(modelCheckpoint['state_dict'])
    model.eval()
    return model.to(device)

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model, path)

def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    import requests
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if content_type == 'application/json':    
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_data = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        image_transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data = image_transform(image_data)
        logger.info(str(data.shape))
        data = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
        logger.info(str(data.shape))
        return [data, image_data.size]
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')
    
def predict_fn(input_object, model):
    input_object, og_img_size = input_object
    with torch.no_grad():
        logger.info(str(input_object.shape))
        # logger.info(model.classifier)
        output = model.backbone(input_object)
        l = model.forward(input_object)
        l = torch.sigmoid(l)
        heatmap = None
        weights = list(model.backbone.parameters())[-2]
        for i in range (0, len(weights)):
            map = output[0,i,:,:]
            if i == 0: 
                heatmap = weights[i] * map
            else: 
                heatmap += weights[i] * map
            npHeatmap = heatmap.cpu().data.numpy()
    return [output,l,npHeatmap, og_img_size]

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    output,l,npHeatmap,og_img_size = predictions
    heatmap = npHeatmap
    heatmap = ((heatmap - heatmap.min()) * (1 / (heatmap.max() - heatmap.min())) * 255).astype(np.uint8)
    # cam = npHeatmap / np.max(npHeatmap)
    cam = cv2.resize(heatmap, og_img_size)
    heatmap = cam
    # heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    #---- Blend original and heatmap 
    temp = heatmap.copy()
    img = temp#[:,:,0]
    # img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    print(img.min())
    img = (img/1).astype('uint8')
    binary = threshold(220,255,img)
    binary = binary.astype(np.int32)
    contours, _ = cv2.findContours(binary,cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE) 
    coords = get_1D_coord(contours)
    data_final = []
    for coord in coords:
        data_final.append(get_json(coord,og_img_size))
        
    print("FINAL JSON")
    output = response_converter(l.tolist(),data_final)
    return json.dumps(output)

def response_converter(labels,coords):
    print(labels,coords)
    '''
    THIS IS A CUSTOM RESPONSE CCONVERTER
    CARPL EXPECTS OUTPUT IN THIS FORMAT:
    {
        "response": {
            "findings":[
                {
                    "name":"class_A",
                    "probability":score_A
                },
                {
                    "name":"class_B",
                    "probability":score_B
                }
            ],
            "rois":[
                    {
                        "finding_name":"class1",
                        "type":"Rectangle",
                        "points":[
                            [50,300],
                            [100,200]
                        ]
                    },
                    {
                        "finding_name":"class2",
                        "type":"Freehand",
                        "points":[
                            [500,300],
                            [500,320],
                            [500,420],
                            [800,420],
                            [800,370],
                            [800,320]
                        ]
                    }
                ]
        }
    }
    '''
    import numpy as np
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    try:

        return {
            
            "response" : {
                "findings":[{
                    "name":class_names[e],
                    "probability":l
                } for e,l in enumerate(labels[0])],
                "rois":[
                    {
                    "finding_name":str(e),
                    "type":"Freehand",
                    "points":coords[e],
                    } for e,l in enumerate(coords)
                ]
            }
        }
    except Exception as e:
        print(e)



class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file, transform=None, policy="ones"):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        import glob
        image_names = glob.glob(image_list_file+"/*.png",recursive = True )
                    
        labels = np.ones((len(image_names),nnClassCount))
        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

class CheXpertTrainer():

    def train (model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, launchTimestamp, checkpoint, save_path):
        
        #SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
                
        #SETTINGS: LOSS
        loss = torch.nn.CrossEntropyLoss()
        
        #LOAD CHECKPOINT 
        use_gpu = torch.cuda.is_available()
        if checkpoint != None:
            state_dict = torch.load(checkpoint)
            model.load_state_dict(state_dict)
            print(">>>>>>>>>>>>>>>>>>> MODEL LOADED SUCCESSFULLY >>>>>>>>>>>>>>")

        
        #TRAIN THE NETWORK
        lossMIN = 100000
        epochID = 0
        lossMIN = 0
        torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, save_path +  "/model.pth")
        for epochID in range(0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            
            batchs, losst, losse = CheXpertTrainer.epochTrain(model, dataLoaderTrain, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)
            lossVal = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)


            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, save_path +  "/model_base.pth")
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
                # save_model(model, save_path)
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
        
        return batchs, losst, losse        
    #-------------------------------------------------------------------------------- 
       
    def epochTrain(model, dataLoaderTrain, dataLoaderVal, optimizer, trMaxEpoch, classCount, loss):
        
        batch = []
        losstrain = []
        losseval = []
        
        model.train()

        for batchID, (varInput, target) in enumerate(dataLoaderTrain):
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                varTarget = target.cuda(non_blocking = True)
            else:
                varTarget = target
            #varTarget = target.cuda()         


            varOutput = model(varInput)
            lossvalue = loss(varOutput, varTarget)
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            l = lossvalue.item()
            losstrain.append(l)
            
            if batchID%35==0:
                print(batchID//35, "% batches computed")
                #Fill three arrays to see the evolution of the loss


                batch.append(batchID)
                
                le = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, classCount, loss).item()
                losseval.append(le)
                
                print(batchID)
                print(l)
                print(le)
                
        return batch, losstrain, losseval
    
    #-------------------------------------------------------------------------------- 
    
    def epochVal(model, dataLoaderVal, optimizer, epochMax, classCount, loss):
        
        model.eval()
        
        lossVal = 0
        lossValNorm = 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderVal):
                use_gpu = torch.cuda.is_available()
                if use_gpu:
                    target = target.cuda(non_blocking = True)
                else:
                    target = target
                    
                varOutput = model(varInput)
                
                losstensor = loss(varOutput, target)
                lossVal += losstensor
                lossValNorm += 1
                
        outLoss = lossVal / lossValNorm
        return outLoss
    
    
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC
        
        
    #-------------------------------------------------------------------------------- 
    
    
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names):   
        
        cudnn.benchmark = True
        use_gpu = torch.cuda.is_available()
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        model.eval()
        
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                use_gpu = torch.cuda.is_available()
                if use_gpu:
                    target = target.cuda()
                    outGT = torch.cat((outGT, target), 0).cuda()
                else:
                    target = target
                    outGT = torch.cat((outGT, target), 0)
                    

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
            
                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
from torchvision.models import densenet121

class ChexNet(nn.Module):
    from torchvision.models import densenet121
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __init__(self):
        super().__init__()
        self.backbone = densenet121(False).features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 Flatten(),
                                 nn.Linear(1024, 14))

    def forward(self, x):
        return self.head(self.backbone(x))

    def predict(self, image):
        """
        input: PIL image (w, h, c)
        output: prob np.array
        """
        image = V(self.tfm(image)[None])
        image=image.cpu()
        py = torch.sigmoid(self(image))
        prob = py.detach().cpu().numpy()[0]
        return prob
    
def create_dataloader(traindir,testdir,validdir,batch_size=128):

    image_transforms = {
    # Train uses data augmentation
    'train':
        
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224), 
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ])
        ,
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    }
    
    
    data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }

    
    return data,dataloaders

        
def preops(traindir,testdir,validdir):
    print("********* inside preops *************")
    print(f"{traindir} , {testdir} , {validdir}")
    # Empty lists
    categories = []
    img_categories = []
    n_train = []
    n_valid = []
    n_test = []
    hs = []
    ws = []

    # Iterate through each category
    for d in os.listdir(traindir):
        if not d.startswith('.'):
            categories.append(d)

            # Number of each image
            train_imgs = os.listdir(traindir + d)
            valid_imgs = os.listdir(validdir + d)
            test_imgs = os.listdir(testdir + d)
            n_train.append(len(train_imgs))
            n_valid.append(len(valid_imgs))
            n_test.append(len(test_imgs))

            # Find stats for train images
            for i in train_imgs:
                if not i.startswith('.'):
                    img_categories.append(d)
                    img = Image.open(traindir + d + '/' + i).convert("RGB")
                    img_array = np.array(img)
                    # Shape
                    hs.append(img_array.shape[0])
                    ws.append(img_array.shape[1])

    # Dataframe of categories
    cat_df = pd.DataFrame({'category': categories,
                           'n_train': n_train,
                           'n_valid': n_valid, 'n_test': n_test}).\
        sort_values('category')

    # Dataframe of training images
    image_df = pd.DataFrame({
        'category': img_categories,
        'height': hs,
        'width': ws
    })
    
    return cat_df

class HeatmapGenerator():
    def __init__ (self, pathModel=None, nnClassCount=14, transCrop=224):
        # model = DenseNet121(nnClassCount)
        use_gpu = None
        
        if use_gpu:
            model = torch.nn.DataParallel(model)
        else:
            model = torch.nn.DataParallel(model)
        
        model = torch.load(pathModel)
        # model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.module.densenet121.features.parameters())[-2]

        #---- Initialize the image transform
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize((transCrop, transCrop)))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)  
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        with torch.no_grad():
            use_gpu = None
 
            imageData = Image.open(pathImageFile).convert('RGB')
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            if use_gpu:
                imageData = imageData
            l = self.model(imageData)
            output = self.model.module.densenet121.features(imageData)
            label = class_names[torch.max(l,1)[1]]
            #---- Generate heatmap
            heatmap = None
            for i in range (0, len(self.weights)):
                map = output[0,i,:,:]
                if i == 0: heatmap = self.weights[i] * map
                else: heatmap += self.weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy()
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        #---- Blend original and heatmap 
        temp = heatmap.copy()
        img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        print(img.min())
        # print(img.shape)
        img = (img/1).astype('uint8')
        binary = threshold(200,255,img)
        binary = binary.astype(np.int32)
        contours, _ = cv2.findContours(binary,cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE) 
        multi_data = []
        multidata_dict={}
        with open("ROIFormat.txt") as json_file:
            data_orig = json.load(json_file)
        coords = get_1D_coord(contours)
        for coord in coords:
            data_final = get_json(data = data_orig, coord=coord)
            multi_data.append(deepcopy(data_final['allTools'][0]))
        multidata_dict['allTools'] = multi_data
        
        print("FINAL JSON")
        print(multidata_dict)
        
#         imgOriginal = cv2.imread(pathImageFile, 1)
#         imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        
        
#         img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)            
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.title(label)
#         plt.imshow(img)
#         plt.plot()
#         plt.axis('off')
#         plt.savefig(pathOutputFile)
#         plt.show()

def find_file_path(filename):
    file_path = None
    for path in glob.glob('**/' + filename, recursive=True):
        if file_path is None:
            file_path = os.path.abspath(path)
            break
    return file_path

def run(args):
    batch_size = 128
    traindir = args.data_dir_train + "/" #datadir + '/train/'
    validdir = args.data_dir_val + "/" #datadir + '/val/'
    testdir = args.data_dir_test + "/" #datadir + '/test/'
    
    use_gpu = torch.cuda.is_available()
    pathFileTrain = args.data_dir_train
    pathFileValid = args.data_dir_val

    # Neural network parameters:
    nnIsTrained = False                 #pre-trained using ImageNet
    nnClassCount = 14                   #dimension of the output

    # Training settings: batch size, maximum number of epochs
    trBatchSize = 64
    trMaxEpoch = 1

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = (320, 320)
    imgtransCrop = 224

    # Class names
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    
    #TRANSFORM DATA
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformList = []
    #transformList.append(transforms.Resize(imgtransCrop))
    transformList.append(transforms.RandomResizedCrop(imgtransCrop))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)      
    transformSequence=transforms.Compose(transformList)

    #LOAD DATASET
    
    # cat_df = preops(traindir,testdir,validdir)
    nnClassCount = len(class_names)
    
    
    # data,dataloaders = create_dataloader(traindir,testdir,validdir,batch_size)
    
    dataset = CheXpertDataSet(pathFileTrain ,transformSequence, policy="ones")
    # datasetTest, datasetTrain = random_split(dataset, [len(dataset), len(dataset)])
    datasetTest = dataset
    datasetTrain = dataset
    datasetValid = CheXpertDataSet(pathFileValid, transformSequence)            
    
    
    

    # dataLoaderTrain = dataloaders["train"]#DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
    # dataLoaderVal = dataloaders["val"]#DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
    # dataLoaderTest = dataloaders["test"]#DataLoader(dataset=datasetTest, num_workers=24, pin_memory=True)
    
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
    dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
    dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=24, pin_memory=True)
    
    # initialize and load the model
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = ChexNet().cuda()
        # model = torch.nn.DataParallel(model).cuda()
    else:
        model = ChexNet()
        # model = torch.nn.DataParallel(model)
        

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    save_path = args.model_dir
    path = find_file_path('chexnet.h5')
    
    batch, losst, losse = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, timestampLaunch, path, save_path)
    print("Model trained")

    losstn = []
    for i in range(0, len(losst), 35):
        losstn.append(np.mean(losst[i:i+35]))

    print(losstn)
    print(losse)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
   
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    parser.add_argument("--data-dir-train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    
    parser.add_argument("--data-dir-test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    
    parser.add_argument("--data-dir-val", type=str, default=os.environ["SM_CHANNEL_VALIDATING"])
    
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args = parser.parse_args()
    
    run(args)
    

