# STD MODULE
import os
import sys
import numpy as np
import cv2
import pandas as pd

# TORCH MODULE
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

# PYTVISION MODULE
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view
from pytvision.datasets.datasets  import Dataset
from pytvision.datasets.factory  import FactoryDataset

# LOCAL MODULE
#from torchlib.datasets.factory  import FactoryDataset 
#from torchlib.datasets import Dataset, SecuencialSamplesDataset, TripletsDataset
from torchlib.datasets import TripletsDataset
from torchlib.neuralnet import NeuralNetTriplet
from misc import get_transforms_det

from argparse import ArgumentParser

# METRICS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics



def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('--project',     metavar='DIR',  help='path to projects')
    parser.add_argument('--projectname', metavar='DIR',  help='name projects')
    parser.add_argument('--pathdataset', metavar='DIR',  help='path to dataset')
    parser.add_argument('--pathnameout', metavar='DIR',  help='path to out dataset')
    parser.add_argument('--filename',    metavar='S',    help='name of the file output')
    parser.add_argument('--model',       metavar='S',    help='filename model')  
    return parser


def main():
    
    parser = arg_parser();
    args = parser.parse_args();

    # Configuration
    project         = args.project
    projectname     = args.projectname
    pathnamedataset = args.pathdataset  
    pathnamemodel   = args.model
    pathproject     = os.path.join( project, projectname )
    pathnameout     = args.pathnameout
    filename        = args.filename
    
    no_cuda=False
    parallel=False
    gpu=0
    seed=1
    brepresentation=True
    brecover_test=True
    
    #imagesize=128
    #idenselect=np.arange(10)
    
    
    # experiments
    experiments = [ 
        { 'name': 'ferp',      'subset': FactoryDataset.training,    'real': True }, # reference
        { 'name': 'ferp',      'subset': FactoryDataset.test,        'real': True },
        { 'name': 'affectnet', 'subset': FactoryDataset.validation,  'real': True },
        { 'name': 'ck',        'subset': FactoryDataset.training,    'real': True },
        { 'name': 'jaffe',     'subset': FactoryDataset.training,    'real': True },
        { 'name': 'bu3dfe',    'subset': FactoryDataset.training,    'real': True },
        ]
 

    # representation datasets
    if brepresentation: 
    
        # Load models
        print('>> Load model ...')
        network = NeuralNetTriplet(
            patchproject=project,
            nameproject=projectname,
            no_cuda=no_cuda,
            parallel=parallel,
            seed=seed,
            gpu=gpu,
            )

        cudnn.benchmark = True

        # load model
        if network.load( pathnamemodel ) is not True:
            print('>>Error!!! load model')
            assert(False)  
    

        size_input = network.size_input
        for  i, experiment in enumerate(experiments):
            
            name_dataset = experiment['name']
            subset = experiment['subset']
            breal = experiment['real']
            dataset = []
             
                
            # real dataset 
            dataset = Dataset(    
                data=FactoryDataset.factory(
                    pathname=pathnamedataset, 
                    name=name_dataset, 
                    subset=subset, 
                    #idenselect=idenselect,
                    download=True 
                ),
                num_channels=3,
                transform=get_transforms_det( network.size_input ),
                )
            

            dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=10 )
            
            print(breal)
            print(subset)
            print(dataloader.dataset.data.classes)
            print(len(dataset))
            print(len(dataloader))
            
            # representation 
            Zs, Ys = network.representation( dataloader )            
            print(Ys.shape, Zs.shape )
            
            reppathname = os.path.join( pathproject, 'rep_{}_{}_{}.pth'.format( projectname, name_dataset, subset ) )
            torch.save( { 'Z':Zs, 'Y':Ys }, reppathname )
            print( 'save {} {} representation ...'.format( name_dataset, subset) )
            
    
            
    # evaluate
    method_name = 'knn'
    param={ 'n_neighbors':11 }
    rep_trn_pathname = os.path.join( pathproject, 'rep_{}_{}_{}.pth'.format( projectname, experiments[0]['name'], experiments[0]['subset'] ) )
    experiments = experiments[1:]
    tuplas=[]
    
    print('|Num\t|Acc\t|Prec\t|Rec\t|F1\t|Set\t')
    for i,experiment in enumerate(experiments):

        name_dataset = experiment['name']
        subset = experiment['subset']
        
        #print('Process {}'.format(i))
        #print('Dataset: {}, subset {}'.format(name_dataset, subset) )
        
        rep_val_pathname = os.path.join( pathproject, 'rep_{}_{}_{}.pth'.format( projectname, name_dataset, subset ) )
        data_emb_train = torch.load(rep_trn_pathname)
        data_emb_val = torch.load(rep_val_pathname)

        Xo  = data_emb_train['Z']
        Yo  = data_emb_train['Y']
        Xto = data_emb_val['Z']
        Yto = data_emb_val['Y']

        if method_name == 'knn':
            clf = KNeighborsClassifier( **param )
        elif method_name == 'gm': 
            clf = GaussianNB( **param )
        else:
            print('not method suport')
            assert(False)
            
        # 
        # clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
        # clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=100, alpha=1e-4,
        #                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
        #                     learning_rate_init=.01)

        clf.fit(Xo,Yo)

        y = Yto
        yhat = clf.predict(Xto)

        acc = metrics.accuracy_score(y, yhat)
        nmi_s = metrics.cluster.normalized_mutual_info_score(y, yhat)
        mi = metrics.cluster.mutual_info_score(y, yhat)
        h1 = metrics.cluster.entropy(y)
        h2 = metrics.cluster.entropy(yhat)
        nmi = 2*mi/(h1+h2)

        #print(mi, h1, h2)

        precision = metrics.precision_score(y, yhat, average='macro')
        recall = metrics.recall_score(y, yhat, average='macro')
        f1_score = 2*precision*recall/(precision+recall)

        
        print( '|{}\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|{}\t'.format(
                i, 
                acc, precision, recall, f1_score,
                subset,
            ).replace('.',',')  )
        
        
        #|Name|Dataset|Cls|Acc| ...
        tupla = { 
            'Dataset': '{}({})'.format(  name_dataset,  subset ),
            'Cls':method_name,
            'Accuracy': acc*100.0,
            'NMI': nmi_s*100.0,
            'Precision': precision*100.0,
            'Recall': recall*100.0,
            'F1 score': f1_score*100.0,        
        }
        tuplas.append(tupla)


    # save
    df = pd.DataFrame(tuplas)
    df.to_csv( os.path.join( pathproject, 'experiments_recovery.csv' ) , index=False, encoding='utf-8' )

    print('experiment')
    print('DONE!!!')
        

if __name__ == '__main__':
    main()