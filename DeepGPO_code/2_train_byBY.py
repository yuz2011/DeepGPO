import torch.nn as nn
import torch.nn.functional as F
import os
from preprocess import PPeptidePipebyBY
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from model_gly import *
from Bertmodel import ModelbyBYms2_bert
import ipdb
import pandas as pd
from pathlib import Path
from utils import *
from transformers import BertConfig
# ----------------------- training time begin ------------------------------#
from timeit import default_timer as timer
train_time_start = timer()
import datetime
starttime = datetime.datetime.now()
print(f"starttime {starttime}",end="\n\n")
# ----------------------- model parameter for optimization ------------------------------#
import argparse
import logging
logger=logging.Logger("my_logger")
def parsering():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001,  help='learning rate')
    parser.add_argument('--warmupsteps', type=int,   default=0,   help='warmupsteps')
    parser.add_argument('--weight_decay', type=float,  default=1e-2,  help='weight_decay')
    parser.add_argument('--device', type=int, default=0, help='cudadevice')
    parser.add_argument('--testdata', type=str, default="alltest", help='data for test')
    parser.add_argument('--step_size', type=int, default=1000, help='step size')
    parser.add_argument('--lr_sche',  default="False")
    parser.add_argument('--model_ablation', type=str, default="DeepFLR", help='model for ablation (BERT, DeepFLR, protein bert,YYglyco)')
    parser.add_argument('--folder_path', type=str, 
                        default="/remote-home1/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/", 
                        help='the folder path for the training data')
    parser.add_argument("--type",type=str, default="mouse")
    parser.add_argument("--task_name",type=str, default="byBY_mouse_five_tissues")
    parser.add_argument("--trainpathcsv",type=str,
                        default="Five_tissues/Mouse_five_tissues_data_1st_redo1_filtered_combine.csv")
    parser.add_argument("--pattern",type=str,
                        default='*_data_1st.csv')
    parser.add_argument("--ms2_method",type=str,default='cos_sqrt')
    parser.add_argument("--DeepGP_modelpath",type=str,default="/remote-home1/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/code2/task_processing/NO/analysis/code/DeepGP_model/epoch-124_step-47988_mediancos-0.927153.pt")
    args = parser.parse_args()
    return args
args=parsering()
lr=args.lr
lr_sche=args.lr_sche
warmupsteps=args.warmupsteps
weight_decay=args.weight_decay
organism=args.type
DeepGP_modelpath=args.DeepGP_modelpath
testdata=args.testdata
pattern=args.pattern
device = torch.device('cuda', args.device) if torch.cuda.is_available() else torch.device('cpu')
# ----------------------- model parameter------------------------------#
set_seed(seed)  
batch_size=128 
task_name=args.task_name
check_point_name=str(starttime)+task_name+testdata+"_checkpoint"
print("GNN_edge_decoder_type ",GNN_edge_decoder_type)
print("GNN_edge_hidden_dim ",GNN_edge_hidden_dim)
print("GNN_edge_num_layers ",GNN_edge_num_layers)
print("GNN_global_hidden_dim ",GNN_global_hidden_dim)
print(f"Project name check_point_name {check_point_name} !",end="\n\n")
print(f"hyper parameter tested lr {lr} !",end="\n\n")
print(f"hyper parameter tested BATCH_SIZE {BATCH_SIZE} !",end="\n\n")
print(f"hyper parameter tested warmupsteps {warmupsteps} !",end="\n\n")
print(f"hyper parameter tested batch_size {batch_size} !",end="\n\n")
print(f"hyper parameter tested weight decay {weight_decay} !",end="\n\n")
# ----------------------- input pre-processing------------------------------#
folder_path = args.folder_path
import folder_walk
trainpathcsv_list=[]
for org in organism.split(","):
    folder_pathi=folder_path+org+"/"
    print(folder_pathi)
    trainpathcsv_list+=folder_walk.trainpathcsv_list(folder_path=folder_pathi,pattern=pattern)
print(f"Please check trainpathcsv_list {trainpathcsv_list}. The {len(trainpathcsv_list)} files contains all the training data!")

traincsv=pd.DataFrame()
trainpathcsv=folder_path+org+"/"+args.trainpathcsv
for x in  trainpathcsv_list:
    if  testdata in x:
        print(f"The test data is {x}! It is removed from the training data!")
    else:
        train=pd.read_csv(x)
        traincsv=pd.concat([traincsv,train])
        traincsv.reset_index(drop=True,inplace=True)
        print(f"with the addition of {x}, the combined file contains {len(traincsv)} lines")
if "weights" in traincsv.columns:
    traincsv=traincsv[traincsv["weights"]>0.5]
traincsv=traincsv[traincsv["ions"]!="{}"]
traincsv.drop_duplicates(inplace=True)
traincsv.reset_index(drop=True,inplace=True)
print("number of training spectra",len(traincsv))
print("number of training iden_pep",len(traincsv["iden_pep"].drop_duplicates()))
output_directory = os.path.dirname(trainpathcsv)
# Check if the directory exists, and create it if not
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
print(trainpathcsv)
if not os.path.exists(trainpathcsv):
    traincsv.to_csv(trainpathcsv,index=False)
traindatajson=trainpathcsv[:-4]+"_byBYprocessed.json"
trainjson=trainpathcsv[:-4]+"_train_byBYprocessed.json"
devjson=trainpathcsv[:-4]+"_dev_byBYprocessed.json"
# ipdb.set_trace()
traindatajson_path = Path(traindatajson)
if traindatajson_path.exists():
    print(f"{traindatajson} exists.")
    # df_fp=pd.read_json(traindatajson)
else:
    print(f"{traindatajson} does not exist. Begin matrixwithdict to produce result...")
    os.system("python matrixwithdict.py \
    --do_byBY \
    --DDAfile {} \
    --outputfile {} \
    --split {}".format(trainpathcsv,traindatajson,"True"))

filename=traindatajson
filename_train=trainjson
filename_dev=devjson
# import ipdb
databundle_train=PPeptidePipebyBY(vocab=vocab).process_from_file(paths=filename_train)
databundle_dev=PPeptidePipebyBY(vocab=vocab).process_from_file(paths=filename_dev)
traindata=databundle_train.get_dataset("train")
devdata=databundle_dev.get_dataset("train")
print("totaldata",devdata)
#totaldata如果不去重，totaldata就要避免数据泄露问题，按照iden_pep来分割

def savingFastnlpdataset_DataFrame(dataset):
    dataset_field=dataset.field_arrays.keys()
    frame=pd.DataFrame(columns=dataset_field)
    for i in range(len(dataset)):
        c_list=[]
        for name in dataset_field:
            target=dataset.field_arrays[name][i]
            if name=="ions_by_p" or name=="ions_BY_p":
                c_list.append(target.cpu().numpy().tolist())
            else:
                c_list.append(target)
        frame.loc[i]=c_list
    return frame

# ----------------------- model ------------------------------#
model_ablation=args.model_ablation #DeepFLR
print(f"hyper parameter tested model_ablation {model_ablation} !",end="\n\n")
if model_ablation=="Transformer":
    config=BertConfig.from_pretrained("bert-base-uncased")
    deepms2=ModelbyBYms2_bert(config)

if model_ablation=="BERT":
    pretrainmodel="bert-base-uncased"  
    # pretrainmodel="/remote-home1/share/hf_cache/huggingface/hub/models--bert-base-uncased"
    deepms2=ModelbyBYms2_bert.from_pretrained(pretrainmodel)

if model_ablation=="DeepGP":
    config=BertConfig.from_pretrained("bert-base-uncased")
    logger.warning("get config")
    bestmodelpath=DeepGP_modelpath
    model_sign=bestmodelpath.split("/")[-1]
    deepms2=ModelbyBYms2_bert(config)
    bestmodel=torch.load(bestmodelpath).state_dict()
    logger.warning("get bestmodel")
    origin_model=deepms2.state_dict()
    for key in origin_model.keys():
        if key in bestmodel.keys():
            if bestmodel[key].shape !=origin_model[key].shape:
                origin_model[key]=bestmodel[key][:origin_model[key].shape[0],] #linear尺寸匹配不上的部分进行裁剪
                logger.warning(f"size different key: {key}")
            else:
                origin_model[key]=bestmodel[key]
        else:
            logger.warning(f"not found key: {key}")  #GNN匹配不是的参数就保持原样
    logger.warning("starting loading")
    deepms2.load_state_dict(origin_model)

logger.warning("get model")
#model info
from torchinfo import summary
summary(deepms2)
logger.warning("after summary")
# ipdb.set_trace()
# ----------------------- Trainer ------------------------------#
from fastNLP import Const
metrics=CossimilarityMetricfortest_byBY(savename=None,pred=Const.OUTPUT,target=Const.TARGET,
                                   seq_len='seq_len',num_col=num_col,sequence='sequence',
                                   charge="charge",decoration="decoration",
                                   args=args)
# from fastNLP import MSELoss
from MSELoss_for_byBY import MSELoss_byBY
loss=MSELoss_byBY(pred_by="pred_by",pred_BY="pred_BY",target_by="target_by",target_BY="target_BY")
# loss=MSELoss(pred=Const.OUTPUT,target=Const.TARGET)
import torch.optim as optim
optimizer=optim.AdamW(deepms2.parameters(),lr=lr,weight_decay=weight_decay)
if lr_sche=="True":
    step_size=args.step_size
    print(f"hyper parameter tested step_size {step_size} !",end="\n\n")
    lr_scheduler = StepLR(optimizer, step_size=step_size,verbose=True)
    print(f"lr scheduler is used ! lr: {lr}, step_size: {step_size}")

from fastNLP import WarmupCallback,SaveModelCallback,LRScheduler
save_path=filename[:-5]+"/checkpoints"
callback=[WarmupCallback(warmupsteps)]
callback.append(WandbCallback(project="Deepsweet",name=check_point_name,config={"lr":lr,"seed":seed,
"Batch_size":BATCH_SIZE,"warmupsteps":warmupsteps,"temperature":None,"weight_decay":None}))
callback.append(SaveModelCallback(save_path,top=5))
if lr_sche=="True":
    callback.append(LRScheduler(lr_scheduler))
#trainer
from fastNLP import Trainer


if vocab_save:
    vocab.save(os.path.join(save_path,"vocab"))
import ipdb
pptrainer=Trainer(model=deepms2,    train_data=traindata,
                    device=device,  dev_data=devdata,
                save_path=save_path,
                  loss=loss,metrics=metrics,callbacks=callback,
                   optimizer=optimizer,n_epochs=N_epochs,batch_size=batch_size,
                   update_every=int(BATCH_SIZE/batch_size),dev_batch_size=batch_size)
pptrainer.train()


# ----------------------- Time ------------------------------#
train_time_end = timer()
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device} for{check_point_name}: {total_time:.3f} seconds")
    return total_time
total_train_time_model_2 = print_train_time(start=train_time_start,
                                           end=train_time_end,
                                           device=device)