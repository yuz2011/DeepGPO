import pandas as pd
import os
import numpy as np
from pathlib import Path
import masses
import mgf_processing
from weights import *
# --------------------------- argparse ---------------------#
import argparse
def parsering():
    parser = argparse.ArgumentParser()
    # Training parameter
    parser.add_argument('--datafold', type=str, 
                        default="/remote-home1/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/PXD005411/",
                        help='datafold ')
    parser.add_argument('--dfname', type=str, 
                        default="pGlycoDB-GP-FDR-Pro_PXD005411.txt",
                        help='pglyco3 crude result ')
    parser.add_argument('--mgfdatafold', type=str, 
                        default="MSConvert_mgf_PXD005411/" , 
                        help='mgf data fold')
    parser.add_argument('--output_name', type=str, 
                        default="PXD005411_MouseBrain_data_1st.csv", help='outputfile name')
    parser.add_argument('--only_duplicated', type=str,default="Drop_duplicated", help='Duplicated/Drop_duplicated/Retained_all')
    parser.add_argument('--mgfsourceorign', type=str,default="pGlyco3", help='Please ensure the tool for producing mgf (MsConvert or pGlyco3)')
    parser.add_argument('--fragmentation', type=str,default="HCD", help='HCD/EThCD/ETD')
    parser.add_argument('--enzyme', type=str,default="None", help='protease used')
    parser.add_argument('--filter_jsonname', type=str,default="SStruespectrum_filtered_O_", help='')
    parser.add_argument('--not_use_weights', action='store_true', help='')
    args = parser.parse_args()
    return args
args=parsering()
DFNAME=args.datafold+args.dfname
mgfdatafold=args.datafold+args.mgfdatafold
output_name=args.datafold+args.output_name
only_duplicated=args.only_duplicated
mgfsourceorign=args.mgfsourceorign
assert mgfsourceorign in ["pGlyco3","MsConvert"], "mgfsourceorign not in [pGlyco3,MsConvert]"
fragmentation=args.fragmentation
assert fragmentation in ["HCD","ETD","EThCD"], "fragmentation not in [HCD,ETD,EThCD]"
Enzyme=args.enzyme
if args.not_use_weights:
    # Code when not using weights
    print("Not using weights")
else:
    # Code when using weights
    print("Using weights")
# --------------------------- hyper paramaters ---------------------#
FRAG_AVA=["ETD","HCD_1","HCD_by","HCD_BY_2"]
if fragmentation=="HCD":
    FRAG_INDEX=[1,2] #"HCD_1" for BY prediction. "HCD_by" for by prediction
if fragmentation=="ETD":
    FRAG_INDEX=[0]
if fragmentation=="EThCD":
    FRAG_INDEX=[0,1,2]
FRAG_MODE=[x for x in FRAG_AVA if FRAG_AVA.index(x) in FRAG_INDEX]
print(f"FRAG_MODE: {FRAG_MODE}")
jsonfold= os.path.join(mgfdatafold, "json/")
jsonname="SStruespectrum.json"
filter_jsonname=args.filter_jsonname+args.only_duplicated+".json" #相比于SStruespectrum.json提取数据中有的scan，来减少搜索范围
TOLER=20
# --------------------------- pglyco3 result processing---------------------#
def pglyco3_result(DFNAME):
    df=pd.read_csv(DFNAME,sep="\t")
    df.reset_index(inplace=True,drop=True)
    df_column=list(df.columns)
    print(f"Columns of df {df_column}",end = "\n\n")
    print(f"df rank should all be 1.. Please check!!: {list(df['Rank'].drop_duplicates())}",end = "\n\n")
    assert list(df['Rank'].drop_duplicates())==[1]
    df["Peptide"]=df["Peptide"].str.replace("J","N")
    if "LocalizedSiteGroups" in df_column:
        if args.not_use_weights:
            use_weights=False
        else:
            use_weights=True
        print("weight is considered: ", use_weights)
    else:
        use_weights=False
    if use_weights:
        df=df[["RawName","Scan", 'Charge',"Peptide","Mod",
           "PlausibleStruct",'GlySite',"RT","PrecursorMZ",'TotalFDR',"LocalizedSiteGroups"]]
    else:
        df=df[["RawName","Scan", 'Charge',"Peptide","Mod",
           "PlausibleStruct",'GlySite',"RT","PrecursorMZ",'TotalFDR']]
    print(f"Row number of df {len(df)}",end = "\n\n")
    df.drop_duplicates(inplace=True)
    print(f"Row number of df after drop_duplicates {len(df)}",end = "\n\n")
    return df,use_weights

def combine_iden_pep(instance):
    a=instance["Peptide"]
    b=instance["Mod"]
    e=""
    if not pd.isna(b):
        b=b.rstrip(";")
        for i in b.split(";"):
            for k in i.split(","):
                k=k[:3]+"."
                e+=k
        b=e
    else:
        b=None
    c=instance["GlySite"]-1  #GlySite 是从1开始的，会比index J 大一
    d=instance["Charge"]
    e=instance["PlausibleStruct"]
    return str(a)+"_"+str(b)+"_"+str(c)+"_"+str(d)+"_"+str(e)

def pglyco3_processing(df,
                    only_duplicated="Drop_duplicated"):
    """Create required columns.
    Args:
    duplicated: True or False: whether or not only peak duplicated columns.
    True: only duplicated row are retained for repeatability test.
    False: only rows with lowest totalFDR for duplicated columns or unique columns are retained for training.
    
    """
    # ipdb.set_trace()
    df["iden_pep"]=df.apply(combine_iden_pep,axis=1) #eg. JASQNQDNVYQGGGVCLDCQHHTTGINCER_16.Car.19.Car.28.Car._0_4_(N(N(H(H(H))(H(H)))))
    if only_duplicated=="Duplicated":
        df1=df[["iden_pep"]].loc[df["iden_pep"].duplicated()].drop_duplicates()
        df=df.loc[df["iden_pep"].isin(df1["iden_pep"])]
    print("Waiting to process multiply glycopeptides")
    if only_duplicated == "Drop_duplicated":
        df.sort_values(by='TotalFDR',ascending=True,inplace=True)
        # ipdb.set_trace()
        df.drop_duplicates(subset=['iden_pep'],inplace=True)
        df.reset_index(drop=True,inplace=True)
    if only_duplicated == "Retained_all":
        pass
    return df
# --------------------------- spectrum filtration---------------------#
#从json中找到相应的谱图，缩小搜索空间
def json_extraction(jsonfold=jsonfold,
                    jsonname=jsonname,
                    filename=filter_jsonname,
                    mgfsourceorign=mgfsourceorign):
    datalis=pd.read_json(os.path.join(jsonfold, jsonname))
    datalis["title"]=datalis["SourceFile"].map(str) + "-" + datalis["Spectrum"].map(str)
    datalis=datalis.loc[datalis["title"].isin(df["GlySpec"])]
    print("Please ensure the Spectrum numbers of MsConvert json files match those of the pGlyco3 result!")
    datalis.reset_index(inplace=True, drop=True)
    datalis.to_json(os.path.join(jsonfold, filename))
    return datalis
# ----------------------- ions picking ------------------------------#
def fragment_training(instance):
    spectrum=instance["GlySpec"]
    datalis_1=datalis.loc[datalis["title"]==spectrum]
    datalis_1=datalis_1.reset_index(drop=True)
    iden_pep=instance["iden_pep"]
    mz_calc=masses.pepfragmass(iden_pep,FRAG_MODE,3) #iden_pep已经改成了glysite，避免多J的可能
    ppm=TOLER
    FragmentMz=[]
    for mz in mz_calc:
        for ion in mz:
            FragmentMz.append(list(ion.values())[0])
    FragmentMz=list(set(FragmentMz))
    mass={"FragmentMz":FragmentMz}
    #FragmentMz：所有算出来的理论质荷比
    mzdict=mgf_processing.putTintensity(ppm, mass, datalis_1)
    for k in list(mzdict.keys()):
        if mzdict[k]==0:
            del mzdict[k]
    mzdict_1={}
    #补上mzdict的碎裂类型
    for i in mz_calc:
        for a in i:
            mz_calc_1=list(a.values())[0]
            if mz_calc_1 in list(mzdict.keys()):
                # print("a",a)
                # print("mzdict[mz_calc_1]",mzdict[mz_calc_1])
                type=list(a.keys())[0]
                intensity=mzdict[mz_calc_1]
                if not mz_calc_1 in mzdict_1.keys():
                    type_list=[]
                    type_list.append(type)
                    ions=(type_list,intensity)
                    mzdict_1[mz_calc_1]=ions
                else:
                    type_list=mzdict_1[mz_calc_1][0]
                    type_list.append(type)
                    ions=(type_list,intensity)
                    mzdict_1[mz_calc_1]=ions
    return mzdict_1

def mz_matching(instance):
    spectrum=instance["GlySpec"]
    datalis_1=datalis.loc[datalis["title"]==spectrum]
    datalis_1=datalis_1.reset_index(drop=True)
    iden_pep=instance["iden_pep"]
    mz_calc=masses.pepfragmass(iden_pep,["HCD_BY_2"],4) #iden_pep已经改成了glysite，避免多J的可能
    ppm=TOLER
    FragmentMz=[]
    for mz in mz_calc:
        for ion in mz:
            FragmentMz.append(list(ion.values())[0])
    FragmentMz=list(set(FragmentMz))
    FragmentMz.sort()
    mzexp=datalis_1["mz"][0]
    mzexp.sort()
    matchmz=[]
    for k in mzexp:
        i = (np.abs(np.array(FragmentMz) - k)).argmin()
        # ipdb.set_trace()
        if abs(FragmentMz[i] - k) < k * TOLER * 1 / 1000000:  #args.ppm=tolerance here,可以改回args版本
            matchmz.append(k)
    return {"matchmz":len(matchmz),"calc":len(FragmentMz),"mzexp":len(mzexp)}
# --------------------------- execution ---------------------#
if __name__=="__main__":
    DFNAME_path = Path(DFNAME)
    print(DFNAME_path)
    assert DFNAME_path.exists()
    # pglyco3 formatted result
    df_fp,use_weights=pglyco3_result(DFNAME)
    df=pglyco3_processing(df_fp,
                        only_duplicated=only_duplicated)
    # if mgfsourceorign=="MsConvert":
    df["GlySpec"]=df["RawName"].map(str) + "-" + df["Scan"].map(str)
    #json file
    json_path=Path(jsonfold,jsonname)
    if json_path.exists():
        print(f"{jsonname} exists.")
    else:
        print(f"{jsonname} does not exist. Begin mgf_process to produce required file...")
        datalis=mgf_processing.mgf_process(mgfdatafold=mgfdatafold,sourceorign=mgfsourceorign)
    #filtered json file
    file3_name_path = Path(jsonfold,filter_jsonname)
    # if file3_name_path.exists():
    #     print(f"{filter_jsonname} exists.")
    #     datalis=pd.read_json(os.path.join(jsonfold, filter_jsonname))
    # else:
    print(f"{file3_name_path} does not exist. Begin json_extraction to produce required file...")
    datalis=json_extraction(jsonfold=jsonfold,
                jsonname=jsonname,
                filename=filter_jsonname,
                mgfsourceorign=mgfsourceorign)
    datalis.drop_duplicates(subset="title",inplace=True)
    df=df[df["GlySpec"].isin(datalis["title"])]
    assert len(df["GlySpec"].drop_duplicates())==len(datalis["title"].drop_duplicates())
    if use_weights:
        df["weights"] = df.apply(lambda x: weights(x, Enzyme), axis=1)
        df=df[[ "GlySpec",'Charge',"RT","PrecursorMZ",'Peptide', 'Mod', 'PlausibleStruct', 'GlySite', 'iden_pep',"TotalFDR","weights"]]
    else:
        df=df[[ "GlySpec",'Charge',"RT","PrecursorMZ",'Peptide', 'Mod', 'PlausibleStruct', 'GlySite', 'iden_pep',"TotalFDR"]]
    df.drop_duplicates(subset=["GlySpec",'Charge',"RT","PrecursorMZ",'Peptide', 'Mod', 'PlausibleStruct', 'GlySite', 'iden_pep',"TotalFDR"],inplace=True)
    df.reset_index(drop=True,inplace=True)
    print("len(df_iden_pep.drop_duplicates())",len(df["iden_pep"].drop_duplicates()))
    print("len(df)",len(df))
    df["ions"]=df.apply(fragment_training,axis=1)
    
    df.to_csv(output_name,index=False)