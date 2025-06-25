import pandas as pd
import os
import numpy as np
from pathlib import Path
import ipdb
import masses
import mgf_processing
from weights import *
#暂时去掉有两个J的，测试数据集也去掉，也可以加上
#1st step: format pglco3 result into training dataset (.csv) for subsequent processing (such as matrixwithdict)
# --------------------------- argparse ---------------------#
import argparse
def parsering():
    parser = argparse.ArgumentParser()
    # Training parameter
    parser.add_argument('--datafold', type=str, 
                        default="/remote-home1/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/mouse/PXD005411/",
                        help='datafold ')
    parser.add_argument('--DFNAME', type=str, 
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
DFNAME=args.datafold+args.DFNAME
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
# import ipdb
# ipdb.set_trace()
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
# --------------------------- glycans ---------------------#
from collections import Counter
import re
def count_ahnf(row):
    # 统计每个字符出现的次数
    counter = Counter(row)
    # 生成新字符串
    new_row = {}
    for char in row:
        if char in 'NAHGF' and counter[char]>0:
            new_row[char]=counter[char]
            # 从计数器中删除已经处理过的字符
            del counter[char]
    return new_row

def dict2str(row):
    newrow=""
    for k in "NHAGF":
        if k in row.keys():
            newrow+=k
            newrow+=str(row[k])
    return newrow
def convert_glycan_string(glycan_str):
    glycan_list = glycan_str.split(';')
    converted = []
    for glycan in glycan_list:
        n = re.search(r'HexNAc\((\d+)\)', glycan)
        h = re.search(r'Hex\((\d+)\)', glycan)
        a = re.search(r'NeuAc\((\d+)\)', glycan)
        g = re.search(r'NeuGc\((\d+)\)', glycan)
        f = re.search(r'Fuc\((\d+)\)', glycan)
        n_val = f'N{n.group(1)}' if n else ''
        h_val = f'H{h.group(1)}' if h else ''
        a_val = f'A{a.group(1)}' if a else ''
        g_val = f'A{g.group(1)}' if g else ''
        f_val = f'A{f.group(1)}' if f else ''
        converted.append(f"{n_val}{h_val}{a_val}{g_val}{f_val}")
    return ';'.join(converted)
def struct(row):
    glycan=row.split(";")
    result=[]
    for i in glycan:
        strcount=convert_glycan_string(i)
        if strcount in glycan_dict.keys():
            result.append(glycan_dict[strcount])
        else:
            print(i)
            import ipdb
            ipdb.set_trace()
    return ';'.join(result)
def convert_modification(mod_str):
    match = re.match(r'([A-Z])(\d+)\(([^/]+) / [^)]+\)', mod_str)
    if match:
        aa, position, mod = match.groups()
        return f"{position},{mod}[{aa}]"
    return mod_str
def Modi(row):
    mods=row.split(";")
    result=[]
    for mod in mods:
        if not "Glycan" in mod:
            result.append(convert_modification(mod))
    return ';'.join(result)
def extract_positions(mod_string):
    # 匹配包含 OGlycan 或 NGlycan 的修饰，提取位置数字
    matches = re.findall(r'[A-Z](\d+)\((?:O|N)Glycan\s*/', mod_string)
    return ';'.join(matches)
glycans=pd.read_csv("/remote-home1/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/code2/task_processing/NO/multiple/glycans/pGlyco-O-Glycan.gdb",sep="\t")
glycans.columns=["text"]
glycans['count'] = glycans['text'].apply(count_ahnf)
glycans["strcount"]=glycans['count'].apply(dict2str)
glycans=glycans[["text","strcount"]]
glycans.rename(columns={'text': 'glycan'}, inplace=True)
glycans.drop_duplicates(subset="strcount",keep="first",inplace=True)
glycan_dict = dict(zip(glycans['strcount'], glycans['glycan']))
# --------------------------- pglyco3 result processing---------------------#
def pglyco3_result(DFNAME):
    df=pd.read_csv(DFNAME,sep="\t")
    df.reset_index(inplace=True,drop=True)
    df_column=list(df.columns)
    print(f"Columns of df {df_column}",end = "\n\n")
    if not "RawName" in df_column:
        raw_name = "2019_09_16_" + DFNAME.removesuffix('_GlycoPSMs.txt').split("/")[-1]
        df['RawName'] = raw_name
       
    # a Byonic score greater than or equal to 200, a logProb value greater than or equal to 2,and a peptide length greater than
 #4 residues. A maximum of two glycosites were allowed for any one glycopeptide. 
    df=df[df["Score"]>=200]
    df=df[df["Localized"]==True] #加上这个可能就是谱图会下降很多
    df=df[df["logProb"]>=2]
    df["peptidelength"]=df["Sequence"].apply(lambda x: len(x))
    df=df[df["peptidelength"]>4]
    df=df[df["GlycoSite"].str.count(";")<2] #之后可以改成为<2,先对双糖基化处理
    df["NewScan"] = np.where(
    df["Fragmentation"] == "ETD",
    df["MasterScan"],
    df["ScanNumber"])
    #矫正一下mods和glycans
    df["PlausibleStruct"]=df["Glycans"].apply(struct)
    df["Mod"]=df["Mods"].apply(Modi)
    # STNASTVPFRNPDENSR_3.Dea._0;2;5_4_(N(A)(H));(N(A)(H(A)))
    # df=df[df["Mod"].str.contains("Dea")]
    # import ipdb
    # ipdb.set_trace()
    df["GlySite"]=df["Mods"].apply(extract_positions)
    df=df[["RawName","NewScan","Charge",'Sequence',  "Mod",
           "PlausibleStruct",'GlySite']]
    df.columns=["RawName","Scan", 'Charge',"Peptide","Mod",
        "PlausibleStruct",'GlySite']
    use_weights=False
    print(f"Row number of df {len(df)}",end = "\n\n")
    df.drop_duplicates(inplace=True)
    print(f"Row number of df after drop_duplicates {len(df)}",end = "\n\n")
    return df,use_weights
def subtract_one(input_str):
    # 将字符串按中文分号分割
    numbers = input_str.split(';')
    # 每个数字减1，并转换为字符串
    result = [str(int(num) - 1) for num in numbers]
    # 用分号连接回去
    return ';'.join(result)
def combine_iden_pep(instance):
    a=instance["Peptide"]
    b=instance["Mod"]
    e=""

    if b!="":
        b=b.rstrip(";")
        for i in b.split(";"):
            for k in i.split(","):
                k=k[:3]+"."
                e+=k
        b=e
    else:
        b=None
    c=subtract_one(instance["GlySite"])  #GlySite 是从1开始的，会比index J 大一
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
    df["iden_pep"]=df.apply(combine_iden_pep,axis=1) #eg. JASQNQDNVYQGGGVCLDCQHHTTGINCER_16.Car.19.Car.28.Car._0_4_(N(N(H(H(H))(H(H)))))
    if only_duplicated=="Duplicated":
        df1=df[["iden_pep"]].loc[df["iden_pep"].duplicated()].drop_duplicates()
        df=df.loc[df["iden_pep"].isin(df1["iden_pep"])]
    # print("Waiting to process multiply glycopeptides")
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
    # print(iden_pep)
    mz_calc=masses.pepfragmass(iden_pep,FRAG_MODE,3) #iden_pep已经改成了glysite，避免多J的可能
    ppm=TOLER
    FragmentMz=[]
    for mz in mz_calc:
        for ion in mz:
            FragmentMz.append(list(ion.values())[0])
    FragmentMz=list(set(FragmentMz))
    # ipdb.set_trace()
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
    # print(mzdict_1)
    # import ipdb
    # ipdb.set_trace()
    return mzdict_1

def mz_matching(instance):
    spectrum=instance["GlySpec"]
    datalis_1=datalis.loc[datalis["title"]==spectrum]
    # ipdb.set_trace()
    datalis_1=datalis_1.reset_index(drop=True)
    iden_pep=instance["iden_pep"]
    # print(iden_pep)
    # ipdb.set_trace()
    mz_calc=masses.pepfragmass(iden_pep,["HCD_BY_2"],4) #iden_pep已经改成了glysite，避免多J的可能
    ppm=TOLER
    FragmentMz=[]
    # ipdb.set_trace()
    for mz in mz_calc:
        for ion in mz:
            FragmentMz.append(list(ion.values())[0])
    FragmentMz=list(set(FragmentMz))
    FragmentMz.sort()
    # ipdb.set_trace()
    mzexp=datalis_1["mz"][0]
    # mzexp=[round(num, 2) for num in mzexp]
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
    df=df[[ "GlySpec",'Charge','Peptide', 'Mod', 'PlausibleStruct', 'GlySite', 'iden_pep']]
    df.drop_duplicates(subset=["GlySpec",'Charge','Peptide', 'Mod', 'PlausibleStruct', 'GlySite', 'iden_pep'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    # df["matching_ration"]=df.apply(mz_matching,axis=1)
    print("len(df_iden_pep.drop_duplicates())",len(df["iden_pep"].drop_duplicates()))
    print("len(df)",len(df))
    df["ions"]=df.apply(fragment_training,axis=1)
    
    df.to_csv(output_name,index=False)