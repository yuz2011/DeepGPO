#选择高置信度谱图
import pandas as pd
import ipdb
import numpy as np
import os
#训练数据集选择，选择pglyco数据置信谱图：1.只有一个肽段加电荷鉴定结果，不考虑共流出谱图2.glyionratio差值大于0 3.不考虑肽段上多可能位点 4.searching FDR<0.1

#YYglyco
#Raw文件融合
def find_raw_subfolders(folder_path):
    raw_subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir == "Raw_O":
                raw_subfolders.append(os.path.join(root, dir))
    return raw_subfolders

folder_path = "/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/NO/HCD/PXD020077/"
raw_subfolders = find_raw_subfolders(folder_path)
# ipdb.set_trace()
output_file = "pGlycoDB-GP-Raw1_all_O.txt"
for folder_path in raw_subfolders:
    file_list = [file for file in os.listdir(folder_path) if file.startswith("pGlycoDB-GP-Raw") and file.endswith(".txt")]
    print(file_list)
    print(len(file_list))
    merged_df = pd.DataFrame()
    for file in file_list:
        # ipdb.set_trace()
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, delimiter='\t')  
        merged_df = merged_df.append(df, ignore_index=True)
    merged_df.to_csv(folder_path+"/"+output_file, sep='\t', index=False)
ipdb.set_trace()

def find_files(folder_path, file_name):
    found_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == file_name:
                found_files.append(os.path.join(root, file))
    return found_files

folder_path = "/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/NO/PXD024995/"
file_name = "pGlycoDB-GP-Raw1_all_O.txt"
found_files = find_files(folder_path, file_name)
for raw_all in found_files:
    print(raw_all)
    dfpglyco=pd.read_csv(raw_all,sep="\t")
    dfpglyco["GlySpec"]=dfpglyco["RawName"]+"-"+dfpglyco["Scan"].map(str)
    diff = dfpglyco.groupby(["GlySpec"])["GlyIonRatio"].apply(lambda x: abs(x.nlargest(2).diff().iloc[-1])).reset_index()
    diff.rename(columns={"GlyIonRatio":"diff"}, inplace=True)
    sorted_diff = diff.sort_values("diff", ascending=False)
    #diff不应该直接去掉na
    # sorted_diff.dropna(subset=["diff"],inplace=True)
    sorted_diff=sorted_diff[sorted_diff["diff"]>0]
    #不应该在这里去掉count重复谱图
    # ipdb.set_trace()
    last_slash_index = raw_all.rfind("/")
    second_last_slash_index = raw_all.rfind("/", 0, last_slash_index)
    plycoresultpath=raw_all[:second_last_slash_index+1]
    files = os.listdir(plycoresultpath)
    filtered_files = [file for file in files if file.startswith("pGlycoDB-GP-FDR-Pro_PXD024996_O_1") and file.endswith(".txt")]
    filtered_files = [string for string in filtered_files if not string.endswith("redo1_filtered.txt")]
    filtered_files = [string for string in filtered_files if not string.endswith("decoy.txt")]
    assert len(filtered_files)==1
    filtered_files=filtered_files[0]
    print(filtered_files)
    # ipdb.set_trace()
    pglycores=pd.read_csv(os.path.join(plycoresultpath, filtered_files),sep="\t")
    columns=pglycores.columns
    length_or=len(pglycores)
    pglycores["title"]=pglycores["RawName"]+"-"+pglycores["Scan"].map(str)
    pglycores=pglycores.drop_duplicates(subset='title', keep=False)
    pglycores=pglycores[pglycores["title"].isin(sorted_diff["GlySpec"])]
    print("the ratio that fileter df/ original df",len(pglycores)/length_or)
    # ipdb.set_trace()
    pglycores=pglycores[columns]
    out=os.path.join(plycoresultpath, filtered_files)[:-4]+"_redo1_filtered.txt"
    pglycores.to_csv(out,index=False,sep="\t")

ipdb.set_trace()
def find_raw_subfolders(folder_path):
    raw_subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir == "Raw":
                raw_subfolders.append(os.path.join(root, dir))
    return raw_subfolders

# folder_path = "/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/human/"
# raw_subfolders = find_raw_subfolders(folder_path)
# # ipdb.set_trace()
# output_file = "pGlycoDB-GP-Raw1_all.txt"
# for folder_path in raw_subfolders:
#     file_list = [file for file in os.listdir(folder_path) if file.startswith("pGlycoDB-GP-Raw") and file.endswith(".txt")]
#     print(file_list)
#     merged_df = pd.DataFrame()
#     for file in file_list:
#         # ipdb.set_trace()
#         file_path = os.path.join(folder_path, file)
#         df = pd.read_csv(file_path, delimiter='\t')  
#         merged_df = merged_df.append(df, ignore_index=True)
#     # ipdb.set_trace()
#     merged_df.to_csv(folder_path+"/"+output_file, sep='\t', index=False)

def find_files(folder_path, file_name):
    found_files = []
    # ipdb.set_trace()
    for root, dirs, files in os.walk(folder_path):
        # ipdb.set_trace()
        for file in files:
            if file == file_name:
                found_files.append(os.path.join(root, file))
    return found_files

folder_path = "/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/NO/PXD024995/"
file_name = "pGlycoDB-GP-Raw1_all.txt"
found_files = find_files(folder_path, file_name)
# ipdb.set_trace()
for raw_all in found_files:
    dfpglyco=pd.read_csv(raw_all,sep="\t")
    dfpglyco["GlySpec"]=dfpglyco["RawName"]+"-"+dfpglyco["Scan"].map(str)
    diff = dfpglyco.groupby(["GlySpec"])["GlyIonRatio"].apply(lambda x: abs(x.nlargest(2).diff().iloc[-1])).reset_index()
    diff.rename(columns={"GlyIonRatio":"diff"}, inplace=True)
    sorted_diff = diff.sort_values("diff", ascending=False)
    #diff不应该直接去掉na
    # sorted_diff.dropna(subset=["diff"],inplace=True)
    sorted_diff=sorted_diff[sorted_diff["diff"]>0]
    #不应该在这里去掉count重复谱图
    # ipdb.set_trace()
    last_slash_index = raw_all.rfind("/")
    second_last_slash_index = raw_all.rfind("/", 0, last_slash_index)
    plycoresultpath=raw_all[:second_last_slash_index+1]
    files = os.listdir(plycoresultpath)
    filtered_files = [file for file in files if  file.endswith("_data_1st.csv")]
    filtered_files = [string for string in filtered_files if not string.endswith("_redo1_filtered.txt")]
    filtered_files = [string for string in filtered_files if  "redo1" in string]
    # ipdb.set_trace()
    print(filtered_files)
    assert len(filtered_files)==1
    filtered_files=filtered_files[0]
    pglycores=pd.read_csv(os.path.join(plycoresultpath, filtered_files))
    # ipdb.set_trace()
    columns=pglycores.columns
    length_or=len(pglycores)
    pglycores=pglycores.drop_duplicates(subset='GlySpec', keep=False)
    pglycores=pglycores[pglycores["GlySpec"].isin(sorted_diff["GlySpec"])]
    print("the ratio that fileter df/ original df",len(pglycores)/length_or)
    # ipdb.set_trace()
    pglycores=pglycores[columns]
    out=os.path.join(plycoresultpath, filtered_files)[:-4]+"_redo1_filtered.csv"
    pglycores.to_csv(out,index=False)