#难点：
#1.谱图数目非常少，严格的过滤规则会降低灵敏度。并且酶切和软件打分并不能给出非常置信的结果
#2.软件定位存在各种不准确的谱图，比如没有位点打分，或者多位点，需要去提高准确性
#3.StcE的酶切特性是有争议的，非常难以确定规则

#方法：
#为了不过度减少灵敏度，并保证一定程度上的准确性。
#相比于直接按照一定的规则去过滤谱图，并且这个过滤规则因为不确定是按照酶切还是位点打分，过于复杂
#所以改成给不同的谱图一个权重。这个权重是加在loss函数中的

#具体：
#给训练谱图加权重 是单个位点的概率分的平方
#满足酶切特点概率分乘以2，最少是1。
#整个肽段只有一个可能位点最少是1。
#多位点是0。报告结果与定位localized group不一致为0.

#note：除此以外，一个训练策略就是不去重
#这算不算few-shot,如果审稿人问为什么不做few-shot怎么办
Enzyme_dict={"None":[],"Trypsin":[],"OgpA":[1],"IMPa":[1],
             "SmE":[1,-1],"StcE":[1,-2],
             "AMsia":[1,-1],"FAsia":[1,-1],"WAsia":[1,-1],"YAsia":[1,-1],
             "AM0627":[1,-1],"F290A":[1,-1],"W149A":[1,-1],"Y287A":[1,-1]}
def multi(instance):
    if  type(instance)==float:
        return "not_found"
    elif ";" in instance:
        import ipdb
        ipdb.set_trace()
        return "Poss_multi"
    else:
        instance = instance.replace("{", "").replace("}", "")
        parts = instance.split(',')
        t1_num =  parts[0][1:]
        t2_num =  parts[1][1:]
        try:
            last_number = float(parts[-1])
            #概率是否需要平方
            # last_number=last_number*last_number
        except ValueError:
            return "最后一个元素不是数字"
        if t1_num != t2_num:

            return last_number / 2
        else:
            return last_number
def checksite(LocalizedSiteGroups,GlySite):
    if  type(LocalizedSiteGroups)==float:
        return True
    elif ";" in LocalizedSiteGroups:
        return True
    else:
        LocalizedSiteGroups = LocalizedSiteGroups.replace("{", "").replace("}", "")
        parts = LocalizedSiteGroups.split(',')
        t1_num =  int(parts[0][1:])
        t2_num =  int(parts[1][1:])
        if t1_num!=GlySite and t2_num!=GlySite:
            return False  
        else:
            return True  
def weights(instance, Enzyme):
    LocalizedSiteGroups=instance["LocalizedSiteGroups"]
    GlySite=instance["GlySite"]
    Peptide=instance["Peptide"]
    prob=multi(LocalizedSiteGroups)
    if prob=="Poss_multi":
        prob=0
        return prob
    elif not checksite(LocalizedSiteGroups,GlySite):
        prob=0
        return prob
    else:
        if not Enzyme in Enzyme_dict.keys():
            raise KeyError("Enzyme not in Enzyme_dict.")
        for i in Enzyme_dict[Enzyme]:
            if i<0:
                Peptide_length=len(Peptide)
                i=Peptide_length+i+1
            if i==GlySite:
                if prob=="not_found":
                    prob=1
                    return prob
                else:
                    prob= prob*2
                    prob=max(1,prob)
                    return prob
        if  prob=="not_found":
            #如果肽段只有一个位点，那么最少也为1
            ST_count=Peptide.count("S")+Peptide.count("T")
            if ST_count==1:
                prob=1
            else:
                prob=0
            return prob
        else:
            ST_count=Peptide.count("S")+Peptide.count("T")
            if ST_count==1:
                # prob= prob*2
                # prob=max(1,prob)
                prob=1
                return prob
            else:
                prob= prob
            return prob
# import pandas as pd
# df=pd.read_csv("/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/data/NO/PXD004590/PXD004590_Chr/pGlycoDB-GP-FDR-Pro-Quant-Site.txt",sep="\t")
# Enzyme = "None"  # or any other enzyme you want to use
# df["weights"] = df.apply(lambda x: weights(x, Enzyme), axis=1)
# import ipdb
# ipdb.set_trace()