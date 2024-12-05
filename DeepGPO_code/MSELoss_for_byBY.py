from fastNLP import LossBase
import torch.nn.functional as F
import torch
class MSELoss_byBY(LossBase):
    r"""
    MSE损失函数

    :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
    :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` >`target`
    :param str reduction: 支持'mean'，'sum'和'none'.

    """
########################adding loss weights
    def __init__(self, pred_by=None,pred_BY=None,target_by=None,target_BY=None, reduction='mean'):
        super(MSELoss_byBY, self).__init__()
        self._init_param_map(pred_by=pred_by,pred_BY=pred_BY,target_by=target_by,target_BY=target_BY,weights="weights",graph_edges="graph_edges")
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
    def get_weighted_loss(self, pred_by,pred_BY, target_by,target_BY,graph_edges=None,weights=None):
        # pred_BY = torch.split(pred_BY, graph_edges.tolist())
        # contents=[BY.reshape(-1,1).squeeze() for BY in pred_BY]
        # max_len = max(map(len, contents))
        # tensor = torch.full((len(contents), max_len), fill_value=0,dtype=torch.float)
        # for i, content_i in enumerate(contents):
        #     tensor[i, :len(content_i)] = content_i
        # pred_BY=tensor.to(weights.device)

        # target_BY = torch.split(target_BY, graph_edges.tolist())
        # contents=[BY.reshape(-1,1).squeeze() for BY in target_BY]
        # max_len = max(map(len, contents))
        # tensor = torch.full((len(contents), max_len), fill_value=0,dtype=torch.float)
        # for i, content_i in enumerate(contents):
        #     tensor[i, :len(content_i)] = content_i
        # target_BY=tensor.to(weights.device)
########################adding loss weights
        byloss=F.mse_loss(input=pred_by.float(), target=target_by.float(), reduction='none').mean(dim=-1)
        BYloss=F.mse_loss(input=pred_BY.float(), target=target_BY.float(), reduction='none').mean(dim=-1)
        BY_weights=torch.repeat_interleave(weights, graph_edges)
        #BYloss=F.mse_loss(input=pred_BY.float(), target=target_BY.float(), reduction='none').mean(dim=-1)直接用小by的同样方式效果更好
        #BYloss=weights*BYloss
        byloss=weights*byloss
        BYloss=BY_weights*BYloss
        if self.reduction=="mean":
            byloss=torch.mean(byloss,dim=-1)
            # BYloss=(BYloss*weights).sum()/graph_edges.sum()
            BYloss=torch.mean(BYloss,dim=-1)
        elif self.reduction=="sum":
            byloss=torch.sum(byloss,dim=-1)
            BYloss=torch.sum(BYloss,dim=-1)
            # BYloss=(BYloss*weights).sum()
        else:#none
            pass
        return byloss,BYloss
    def get_loss(self, pred_by,pred_BY, target_by,target_BY,graph_edges=None,weights=None):

        if weights is not None:
            # import ipdb
            # ipdb.set_trace()
            byloss,BYloss=self.get_weighted_loss(pred_by,pred_BY, target_by,target_BY,graph_edges,weights)

        else:
            byloss=F.mse_loss(input=pred_by.float(), target=target_by.float(), reduction=self.reduction)
            BYloss=F.mse_loss(input=pred_BY.float(), target=target_BY.float(), reduction=self.reduction)
            # byloss=F.mse_loss(input=torch.log2(pred_by.float()+1), target=torch.log2(target_by.float()+1), reduction=self.reduction)
            # import ipdb
            # ipdb.set_trace()
            # print("loss ratio",BYloss/byloss)
        loss=50*byloss+BYloss

        # loss=byloss+0.02*BYloss
        return loss
