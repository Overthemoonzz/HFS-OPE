'''aWELv
Reference:
	Hongzhi Liu, Yingpeng Du, and Zhonghai Wu. 2022. Generalized Ambiguity Decomposition for Ranking Ensemble Learning. 
	Journal of Machine Learning, Research 23, 88 (2022), 1–36.
'''
import torch.nn as nn
import torch

class AWELV(nn.Module):
	def __init__(self, user_num, model_num=6, hidden_size=32):
		super().__init__()
		self.uid_embeddings = nn.Embedding(user_num, hidden_size)
		self.model_embeddings = nn.Embedding(model_num, hidden_size)
		self.model_num = model_num

	def forward(self, scores, uids, mask=None):
		# scores: [B, L, M]; uids: [B]; mask: [B, L] or None
		h_u = self.uid_embeddings(uids)  # [B, H]
		dots = []
		for m in range(self.model_num):
			h_m = self.model_embeddings(torch.tensor([m], device=scores.device))  # [1, H]
			dots.append((h_u * h_m).sum(dim=1, keepdim=True))  # [B,1]
		w = torch.cat(dots, dim=1)  # [B, M]
		w = w.unsqueeze(1).repeat(1, scores.size(1), 1)  # [B, L, M]
		w = torch.softmax(w, dim=-1)
		ens = (w * scores).sum(dim=2)  # [B, L]
		if mask is not None:
			ens = ens * mask
		return ens, w # [B, L]， [B, L, M]