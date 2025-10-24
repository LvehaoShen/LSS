import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from .layers import MLP, FC

try:
        from torch_scatter import scatter_mean  # type: ignore
except (ImportError, OSError):

        def scatter_mean(src, index=None, dim=0, dim_size=None):
                """Fallback implementation when torch_scatter is unavailable."""

                if index is None:
                        return torch.mean(src, dim=dim)

                if dim != 0:
                        raise RuntimeError(
                                "Fallback scatter_mean only supports dim=0. Install torch_scatter "
                                "for broader axis support." )

                if dim_size is None:
                        dim_size = int(torch.max(index).item()) + 1 if index.numel() > 0 else 0

                out_shape = (dim_size,) + tuple(src.shape[1:])
                out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

                index_expand = index.view(index.shape[0], *([1] * (src.dim() - 1))).expand_as(src)
                out.scatter_add_(0, index_expand, src)

                count_dtype = src.dtype if torch.is_floating_point(src) else torch.float32
                count = torch.zeros(dim_size, dtype=count_dtype, device=src.device)
                ones = torch.ones(index.shape[0], dtype=count_dtype, device=src.device)
                count.scatter_add_(0, index, ones)

                count = count.clamp_min(1).view(dim_size, *([1] * (src.dim() - 1)))
                return out / count

from torch_geometric.nn import GINConv, GINEConv, NNConv, GATConv, GraphConv, SAGEConv
from .GINlayers import NNGINConv, NNGINConcatConv

class Graph2Vec(nn.Module):
	def __init__(self, num_node_feat, num_edge_feat, g_hid, out_g_ch, dropout = True):
		super(Graph2Vec, self).__init__()
		nn1 = nn.Sequential(nn.Linear(num_node_feat, g_hid),
							nn.ReLU(),
							nn.Linear(g_hid, g_hid))
		nn2 = nn.Sequential(nn.Linear(g_hid, g_hid),
							nn.ReLU(),
							nn.Linear(g_hid, g_hid))
		nn3 = nn.Sequential(nn.Linear(g_hid, g_hid),
							nn.ReLU(),
							nn.Linear(g_hid, out_g_ch))
		self.cov1 = GINConv(nn = nn1)
		self.cov2 = GINConv(nn = nn2)
		self.cov3 = GINConv(nn = nn3)
		self.dropout = dropout


	def forward(self, x, edge_index, edge_attr):
		x = self.cov1(x = x, edge_index = edge_index)
		x = F.dropout(x, self.dropout, training=self.training)

		x = self.cov2(x=x, edge_index=edge_index)
		x = F.dropout(x, self.dropout, training=self.training)

		x = self.cov3(x=x, edge_index=edge_index)
		x = torch.unsqueeze(torch.sum(x, dim= 0), dim= 0)
		#x = scatter_mean(x, dim=0)
		return x


class DecomGNN(nn.Module):
	def __init__(self, args, num_node_feat, num_edge_feat):
		super(DecomGNN, self).__init__()
		self.num_node_feat = num_node_feat
		self.num_edge_feat = num_edge_feat
		self.num_layers = args.num_layers
		self.num_hid = args.num_g_hid
		self.num_e_hid = args.num_e_hid
		self.num_out = args.out_g_ch
		self.model_type = args.model_type
		self.dropout = args.dropout
		self.convs = nn.ModuleList()


		cov_layer = self.build_cov_layer(self.model_type)

		for l in range(self.num_layers):
			hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
			hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid

			if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT" \
					or self.model_type == "GCN" or self.model_type == "SAGE":
				self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
			elif self.model_type == "NN" or self.model_type == "NNGIN" or self.model_type == "NNGINConcat":
				self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim, self.num_e_hid))
			else:
				print("Unsupported model type!")


	def build_cov_layer(self, model_type):
		if model_type == "GIN":
			return lambda in_ch, hid_ch : GINConv(nn= nn.Sequential(
				nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)) )
		elif model_type == "GINE":
			return lambda in_ch, hid_ch : GINEConv(nn= nn.Sequential(
				nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)) )
		elif model_type == "NN":
			return lambda in_ch, hid_ch, e_hid_ch : NNConv(in_ch, hid_ch,
				nn= nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch),
								  nn.ReLU(), nn.Linear(e_hid_ch, in_ch * hid_ch)) )
		elif model_type == "NNGIN":
			return lambda in_ch, hid_ch, e_hid_ch : NNGINConv(
				edge_nn= nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(), nn.Linear(e_hid_ch, in_ch)),
				node_nn= nn.Sequential(nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)) )
		elif model_type == "NNGINConcat":
			return lambda in_ch, hid_ch, e_hid_ch : NNGINConcatConv(
				edge_nn=nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(), nn.Linear(e_hid_ch, in_ch)),
				node_nn=nn.Sequential(nn.Linear(in_ch * 2, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)) )
		elif model_type == "GAT":
			return GATConv
		elif model_type == "SAGE":
			return SAGEConv
		elif model_type == "GCN":
			return GraphConv
		else:
			print("Unsupported model type!")


	def forward(self, x, edge_index, edge_attr = None):

		for i in range(self.num_layers):
			if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT" \
					or self.model_type =="GCN" or self.model_type == "SAGE":
				x = self.convs[i](x, edge_index) # for GIN and GINE
			elif self.model_type == "NN" or self.model_type == "NNGIN" or self.model_type == "NNGINConcat":
				x = self.convs[i](x, edge_index, edge_attr)
			else:
				print("Unsupported model type!")

			if i < self.num_layers - 1:
				x = F.dropout(x, p = self.dropout, training=self.training)

		x = torch.unsqueeze(torch.sum(x, dim=0), dim=0)
		return x


class Attention(nn.Module):
	"""
	Simple Attention layer
	"""
	def __init__(self, n_expert, n_hidden, v_hidden):
		super(Attention, self).__init__()
		self.n_expert = n_expert
		self.n_hidden = n_hidden
		self.v_hidden = v_hidden
		self.w1 = nn.Parameter(torch.FloatTensor(self.n_hidden, self.v_hidden)) # [n_hid, v_hid]
		self.w2 = nn.Parameter(torch.FloatTensor(self.n_expert, self.n_hidden)) # [n_expert, n_hid]
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.w1.size(1))
		self.w1.data.uniform_(-stdv, stdv)
		stdv = 1. / math.sqrt(self.w2.size(1))
		self.w2.data.uniform_(-stdv, stdv)

	def forward(self, x):
		x = torch.transpose(x, 0, 1) # [out_g_ch, patch_num]

		support = F.tanh(self.w1.matmul(x)) # [n_hid, patch_num]

		output = F.softmax(self.w2.matmul(support), dim = 1) #[n_expert, patch_num]
		return output



class CardNet(nn.Module):
	def __init__(self, args, num_node_feat, num_edge_feat):
		super(CardNet, self).__init__()
		self.num_node_feat = num_node_feat
		self.num_edge_feat = num_edge_feat
		self.num_expert = args.num_expert
		self.out_g_ch = args.out_g_ch
		self.num_att_hid = args.num_att_hid
		self.num_mlp_hid = args.num_mlp_hid
		self.num_classes = args.max_classes
		self.multi_task = args.multi_task
		self.pool_type = args.pool_type

		self.graph2vec = DecomGNN(args, self.num_node_feat, self.num_edge_feat)
		self.att_layer = Attention(self.num_expert, self.num_att_hid, self.out_g_ch)
		self.mlp_in_ch = self.num_expert * self.out_g_ch if self.pool_type == "att" else self.out_g_ch

		self.mlp = MLP(in_ch= self.mlp_in_ch, hid_ch= self.num_mlp_hid, out_ch= 1)

		self.fc_hid = FC(in_ch= self.mlp_in_ch, out_ch=self.num_mlp_hid)
		self.fc_reg = FC(in_ch= self.num_mlp_hid, out_ch= 1)
		self.fc_cla = FC(in_ch=self.num_mlp_hid, out_ch= self.num_classes)

	def forward(self, decomp_x, decomp_edge_index, decomp_edge_attr):
		g, output_cla = None, None

		def _strip_leading_batch_dim(tensor):
			"""Remove a leading singleton batch dimension while preserving feature axes."""

			if not torch.is_tensor(tensor):
				return tensor
			if tensor.dim() > 0 and tensor.size(0) == 1 and tensor.dim() > 1:
				return tensor.squeeze(0)
			return tensor

		for x, edge_index, edge_attr in zip(decomp_x, decomp_edge_index, decomp_edge_attr):
			x = _strip_leading_batch_dim(x)
			edge_index = _strip_leading_batch_dim(edge_index)
			edge_attr = _strip_leading_batch_dim(edge_attr)
			if g is None:
				g = self.graph2vec(x, edge_index, edge_attr)
			else:
				g = torch.cat([g, self.graph2vec(x, edge_index, edge_attr)], dim = 0)
		#print(g.shape)
		# g: [patch_num, out_g_ch]
		if self.pool_type == "sum":
			g = torch.sum(g, dim=0)
			g = g.unsqueeze(dim=0)
		elif self.pool_type == "mean":
			g = torch.mean(g, dim= 0)
			g = g.unsqueeze(dim=0)
		elif self.pool_type == "max":
			g, _ = torch.max(g, dim= 0, keepdim=True)

		else:
			att_wights = self.att_layer(g) # [num_expert, patch_num]
			g = att_wights.matmul(g) # g: [num_expert, out_g_ch]
			g = g.view((1, self.num_expert * self.out_g_ch))


		if self.multi_task:
			hid_g = F.relu(self.fc_hid(g))
			output = self.fc_reg(hid_g)
			output_cla = F.log_softmax(self.fc_cla(hid_g), dim= 1)
		else:
			output = self.mlp(g)

		return output, output_cla
