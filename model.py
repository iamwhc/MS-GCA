import torch
import torch.nn as nn
import dhg
import warnings
import torch.nn.init as init
from dhg.nn import UniGATConv, UniSAGEConv, UniGCNConv

warnings.filterwarnings("ignore")
import torch.nn.functional as F
from dhg.structure.graphs import Graph
from dhg.structure.hypergraphs import Hypergraph

class HGNNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act = nn.LeakyReLU(0.25)
        self.act2 = nn.ReLU()
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        init.kaiming_normal_(self.theta.weight)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.act2(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X

class GCNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        init.kaiming_normal_(self.theta.weight)

    def forward(self, X: torch.Tensor, g: Graph) -> torch.Tensor:
        X = self.theta(X).double()
        X = g.smoothing_with_GCN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X.float()

class GraphFeatureAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        attention_weights = F.softmax(torch.matmul(node_features, node_features.T), dim=-1)
        weighted_feature = torch.matmul(attention_weights, node_features)
        graph_feature = torch.mean(weighted_feature, dim=0, keepdim=True)
        return graph_feature

class Attn_Net_Gated(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # 门控注意力分支 a：Linear + Tanh
        self.attention_a = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # 门控注意力分支 b：Linear + Sigmoid
        self.attention_b = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # 注意力分数输出层
        self.attention_c = nn.Linear(hidden_dim, 1)

        # 特征输出投影（保持与原始模型的输出维度一致）
        self.out_proj = nn.Linear(in_features, in_features)

        # 初始化权重
        init.kaiming_normal_(self.attention_a[0].weight)
        init.kaiming_normal_(self.attention_b[0].weight)
        init.kaiming_normal_(self.attention_c.weight)
        init.kaiming_normal_(self.out_proj.weight)

    def forward(self, node_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, M = node_features.size()  # N: num_nodes, M: in_features

        # 门控注意力计算
        a = self.attention_a(node_features)  # N x hidden_dim
        b = self.attention_b(node_features)  # N x hidden_dim
        A = a * b  # N x hidden_dim
        attention_scores = self.attention_c(A)  # N x 1

        # Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=0)  # N x 1

        # 计算加权特征
        weighted_feature = torch.matmul(attention_weights.t(), node_features)  # 1 x M
        graph_feature = self.out_proj(weighted_feature)  # 1 x M

        # 节点注意力分数
        node_attention_scores = attention_weights.squeeze(1)  # N

        return graph_feature, node_attention_scores

class HGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: list,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.3,
        feature_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.hgnn = nn.ModuleList([
            UniGATConv(in_channels, hid_channels[0], use_bn=use_bn, drop_rate=drop_rate),
            UniGATConv(hid_channels[0], in_channels, use_bn=use_bn, is_last=True)
        ])
        self.feature_aggregator = GraphFeatureAggregator()
        self.final_layer = nn.Linear(6 * in_channels, num_classes)
        self.bn = nn.LayerNorm(in_channels)
        self.bn2 = nn.LayerNorm(6 * in_channels)
        self.feature_aggregator2 = Attn_Net_Gated(in_features=in_channels)

    def process_single_input(self, x, hg):
        for layer in self.hgnn:
            x = layer(x, hg)
        y = hg.v2e(x, 'softmax_then_sum')
        x = self.feature_aggregator2(x)
        y = self.feature_aggregator2(y)
        x = self.bn(x)
        y = self.bn(y)
        return x, y

    def forward(self, X: list, hg: list) -> torch.Tensor:
        x_outputs = []
        y_outputs = []
        for i in range(3):
            x, y = self.process_single_input(X[i], hg[i])
            x_outputs.append(x)
            y_outputs.append(y)
        X_all = torch.cat(x_outputs, dim=1)
        Y_all = torch.cat(y_outputs, dim=1)
        X = torch.cat([X_all, Y_all], dim=1)
        X = self.bn2(X)
        X = self.final_layer(X)
        return X

class SingleChannelMoudel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hid_channels: list,
            num_classes: int,
            use_bn: bool = False,
            drop_rate: float = 0.3,
            num_heads: int = 8,
            reduced_dim: int = 512,
            used_indices=None,
    ):
        super().__init__()
        if used_indices is None:
            used_indices = [0, 1, 2]
        self.in_channels = in_channels
        self.used_indices = used_indices
        self.num_inputs = len(used_indices)
        self.hgnn = nn.ModuleList([
            HGNNConv(in_channels, hid_channels[0], use_bn=use_bn, drop_rate=drop_rate, is_last=True),
            HGNNConv(hid_channels[0], 64, use_bn=use_bn, is_last=True)
        ])
        self.gnn = nn.ModuleList([
            GCNConv(in_channels, hid_channels[0], use_bn=use_bn, drop_rate=drop_rate, is_last=True),
            GCNConv(hid_channels[0], 64, use_bn=use_bn, is_last=True)
        ])
        total_dim = 2 * self.num_inputs * 64
        self.feature_aggregator = Attn_Net_Gated(in_features=64)
        self.bn = nn.LayerNorm(64)
        self.bn2 = nn.LayerNorm(total_dim)
        self.pool = nn.AdaptiveAvgPool1d(reduced_dim)
        self.final_layer = nn.Linear(total_dim, num_classes)

    def forward(self, X_H: list, X_G: list, hg: list, g: list) -> tuple[torch.Tensor, torch.Tensor]:
        out = []
        hgnn_attention_scores = []

        # 处理 HGNN 的输入
        for idx in self.used_indices:
            x = X_H[idx]
            for layer in self.hgnn:
                x = layer(x, hg[idx])
            x_agg, x_scores = self.feature_aggregator(x)
            x_agg = self.bn(x_agg)
            out.append(x_agg)
            hgnn_attention_scores.append(x_scores)

        # 处理 GNN 的输入
        for idx in self.used_indices:
            y = X_G[idx]
            for layer in self.gnn:
                y = layer(y, g[idx])
            y_agg, _ = self.feature_aggregator(y)
            y_agg = self.bn(y_agg)
            out.append(y_agg)

        # 组合输出
        X = torch.cat(out, dim=1)
        X = self.bn2(X)
        X = self.final_layer(X)

        # 超图注意力分数平均
        if self.num_inputs > 0:
            total_hgnn_scores = torch.stack(hgnn_attention_scores, dim=0).mean(dim=0)
        else:
            total_hgnn_scores = torch.zeros_like(X_H[0][:,0])

        return X, total_hgnn_scores

class CrossNet(nn.Module):
    def __init__(self,
        in_channels: int,
        hid_channels: list,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hgnn = nn.ModuleList([
            HGNNConv(in_channels, hid_channels[0], use_bn=use_bn, drop_rate=drop_rate),
            HGNNConv(hid_channels[0], feature_dim, use_bn=use_bn, is_last=True)
        ])
        self.gnn = nn.ModuleList([
            GCNConv(in_channels, hid_channels[0], use_bn=use_bn, drop_rate=drop_rate),
            GCNConv(hid_channels[0], feature_dim, use_bn=use_bn, is_last=True)
        ])
        self.feature_aggregator = GraphFeatureAggregator()
        self.Linner = nn.Linear(6 * feature_dim, feature_dim)
        self.final_layer = nn.Linear(6 * feature_dim, num_classes)
        self.act = nn.ELU()
        self.bn = nn.LayerNorm(feature_dim)
        self.bn2 = nn.LayerNorm(6 * feature_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, X_H: list, X_G: list, hg: list, g: list) -> torch.Tensor:
        outputs = []
        for i in range(3):
            x = X_H[i]
            for layer in self.hgnn:
                x = layer(x, hg[i])
            x = self.feature_aggregator(x)
            x = self.bn(x)
            outputs.append(x)

        for i in range(3):
            x = X_G[i]
            for layer in self.gnn:
                x = layer(x, g[i])
            x = self.feature_aggregator(x)
            x = self.bn(x)
            outputs.append(x)

        X = torch.cat(outputs, dim=1)
        X = self.bn2(X)
        X = self.final_layer(X)
        return X

class NewCrossNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hid_channels: list,
            num_classes: int,
            use_bn: bool = False,
            drop_rate: float = 0.3,
            num_heads: int = 8,
            reduced_dim: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hgnn = nn.ModuleList([
            HGNNConv(in_channels, hid_channels[0], use_bn=use_bn, drop_rate=drop_rate),
            HGNNConv(hid_channels[0], 64, use_bn=use_bn, is_last=True)
        ])
        self.gnn = nn.ModuleList([
            GCNConv(in_channels, hid_channels[0], use_bn=use_bn, drop_rate=drop_rate),
            GCNConv(hid_channels[0], 64, use_bn=use_bn, is_last=True)
        ])
        total_dim = 6 * 64
        self.feature_aggregator = Attn_Net_Gated(in_features=64)
        self.bn = nn.LayerNorm(64)
        self.bn2 = nn.LayerNorm(total_dim)
        self.pool = nn.AdaptiveAvgPool1d(reduced_dim)
        self.final_layer = nn.Linear(total_dim, num_classes)

    def forward(self, X_H: list, X_G: list, hg: list, g: list) -> torch.Tensor:
        outputs = []
        for i in range(3):
            x = X_H[i]
            for layer in self.hgnn:
                x = layer(x, hg[i])
            x_agg = self.feature_aggregator(x)
            x_agg = self.bn(x_agg)
            outputs.append(x_agg)

        for i in range(3):
            x = X_G[i]
            for layer in self.gnn:
                x = layer(x, g[i])
            x_agg = self.feature_aggregator(x)
            x_agg = self.bn(x_agg)
            outputs.append(x_agg)

        X = torch.cat(outputs, dim=1)
        X = self.bn2(X)
        X = self.final_layer(X)
        return X

if __name__ == "__main__":
    X1 = torch.rand(100, 64)
    X2 = torch.rand(100, 64)
    X3 = torch.rand(100, 64)
    X_H = [X1, X2, X3]
    X_G = [X1, X2, X3]

    H1 = dhg.random.hypergraph_Gnm(100, 100)
    H2 = dhg.random.hypergraph_Gnm(100, 100)
    H3 = dhg.random.hypergraph_Gnm(100, 100)
    H = [H1, H2, H3]

    G1 = dhg.random.graph_Gnm(100, 10)
    G2 = dhg.random.graph_Gnm(100, 10)
    G3 = dhg.random.graph_Gnm(100, 10)
    G = [G1, G2, G3]

    # 测试单输入
    model1 = SingleChannelMoudel(64, [64, 128, 512], 2, used_indices=[0])
    output1, hgnn_scores1 = model1(X_H, X_G, H, G)
    print("单输入输出尺寸:", output1.size())
    print("超图注意力分数尺寸（单输入）:", hgnn_scores1.size())
    print("超图分数范围:", hgnn_scores1.min().item(), "到", hgnn_scores1.max().item())

    # 测试双输入
    model2 = SingleChannelMoudel(64, [64, 128, 512], 2, used_indices=[0, 2])
    output2, hgnn_scores2 = model2(X_H, X_G, H, G)
    print("双输入输出尺寸:", output2.size())
    print("超图注意力分数尺寸（双输入）:", hgnn_scores2.size())
    print("超图分数范围:", hgnn_scores2.min().item(), "到", hgnn_scores2.max().item())

    # 测试三输入
    model3 = SingleChannelMoudel(64, [64, 128, 512], 2, used_indices=[0, 1, 2])
    output3, hgnn_scores3 = model3(X_H, X_G, H, G)
    print("三输入输出尺寸:", output3.size())
    print("超图注意力分数尺寸（三输入）:", hgnn_scores3.size())
    print("超图分数范围:", hgnn_scores3.min().item(), "到", hgnn_scores3.max().item())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print("模型1参数量:", count_parameters(model1))
    print("模型2参数量:", count_parameters(model2))
    print("模型3参数量:", count_parameters(model3))