import typing

import xarray
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv, global_mean_pool


class RoomTemperatureGNNModule(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = 16):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, out_channels)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer_norm(x)
        x = torch.nn.functional.relu(self.gcn1(x, edge_index))
        x = torch.nn.functional.relu(self.gcn2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


class RoomTemperatureModel:
    class Input(xarray.DataArray):
        __slots__ = ()

        _dims_ = ['Batch', 'Time', 'Room', 'Feature']
        _coords_ = {
            'Batch': -1,
            'Time': -1,
            'Room': -1,
            'Feature': ['temperature', 'temperature_delta', 'temperature_error'],
        }
    
        def __new__(cls, data):
            return xarray.DataArray(
                data=data,
                dims=cls._dims_,
                coords=cls._coords_,
            )

    class Output(xarray.DataArray):
        __slots__ = ()

        _dims_ = ['Batch', 'Time', 'Room', 'Feature']
        _coords_ = {
            'Batch': -1, 
            'Time': -1,
            'Room': -1,
            'Feature': ['temperature'],
        }
    
        def __new__(cls, data):
            return xarray.DataArray(
                data=data,
                dims=cls._dims_,
                coords=cls._coords_,
            )

    def __init__(self, lr: float = 1e-2, weight_decay: float = 1e-4):
        self.module = RoomTemperatureGNNModule(
            in_channels=len(self.Input._coords_['Feature']), 
            out_channels=len(self.Output._coords_['Feature']), 
        )
        self.optimizer = torch.optim.Adam(
            self.module.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
        )
        self.criterion = torch.nn.MSELoss()

    def _generate_fully_connected_edge_index(self, num_nodes: int):
        adj_matrix = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        edge_index, _ = dense_to_sparse(adj_matrix)
        return edge_index

    def _generate_module_inputs(self, input: Input):
        input = self.Input(input)
        num_nodes = input.sizes['Room']
        return {
            'x': torch.tensor(input.values, dtype=torch.float32),
            'edge_index': self._generate_fully_connected_edge_index(num_nodes),
            'batch': torch.zeros(num_nodes, dtype=torch.long),
        }
    
    class TrainResult(typing.TypedDict):
        output_pred: 'RoomTemperatureModel.Output'
        loss: float
        
    def train(self, input: Input, output: Output) -> TrainResult:
        input, output = self.Input(input), self.Output(output)
        self.optimizer.zero_grad()
        output_pred = self.module(**self._generate_module_inputs(input))
        loss = self.criterion(
            output_pred, 
            torch.tensor(output.values, dtype=torch.float32),
        )
        loss.backward()
        self.optimizer.step()
        return self.TrainResult(output_pred=output_pred, loss=loss)

    def predict(self, input: Input) -> Output:
        input = self.Input(input)
        self.module.eval()
        with torch.no_grad():
            prediction = self.module(**self._generate_module_inputs(input))
        return self.Output(prediction)
