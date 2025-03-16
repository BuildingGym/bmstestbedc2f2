import typing

import xarray
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv


class RoomTemperatureGNNModule(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int, out_channels: int, 
        num_nodes: int,
        gcn_hidden_dim: int = 64,
        lstm_hidden_dim_per_node: int = 128,
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.gcn1 = GCNConv(in_channels, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        # TODO
        self.lstm = torch.nn.LSTM(
            gcn_hidden_dim * num_nodes, lstm_hidden_dim_per_node * num_nodes, 
            batch_first=True,
        )
        self.fc = torch.nn.Linear(lstm_hidden_dim_per_node, out_channels)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        #batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer_norm(x)

        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index) # batch, time, node, feature

        d_batch, d_time, d_node, d_feature = x.shape

        x = x.reshape((d_batch, d_time, d_node * d_feature))
        x, _ = self.lstm(x) # batch, time, hidden

        x = x[:, -1, :] # batch, hidden - last time
        x = x.reshape((d_batch, d_node, -1))
        return self.fc(x)


class RoomTemperatureModel:
    class Input(xarray.DataArray):
        __slots__ = ()

        _dims_ = ['Batch', 'Time', 'Room', 'Feature']
        _coords_ = {
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

        _dims_ = ['Batch', 'Room', 'Feature']
        _coords_ = {
            'Feature': ['temperature'],
        }
    
        def __new__(cls, data):
            return xarray.DataArray(
                data=data,
                dims=cls._dims_,
                coords=cls._coords_,
            )

    def __init__(
        self, 
        num_rooms: int,
        lr: float = 1e-2, 
        weight_decay: float = 1e-4,
    ):
        self.module = RoomTemperatureGNNModule(
            in_channels=len(self.Input._coords_['Feature']), 
            out_channels=len(self.Output._coords_['Feature']), 
            num_nodes=num_rooms,
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
            #'batch': torch.zeros(num_nodes, dtype=torch.long),
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
            output_pred = self.module(**self._generate_module_inputs(input))
        return self.Output(output_pred)

    def save(self, file_like: torch.serialization.FILE_LIKE):
        torch.save(self.module.state_dict(), file_like)
    
    def restore(self, file_like: torch.serialization.FILE_LIKE):
        self.module.load_state_dict(torch.load(file_like, weights_only=True))
