import typing

import xarray
import torch


class PowerLSTMModule(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 100, output_dim: int = 1):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 50),
            torch.nn.Linear(50, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class PowerLSTMModel:
    class Input(xarray.DataArray):
        __slots__ = ()

        _dims_ = ['Batch', 'Time', 'Feature']
        _coords_ = {
            'Batch': -1,
            'Time': -1,
            'Feature': ['power'],
        }
    
        def __new__(cls, data):
            return xarray.DataArray(
                data=data,
                dims=cls._dims_,
                coords=cls._coords_,
            )

    class Output(xarray.DataArray):
        __slots__ = ()

        _dims_ = ['Batch', 'Time', 'Feature']
        _coords_ = {
            'Batch': -1, 
            'Time': -1,
            'Feature': ['power'],
        }
    
        def __new__(cls, data):
            return xarray.DataArray(
                data=data,
                dims=cls._dims_,
                coords=cls._coords_,
            )

    def __init__(self, lr: float = 1e-2, weight_decay: float = 1e-4):
        self.module = PowerLSTMModule(
            input_dim=len(self.Input._coords_['Feature']),
            output_dim=len(self.Output._coords_['Feature']),
        )
        self.optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion = torch.nn.functional.mse_loss

    class TrainResult(typing.TypedDict):
        output_pred: 'PowerLSTMModel.Output'
        loss: float

    def train(self, input: Input, output: Output) -> TrainResult:
        input, output = self.Input(input), self.Output(output)
        self.optimizer.zero_grad()
        output_pred = self.module(torch.tensor(input.values, dtype=torch.float32))
        loss = self.criterion(
            output_pred, 
            torch.tensor(output.values, dtype=torch.float32),
        )
        loss.backward()
        self.optimizer.step()
        return self.TrainResult(output_pred=output_pred, loss=loss.item())

    def predict(self, input: Input) -> Output:
        input = self.Input(input)
        self.module.eval()
        with torch.no_grad():
            prediction = self.module(torch.tensor(input.values, dtype=torch.float32))
        return self.Output(prediction)