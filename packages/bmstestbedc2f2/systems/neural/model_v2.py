import typing

import xarray
import pandas
import torch


class LSTMModule(torch.nn.Module):
    def __init__(
        self, 
        input_dim: int, output_dim: int, 
        hidden_dims: tuple[int, int] = (128, 64),
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dims[0], 
            num_layers=2, batch_first=True,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class Model:
    class Input(xarray.DataArray):
        __slots__ = ()

        _dims_ = ['Batch', 'Time', 'Feature']
        _coords_ = {
            'Feature': [
                'power',
                *(
                    rf'{feature_name}.{room_id}'
                    for room_id in range(8)
                    for feature_name in (
                        'temperature', 
                        'temperature_delta',
                        'temperature_error',
                    )
                ),
            ]
        }

        def __new__(cls, data):
            return xarray.DataArray(
                data=data,
                dims=cls._dims_,
                coords=cls._coords_,
            )

        @classmethod
        def from_dataframe(cls, df: pandas.DataFrame):
            def reset_time_index_to_positional(
                df: pandas.DataFrame, 
                inplace: bool = False,
            ):
                df = df.reset_index(drop=True, inplace=inplace)
                df.index.names = ['Time']
                return df

            return cls(
                df
                .groupby('Batch')
                .apply(reset_time_index_to_positional)
                .to_xarray()
                .to_dataarray(dim='Feature')
                .transpose(*cls._dims_)
                .sel(cls._coords_)
            )

    class Output(xarray.DataArray):
        __slots__ = ()

        _dims_ = ['Batch', 'Feature']
        _coords_ = {
            'Feature': [
                'power',
                *(
                    rf'{feature_name}.{room_id}'
                    for room_id in range(8)
                    for feature_name in (
                        'temperature',
                    )
                ),
            ]
        }
    
        def __new__(cls, data):
            return xarray.DataArray(
                data=data,
                dims=cls._dims_,
                coords=cls._coords_,
            )
        
        @classmethod
        def from_dataframe(cls, df: pandas.DataFrame):
            return cls(
                df
                .droplevel('Time', axis='index')
                .to_xarray()
                .to_dataarray(dim='Feature')
                .transpose(*cls._dims_)
                .sel(cls._coords_)
            )

    def __init__(
        self, 
        lr: float = 1e-2, 
        weight_decay: float = 1e-4,
        device: str | None = None,
    ):
        self.device = device
        self.module = LSTMModule(
            input_dim=len(self.Input._coords_['Feature']),
            output_dim=len(self.Output._coords_['Feature']),
        )
        if self.device is not None:
            self.module = self.module.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion = torch.nn.functional.mse_loss

    class TrainResult(typing.TypedDict):
        output_pred: 'Model.Output'
        loss: float

    def train(self, input: Input, output: Output) -> TrainResult:
        input, output = self.Input(input), self.Output(output)
        self.optimizer.zero_grad()
        output_pred = self.module(
            torch.tensor(
                input.values, 
                dtype=torch.float32,
                device=self.device,
            )
        )
        loss = self.criterion(
            output_pred, 
            torch.tensor(
                output.values, 
                dtype=torch.float32,
                device=self.device,
            ),
        )
        loss.backward()
        self.optimizer.step()
        return self.TrainResult(output_pred=output_pred, loss=loss.item())

    def predict(self, input: Input) -> Output:
        input = self.Input(input)
        self.module.eval()
        with torch.no_grad():
            prediction = self.module(
                torch.tensor(
                    input.values, 
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        # TODO
        return self.Output(prediction.numpy(force=True))
    
    def save(self, file_like: torch.serialization.FILE_LIKE):
        torch.save(self.module.state_dict(), file_like)
    
    def restore(self, file_like: torch.serialization.FILE_LIKE):
        self.module.load_state_dict(
            torch.load(file_like, 'cpu', weights_only=True)
        )
        if self.device is not None:
            self.module.to(self.device)
