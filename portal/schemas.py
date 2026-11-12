from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

Dataset = Literal['bloodmnist', 'organamnist', 'pathmnist']
Strategy = Literal['fedavg', 'fedbn', 'scaffold', 'fedbns', 'scr']
ACTIVE = {'preparing', 'waiting', 'starting', 'running', 'stopping', 'finalizing', 'awaiting_clients'}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Login(StrictModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class RunSpec(StrictModel):
    name: str = Field(default='Collaborative training', min_length=1, max_length=100)
    dataset: Dataset = 'bloodmnist'
    strategy: Strategy = 'scr'
    model: Literal['baseline_bn', 'baseline', 'cnn'] = 'baseline_bn'
    partition: Literal['dirichlet_0.3', 'dirichlet_0.1', 'dirichlet_1.0', 'practical'] = 'dirichlet_0.3'
    seed: int = Field(default=42, ge=0, le=2147483647)
    rounds: int = Field(default=3, ge=1, le=500)
    local_epochs: int = Field(default=1, ge=1, le=50)
    batch_size: int = Field(default=16, ge=2, le=1024)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    weight_decay: float = Field(default=0, ge=0, le=1)
    scr_gamma: float = Field(default=1, gt=0, le=10000)
    scr_head_lambda: float = Field(default=0.01, ge=0, le=100)
    round_timeout: int = Field(default=600, ge=30, le=3600)
    node_ids: list[str] = Field(min_length=1, max_length=12)

    @field_validator('node_ids')
    @classmethod
    def unique_nodes(cls, values):
        if len(values) != len(set(values)):
            raise ValueError('Each institution can participate only once.')
        return values


class PairRequest(StrictModel):
    token: str = Field(min_length=16, max_length=256)
    name: str = Field(min_length=1, max_length=80)
    instance_id: str = Field(min_length=1, max_length=100)
    device_type: Literal['pc', 'jetson_nano', 'jetson_orin'] = 'pc'


class ConnectRequest(StrictModel):
    coordinator_url: str = Field(min_length=8, max_length=250)
    token: str = Field(min_length=16, max_length=256)


class DatasetRequest(StrictModel):
    dataset: Dataset


class PredictRequest(StrictModel):
    model_id: str = Field(min_length=1, max_length=100)
    sample_index: int | None = Field(default=None, ge=0)
    image_base64: str | None = Field(default=None, max_length=7_000_000)


class ImportModel(StrictModel):
    path: str = Field(min_length=1, max_length=1000)
    dataset: Dataset
    model: Literal['baseline_bn', 'baseline', 'cnn'] = 'baseline_bn'
    strategy: Strategy = 'scr'
    selection_rule: Literal['final_round', 'best_test_round', 'validation_selected'] = 'final_round'
    source_run: str = Field(min_length=1, max_length=150)
    source_round: int = Field(ge=0)


class Envelope(BaseModel):
    """Stable response envelope; worker measurements remain extensible JSON."""
    data: object


class Heartbeat(StrictModel):
    capabilities: dict = Field(default_factory=dict)
    telemetry: dict = Field(default_factory=dict)
    participations: list[dict] = Field(default_factory=list, max_length=20)
    events: list[dict] = Field(default_factory=list, max_length=500)
    models: list[dict] = Field(default_factory=list, max_length=100)
