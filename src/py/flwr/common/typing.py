# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower type definitions."""


from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

Weights = List[np.ndarray]

# The following union type contains Python types corresponding to ProtoBuf types that
# ProtoBuf considers to be "Scalar Value Types", even though some of them arguably do
# not conform to other definitions of what a scalar is. Source:
# https://developers.google.com/protocol-buffers/docs/overview#scalar
Scalar = Union[bool, bytes, float, int, str]

Metrics = Dict[str, Scalar]
MetricsAggregationFn = Callable[[List[Tuple[int, Metrics]]], Metrics]

Config = Dict[str, Scalar]
Properties = Dict[str, Scalar]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PARAMETERS_NOT_IMPLEMENTED = 1


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str


@dataclass
class ParametersRes:
    """Response when asked to return parameters."""

    parameters: Parameters

@dataclass
class SampleLatency:
    """."""

    parameters: Parameters
    config: Dict[str, Scalar]

@dataclass
class SampleLatencyRes:
    """."""

    msg_receive_time: float
    msg_sent_time: float
    train_time_per_epoch: float
    train_time_per_batch: float
    inference_time: float
    parameters: Parameters
    num_examples: int
    train_time_per_epoch_list: List[float]
    train_time_per_batch_list: List[float]

@dataclass
class DeviceInfoRes:
    """."""

    device_id: int

@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]
    sampleloss: List[float]

@dataclass
class FitRes:
    """Fit response from a client."""

    parameters: Parameters
    num_examples: int
    metrics: Dict[str, Scalar]
    loss_min: float
    loss_max: float
    loss_square_sum: float
    overthreshold_loss_count: int
    loss_sum: float
    loss_count: int
    train_time_per_epoch: float
    train_time_per_batch: float
    train_time_per_epoch_list: List[float]
    train_time_per_batch_list: List[float]

@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    loss: float
    num_examples: int
    metrics: Dict[str, Scalar]


@dataclass
class PropertiesIns:
    """Properties requests for a client."""

    config: Config


@dataclass
class PropertiesRes:
    """Properties response from a client."""

    status: Status
    properties: Properties


@dataclass
class Reconnect:
    """Reconnect message from server to client."""

    seconds: Optional[int]


@dataclass
class Disconnect:
    """Disconnect message from client to server."""

    reason: str
