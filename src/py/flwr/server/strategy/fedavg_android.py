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
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy with custom
serialization for Android devices.

Paper: https://arxiv.org/abs/1602.05629
"""


from numbers import Integral
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    SampleLatency,
    SampleLatencyRes,
    DeviceInfoRes,
    Scalar,
    Weights,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, weighted_loss_avg
from .strategy import Strategy

import json


class FedAvgAndroid(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        sample_loss_fn: Optional[
            Callable[[Weights], Optional[Tuple[int, float]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        latency_sampling=False,
        ss_baseline=False,
        fedprox=False,
        fedbalancer=False,
        fb_client_selection=False,
        ddl_baseline_fixed=True,
        ddl_baseline_fixed_value_multiplied_at_mean=1.0,
        num_epochs=5,
        batch_size=10,
        ddl_baseline_smartpc=False,
        ddl_baseline_wfa=False,
        clients_per_round=5,
        fb_p=0.0,
        lss=0.05,
        dss=0.05,
        w=20,
        total_client_num=21,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
                Optional function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, Scalar]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, Scalar]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Parameters, optional): Initial global model parameters.
        """
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.sample_loss_fn = sample_loss_fn

        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

        self.latency_sampling = latency_sampling
        self.ss_baseline=ss_baseline
        self.fedprox=fedprox
        self.fedbalancer=fedbalancer
        self.fb_client_selection=fb_client_selection
        self.ddl_baseline_fixed=ddl_baseline_fixed
        self.ddl_baseline_fixed_value_multiplied_at_mean=ddl_baseline_fixed_value_multiplied_at_mean
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.ddl_baseline_smartpc=ddl_baseline_smartpc
        self.ddl_baseline_wfa=ddl_baseline_wfa
        self.clients_per_round=clients_per_round

        self.fb_p=fb_p
        self.lss=lss
        self.dss=dss
        self.w=w

        self.total_client_num=total_client_num

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = self.parameters_to_weights(parameters)

        weights[0] = weights[0].reshape((16,9,192))
        weights[2] = weights[2].reshape((6144,256))
        weights[4] = weights[4].reshape((256,6))

        eval_res = self.eval_fn(weights)

        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, {"accuracy": metrics}
    
    def calculate_sample_loss(
        self, parameters: Parameters, x_test: np.ndarray, y_test: np.ndarray
    ) -> List[float]:
        """Evaluate model parameters using an evaluation function."""
        if self.sample_loss_fn is None:
            # No evaluation function provided
            return None
        weights = self.parameters_to_weights(parameters)

        weights[0] = weights[0].reshape((16,9,192))
        # weights[2] = weights[2].reshape((5376,256))
        weights[2] = weights[2].reshape((6144,256))
        weights[4] = weights[4].reshape((256,6))

        sample_loss_res = self.sample_loss_fn(weights, x_test, y_test)

        if sample_loss_res is None:
            return None
        return sample_loss_res

    # def configure_fit(
    #     self, rnd: int, parameters: Parameters, client_manager: ClientManager, clients_per_round: int, deadline: float, fedbalancer: bool
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager, clients_per_round: int, deadline: float, fedbalancer: bool, oort_non_pacer_deadline: float, clients_explored: dict
    ) -> List[Tuple[ClientProxy, Parameters, dict]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd, self.batch_size, self.num_epochs, deadline, self.fedprox, self.fedbalancer, self.fb_p, self.ss_baseline)
        # fit_ins = FitIns(parameters, config, [])

        if fedbalancer:
            clients = client_manager.fb_oort_sample(
                num_clients=clients_per_round, clients_explored=clients_explored, min_num_clients=clients_per_round, current_round = (rnd-1), # This is rnd-1 to align with how FB is implemented in FLASH
                oort_non_pacer_deadline=oort_non_pacer_deadline, num_epochs=self.num_epochs
            )
        else:
            clients = client_manager.sample(
                num_clients=clients_per_round, min_num_clients=clients_per_round
            )

        # Return client/config pairs
        return [(client, parameters, config) for client in clients]
    
    def configure_sample_latency(
        self, parameters: Parameters, client: ClientProxy
    ) -> List[Tuple[ClientProxy, SampleLatency]]:
        """Configure the next round of training."""
        config = {
            "batch_size": 10,
            "local_epochs": 5,
        }
        sample_latency = SampleLatency(parameters, config)

        # Return client/config pairs
        return [(client, sample_latency)]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction_eval is 0
        if self.fraction_eval == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes, int]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # for client, fit_res, time in results:
        #     print(self.parameters_to_weights(fit_res.parameters))
        # Convert results
        weights_results = [
            (self.parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res, time in results
        ]
        return self.weights_to_parameters(aggregate(weights_results)), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {}

    def weights_to_parameters(self, weights: Weights) -> Parameters:
        """Convert NumPy weights to parameters object."""
        tensors = [self.ndarray_to_bytes(ndarray) for ndarray in weights]
        return Parameters(tensors=tensors, tensor_type="numpy.nda")

    def parameters_to_weights(self, parameters: Parameters) -> Weights:
        """Convert parameters object to NumPy weights."""
        return [self.bytes_to_ndarray(tensor) for tensor in parameters.tensors]

    # pylint: disable=R0201
    def ndarray_to_bytes(self, ndarray: np.ndarray) -> bytes:
        """Serialize NumPy array to bytes."""
        return ndarray.tobytes()

    # pylint: disable=R0201
    def bytes_to_ndarray(self, tensor: bytes) -> np.ndarray:
        """Deserialize NumPy array from bytes."""
        ndarray_deserialized = np.frombuffer(tensor, dtype=np.float32)
        return cast(np.ndarray, ndarray_deserialized)
