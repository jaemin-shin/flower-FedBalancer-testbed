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
"""Flower server."""


import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple
from xmlrpc.client import Boolean

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    SampleLatency,
    SampleLatencyRes,
    DeviceInfoRes,
    Parameters,
    Reconnect,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[BaseException],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[BaseException],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]],
    List[BaseException],
]

# For FedBalancer, experiment with deadline-based FL
from datetime import datetime, timedelta
import numpy as np
import sys
import json
import os
from collections import defaultdict
import ast

class Server:
    """Flower server."""

    def __init__(
        self, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None

        self.elapsed_time = timedelta(seconds=0)
        self.last_checked_time = datetime.now()
        
        self.deadline = 120
        self.oort_non_pacer_deadline = 120

        self.sample_latency_iteration = 10

        self.loss_threshold = 0.0
        self.loss_threshold_percentage = 0.0

        self.deadline_percentage = 1.0

        self.prev_train_losses = []
        self.current_round_loss_min = []
        self.current_round_loss_max = []

        self.current_round = 0

        self.train_data = {}
        self.clients_id_to_device_id = {}

        self.clients_explored = {}

        self.guard_time = 0

        self.round_exploited_utility = []

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])
        
        # Load client data if FedBalancer
        if self.strategy.fedbalancer or (self.strategy.ss_baseline):
            train_clients, train_groups, train_data, clients_id_to_device_id = read_dir('/mnt/sting/jmshin/FedBalancer/FLASH_jm/data/har_raw/data/train/')
            self.train_data = train_data
            self.clients_id_to_device_id = clients_id_to_device_id
        
        # FedBalancer experiment: Sampling user's download / train / upload latency for 10 rounds

        # client -> server, message with {message receive time on client, message sent time on client, train_time_per_epoch, train_time_per_batch}
        # download_time: (message receive time on client) - (message sent time on server)
        # upload_time: (message receive time on server) - (message sent time on client)

        all_clients = self._client_manager.sample(self.strategy.total_client_num) # TODO: SHOULD BE NUMBER OF CLIENTS PARTICIPATING, IN REAL EXPERIMENT IT IS 21

        # Get device info from clients
        device_ids = device_info(all_clients)
        for device_id in device_ids:
            self.clients_explored[device_id] = 0
        
        # Check if range(1,22) ids are all included
        # for tmp_id in range(1,22):
        #     if tmp_id not in device_ids:
        #         assert(1==0)

        if self.strategy.ddl_baseline_smartpc or self.strategy.ddl_baseline_wfa:
            self.deadline = 100000
        else:
            log(INFO, "Requesting latency sampling over all clients (" + str(self.strategy.total_client_num)+" clients currently)")

            # f = open('/home/jmshin/FedBalancer_realexperiment/flower/examples/android/log/latency_profile_final_210515_paddingsamemodel.log')
            # lines = f.readlines()

            # download_times = {}
            # upload_times = {}
            # per_epoch_train_times = {}
            # per_batch_train_times = {}
            # inference_times = {}
            # round_completion_times = {}
            # networking_times = {}

            # client_id = -1
            # for line in lines[1459:]:
            #     tmp = line.strip().split('|')
            #     if 'client_id' in tmp[-1].split(',')[0]:
            #         client_id = int(tmp[-1].split(',')[1])
            #     elif 'download_times' in tmp[-1].split(',')[0]:
            #         download_times[client_id] = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
            #     elif 'upload_times' in tmp[-1].split(',')[0]:
            #         upload_times[client_id] = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
            #     elif 'per_epoch_train_times' in tmp[-1].split(',')[0]:
            #         per_epoch_train_times[client_id] = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
            #     elif 'per_batch_train_times' in tmp[-1].split(',')[0]:
            #         per_batch_train_times[client_id] = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
            #     elif 'inference_times' in tmp[-1].split(',')[0]:
            #         inference_times[client_id] = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
            #     elif 'round_completion_times' in tmp[-1].split(',')[0]:
            #         round_completion_times[client_id] = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
            #     elif 'networking_times' in tmp[-1].split(',')[0]:
            #         networking_times[client_id] = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))

            if self.strategy.latency_sampling:
                for iteration in range(self.sample_latency_iteration):
                    all_clients = self._client_manager.sample(len(list(self._client_manager.clients)))
                    for client in all_clients:
                    
                        client_instructions = self.strategy.configure_sample_latency(
                            parameters=self.parameters, client=client
                        )

                        server_msg_sent_time = datetime.now()

                        results, failures, server_msg_receive_time = sample_latency_clients(
                            client_instructions,
                            max_workers=self.max_workers
                        )

                        # print(server_msg_sent_time, server_msg_receive_time)

                        if len(results) != 0 and len(results[0]) != 0:
                            # print("EPOCH LIST", results[0][1].train_time_per_epoch_list)

                            client.download_times.append(round(max(0, (datetime.strptime(results[0][1].msg_receive_time, "%Y-%m-%d %H:%M:%S.%f") - server_msg_sent_time).total_seconds()), 3))
                            client.upload_times.append(round(max(0, (server_msg_receive_time - datetime.strptime(results[0][1].msg_sent_time, "%Y-%m-%d %H:%M:%S.%f")).total_seconds()), 3))

                            # if results[0][1].train_time_per_epoch > 10.0:
                            #     results[0][1].train_time_per_epoch /= 5
                            #     results[0][1].train_time_per_batch /= 5
                            # client.per_epoch_train_times.append(round(results[0][1].train_time_per_epoch, 3))
                            # client.per_batch_train_times.append(round(results[0][1].train_time_per_batch, 3))

                            client.per_epoch_train_times.append(round(np.mean(results[0][1].train_time_per_epoch_list), 3))
                            client.per_batch_train_times.append(round(np.mean(results[0][1].train_time_per_batch_list), 3))

                            client.inference_times.append(round(results[0][1].inference_time, 3))

                            client.round_completion_times.append(round(((server_msg_receive_time) - (server_msg_sent_time)).total_seconds(), 3))
                            # client.networking_times.append(round(((server_msg_receive_time) - (server_msg_sent_time)).total_seconds() - results[0][1].train_time_per_epoch * self.strategy.num_epochs - results[0][1].inference_time, 3))
                            client.networking_times.append(round(((server_msg_receive_time) - (server_msg_sent_time)).total_seconds() - np.mean(results[0   ][1].train_time_per_epoch_list) * self.strategy.num_epochs - results[0][1].inference_time, 3))

                            log(INFO, "client_id,"+str(client.device_id))
                            log(INFO, "download_times,"+str(client.download_times))
                            log(INFO, "upload_times,"+str(client.upload_times))
                            log(INFO, "per_epoch_train_times,"+str(client.per_epoch_train_times))
                            log(INFO, "per_batch_train_times,"+str(client.per_batch_train_times))
                            log(INFO, "inference_times,"+str(client.inference_times))
                            log(INFO, "round_completion_times,"+str(client.round_completion_times))
                            log(INFO, "networking_times,"+str(client.networking_times))

                            # log(INFO, "client_id,"+str(client.device_id))
                            # log(INFO, "download_times,"+str(client.download_times)+" from log: "+str(np.mean(download_times[client.device_id])))
                            # log(INFO, "upload_times,"+str(client.upload_times)+" from log: "+str(np.mean(upload_times[client.device_id])))
                            # log(INFO, "per_epoch_train_times,"+str(client.per_epoch_train_times)+" from log: "+str(np.mean(per_epoch_train_times[client.device_id])))
                            # log(INFO, "per_batch_train_times,"+str(client.per_batch_train_times)+" from log: "+str(np.mean(per_batch_train_times[client.device_id])))
                            # log(INFO, "inference_times,"+str(client.inference_times)+" from log: "+str(np.mean(inference_times[client.device_id])))
                            # log(INFO, "round_completion_times,"+str(client.round_completion_times)+" from log: "+str(np.mean(round_completion_times[client.device_id])))
                            # log(INFO, "networking_times,"+str(client.networking_times)+" from log: "+str(np.mean(networking_times[client.device_id])))
                
            for client in all_clients:
                f = open('/home/jmshin/FedBalancer_realexperiment/flower/examples/android/log/client_'+str(client.device_id)+'_latency.log')
                lines = f.readlines()

                for line_idx in range(len(lines)-1, 0, -1):
                    tmp = lines[line_idx].strip().split('|')
                    print(tmp)
                    if 'client_id' in tmp[-1].split(',')[0]:
                        break
                    elif 'download_times' in tmp[-1].split(',')[0]:
                        download_times = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
                    elif 'upload_times' in tmp[-1].split(',')[0]:
                        upload_times = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
                    elif 'per_epoch_train_times' in tmp[-1].split(',')[0]:
                        per_epoch_train_times = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
                    elif 'per_batch_train_times' in tmp[-1].split(',')[0]:
                        per_batch_train_times = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
                    elif 'inference_times' in tmp[-1].split(',')[0]:
                        inference_times = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
                    elif 'round_completion_times' in tmp[-1].split(',')[0]:
                        round_completion_times = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
                    elif 'networking_times' in tmp[-1].split(',')[0]:
                        networking_times = ast.literal_eval(','.join(tmp[-1].split(',')[1:]))
                
                log(INFO, "CLIENT ID: " + str(client.device_id) + " LATENCY SAMPLE LENGTH: " + str(len(round_completion_times)))
                
                client.download_times += download_times
                client.upload_times += upload_times
                client.per_epoch_train_times += per_epoch_train_times
                client.per_batch_train_times += per_batch_train_times
                client.inference_times += inference_times
                client.round_completion_times += round_completion_times
                client.networking_times += networking_times

                # for client in all_clients:
                    # client.download_time /= self.sample_latency_iteration
                    # client.upload_time /= self.sample_latency_iteration
                    # client.per_epoch_train_time /= self.sample_latency_iteration
                    # client.per_batch_train_time /= self.sample_latency_iteration
                    # client.inference_time /= self.sample_latency_iteration

                    # client.round_completion_time /= self.sample_latency_iteration
                    # client.networking_time /= self.sample_latency_iteration

                    # print(client.download_time, client.upload_time, np.mean(client.per_epoch_train_times), np.mean(client.per_batch_train_times), client.inference_time)

            # Configure deadline based on the calculated latency
            if self.strategy.ddl_baseline_fixed or self.strategy.fedbalancer:
                round_duration_summ_list = []
                for c in all_clients:
                    if self.strategy.fedbalancer:                        
                        round_duration_summ_list.append(np.mean(c.round_completion_times))
                        # round_duration_summ_list.append(np.mean(round_completion_times[c.device_id]))
                    else:
                        if self.strategy.ss_baseline and c.device_id in [13, 16, 17, 21]:
                            big_clients_data_len = {}
                            big_clients_data_len[13] = 408
                            big_clients_data_len[16] = 409
                            big_clients_data_len[17] = 392
                            big_clients_data_len[21] = 383

                            # print(big_clients_data_len[c.device_id], (big_clients_data_len[c.device_id] - 382), (big_clients_data_len[c.device_id] - 382) / big_clients_data_len[c.device_id])
                            minus_ratio = (big_clients_data_len[c.device_id] - 382) / big_clients_data_len[c.device_id]
                            print("MINUS RATIO", np.mean(c.per_epoch_train_times)*self.strategy.num_epochs*minus_ratio)
                            round_duration_summ_list.append(np.mean(c.round_completion_times) - np.mean(c.inference_times) - np.mean(c.per_epoch_train_times)*self.strategy.num_epochs*minus_ratio)
                        else:
                            round_duration_summ_list.append(np.mean(c.round_completion_times) - np.mean(c.inference_times))
                        # round_duration_summ_list.append(np.mean(round_completion_times[c.device_id]) - np.mean(inference_times[c.device_id]))
                if self.strategy.ddl_baseline_fixed:
                    self.deadline = (np.mean(round_duration_summ_list) * self.strategy.ddl_baseline_fixed_value_multiplied_at_mean)
                if self.strategy.fedbalancer:
                    self.deadline = (np.mean(round_duration_summ_list))
            
            # print(round_duration_summ_list)
            log(INFO, "Deadline {}".format(self.deadline))
            self.oort_non_pacer_deadline = self.deadline
        
        if self.strategy.latency_sampling:
            assert(1==0)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        self.elapsed_time = timedelta(seconds=0)
        self.last_checked_time = datetime.now()

        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            self.current_round = current_round - 1 # This is just to align with how we implemented at FLASH simulation
            # Train model and replace previous global model
            res_fit = self.fit_round(rnd=current_round)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            # res_fed = self.evaluate_round(rnd=current_round)
            # if res_fed:
            #     loss_fed, evaluate_metrics_fed, _ = res_fed
            #     if loss_fed:
            #         history.add_loss_distributed(rnd=current_round, loss=loss_fed)
            #         history.add_metrics_distributed(
            #             rnd=current_round, metrics=evaluate_metrics_fed
            #         )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
        self, rnd: int
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "evaluate_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "evaluate_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
        )
        log(
            DEBUG,
            "evaluate_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(rnd, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)
    
    def filter_if_more_than_ratio_is_explored(self, ratio):
        print(self.clients_explored)
        return sum(list(self.clients_explored.values())) >= len(list(self.clients_explored.keys())) * ratio

    def output_current_round_deadline(self, selected_clients, num_epochs):
        t_max = sys.maxsize
        total_user_count = len(selected_clients)

        complete_user_counts_per_time = []
        max_complete_user_counts_per_time = -1
        max_complete_user_counts_per_time_idx = -1

        client_complete_time = {}

        for c in selected_clients:
            if len(c.per_epoch_train_times) >= 5:
                client_complete_time[c.device_id] = np.mean(c.networking_times) + np.mean(c.per_epoch_train_times[-5:]) * num_epochs + self.guard_time
            else:
                client_complete_time[c.device_id] = np.mean(c.networking_times) + np.mean(c.per_epoch_train_times) * num_epochs + self.guard_time

        for i in range(1, t_max):
            complete_user_count = 0
            for c in selected_clients:
                if client_complete_time[c.device_id] <= i:
                    complete_user_count += 1
            complete_user_counts_per_time.append(complete_user_count/(i))
            
            if max_complete_user_counts_per_time < complete_user_count/(i):
                max_complete_user_counts_per_time = complete_user_count/(i)
                max_complete_user_counts_per_time_idx = i
            
            if complete_user_count == total_user_count:
                break
            
        return max_complete_user_counts_per_time_idx
    
    def update_fb_variables(self, client_instructions):
        if self.loss_threshold == 0 and self.loss_threshold_percentage == 0:
            self.loss_threshold = 0
            log(INFO, "loss_threshold {}".format(self.loss_threshold))
        else:
            loss_low = np.min(self.current_round_loss_min)
            loss_high = np.mean(self.current_round_loss_max)
            self.loss_threshold = loss_low + (loss_high - loss_low) * self.loss_threshold_percentage
            log(INFO, 'loss_low {}, loss_high {}, loss_threshold {}'.format(loss_low, loss_high, self.loss_threshold))
        
        if self.filter_if_more_than_ratio_is_explored(0.5):
            selected_clients = [elem[0] for elem in client_instructions]
            deadline_low = self.output_current_round_deadline(selected_clients, 1)
            deadline_high = self.output_current_round_deadline(selected_clients, self.strategy.num_epochs)

            self.deadline = deadline_low + (deadline_high - deadline_low) * self.deadline_percentage
            log(INFO, 'deadline_low {}, deadline_high {}, deadline {}'.format(deadline_low, deadline_high, self.deadline))
        
        log(INFO, 'loss_threshold_percentage {}, deadline_percentage {}'.format(self.loss_threshold_percentage, self.deadline_percentage))

    def update_fb_variable_percentages(self, results):
        loss_sum_list = [fit_res.loss_sum for client, fit_res, time in results]
        loss_count_list = [fit_res.loss_count for client, fit_res, time in results]
        loss_min_list = [fit_res.loss_min for client, fit_res, time in results]
        loss_max_list = [fit_res.loss_max for client, fit_res, time in results]

        self.current_round_loss_min = loss_min_list
        self.current_round_loss_max = loss_max_list
        
        # print(sum(loss_sum_list), sum(loss_count_list), self.deadline)
        current_round_loss = (sum(loss_sum_list) / sum(loss_count_list)) / self.deadline

        self.prev_train_losses.append(current_round_loss)

        # print(self.current_round, int(self.strategy.w), len(self.prev_train_losses))

        if self.current_round % int(self.strategy.w) == int(self.strategy.w) - 1:
            if len(self.prev_train_losses) >= 2 * self.strategy.w - 1:
                nonscaled_reward = (np.mean(self.prev_train_losses[-(self.strategy.w*2):-(self.strategy.w)]) - np.mean(self.prev_train_losses[-(self.strategy.w):]))
                # print("NONSCALED_REWARD", nonscaled_reward)
                log(INFO, "nonscaled reward {}".format(nonscaled_reward))
                if nonscaled_reward > 0:
                    if self.loss_threshold_percentage + self.strategy.lss <= 1:
                        self.loss_threshold_percentage += self.strategy.lss
                    if self.deadline_percentage - self.strategy.dss >= 0:
                        self.deadline_percentage -= self.strategy.dss
                else:
                    if self.loss_threshold_percentage - self.strategy.lss >= 0:
                        self.loss_threshold_percentage -= self.strategy.lss
                    if self.deadline_percentage + self.strategy.dss <= 1:
                        self.deadline_percentage += self.strategy.dss
    
    def update_elapsed_time(self, delta=None):
        curr_time = datetime.now()
        if delta == None:
            delta = curr_time - self.last_checked_time
        else:
            delta = timedelta(milliseconds=delta*1000)
        self.elapsed_time += delta
        self.last_checked_time = curr_time
    
    def calculate_clients_sample_loss(self, parameters, client_instructions):
        clients = [elem[0] for elem in client_instructions]
        for client in clients:
            x_train = np.array(self.train_data[self.clients_id_to_device_id[client.device_id - 1]]['x'])
            y_train = np.array(self.train_data[self.clients_id_to_device_id[client.device_id - 1]]['y']).reshape(-1,1)
            client.current_whole_data_loss_list = self.strategy.calculate_sample_loss(parameters=parameters, x_test=x_train, y_test=y_train)

    def fit_round(
        self, rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        pacer_window = 20
        if self.current_round > pacer_window * 2:
            if sum(self.round_exploited_utility[-(2*pacer_window):-(pacer_window)]) > sum(self.round_exploited_utility[-(pacer_window):]):
                self.deadline += 10

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager, clients_per_round=self.strategy.clients_per_round, deadline=self.deadline, fedbalancer=self.strategy.fedbalancer, oort_non_pacer_deadline=self.oort_non_pacer_deadline, clients_explored=self.clients_explored
        )

        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None

        # curr_time = datetime.now()
        # self.elapsed_time += curr_time - self.last_checked_time
        # self.last_checked_time = curr_time

        if self.strategy.fedbalancer:
            self.update_fb_variables(client_instructions)

        self.update_elapsed_time()

        if self.strategy.fedbalancer or self.strategy.ss_baseline:
            self.calculate_clients_sample_loss(parameters = self.parameters, client_instructions = client_instructions)
            self.update_elapsed_time(0)

        log(
            DEBUG,
            "elapsed_time: %s | fit_round: strategy sampled %s clients (out of %s)",
            self.elapsed_time,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round        
        results, failures, smartpc_elapsed_time = fit_clients(
            client_instructions,
            max_workers=self.max_workers,
            deadline=self.deadline,
            ddl_baseline_smartpc=self.strategy.ddl_baseline_smartpc,
            clients_per_round=self.strategy.clients_per_round,
            loss_threshold=self.loss_threshold
        )

        if len(results) == len(client_instructions):
            self.update_elapsed_time()
        elif self.strategy.ddl_baseline_smartpc:
            self.update_elapsed_time(smartpc_elapsed_time)
        else:
            self.update_elapsed_time(self.deadline)

        #TODO: Add logic when the round ended before deadline
        # curr_time = datetime.now()
        # self.elapsed_time += timedelta(milliseconds=self.deadline*1000)
        # self.last_checked_time = curr_time

        log(
            DEBUG,
            "elapsed_time: %s | fit_round received %s results and %s failures",
            self.elapsed_time,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(rnd, results, failures)

        if len(results) == 0:
            self.guard_time += 10
        else:
            self.guard_time = 0

            current_round_exploited_utility = 0

            # Update training epoch and batch latency
            for client, fit_res, time in results:
                # if fit_res.train_time_per_epoch >= 2 * np.mean(client.per_epoch_train_times):
                #     client.per_epoch_train_times.append(round(fit_res.train_time_per_epoch/self.strategy.num_epochs,3))
                #     client.per_batch_train_times.append(round(fit_res.train_time_per_batch/self.strategy.num_epochs,3))

                # client.per_epoch_train_times.append(round(fit_res.train_time_per_epoch,3))
                # client.per_batch_train_times.append(round(fit_res.train_time_per_batch,3))
                client.per_epoch_train_times.append(round(np.mean(fit_res.train_time_per_epoch_list),3))
                client.per_batch_train_times.append(round(np.mean(fit_res.train_time_per_batch_list),3))

                client.overthreshold_loss_sum = fit_res.loss_square_sum
                client.overthreshold_loss_count = fit_res.overthreshold_loss_count

                self.clients_explored[client.device_id] = 1

                if self.strategy.fedbalancer and self.filter_if_more_than_ratio_is_explored(1):
                    self._client_manager.epsilon = 0.0
                
                if self.strategy.fedbalancer:
                    current_round_exploited_utility += client.client_utility
                
                log(INFO, "Num examples from client " + str(client.device_id) + ": " + str(fit_res.num_examples))

            # loss_list = [fit_res.loss_sum for client, fit_res, time in results]

            if self.strategy.fedbalancer:
                self.update_fb_variable_percentages(results)
                self.round_exploited_utility.append(current_round_exploited_utility)

        parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = Reconnect(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
        )

    def _get_initial_parameters(self) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        log(INFO, str(random_client))
        parameters_res = random_client.get_parameters()
        log(INFO, "Received initial parameters from one random client")
        return parameters_res.parameters

def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, Reconnect]],
    max_workers: Optional[int],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect

def sample_latency_clients(
    client_instructions: List[Tuple[ClientProxy, SampleLatency]],
    max_workers: Optional[int]
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(sample_latency_client, client_proxy, sl)
            for client_proxy, sl in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None
        )

    server_msg_receive_time = datetime.now()

    # Gather results
    results: List[Tuple[ClientProxy, SampleLatencyRes]] = []
    failures: List[BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures, server_msg_receive_time

def device_info(all_clients):
    # Get initial parameters from one of the clients
    device_ids = []
    log(INFO, "Requesting device info from all clients")
    for client in all_clients:
        device_info_res = client.device_info()
        client.device_id = device_info_res.device_id
        device_ids.append(client.device_id)
        log(INFO, "Device_id "+str(client.device_id))
    log(INFO, "Received device info from all clients")
    return device_ids

def fit_clients(
    # client_instructions: List[Tuple[ClientProxy, FitIns]],
    client_instructions: List[Tuple[ClientProxy, Parameters, dict]],
    max_workers: Optional[int],
    deadline: int,
    ddl_baseline_smartpc: Boolean,
    clients_per_round: int,
    loss_threshold: float,
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        submitted_fs = {
            # executor.submit(fit_client, client_proxy, ins)
            # for client_proxy, ins in client_instructions
            executor.submit(fit_client, client_proxy, parameters, config, loss_threshold)
            for client_proxy, parameters, config in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            # timeout=None,
            timeout=deadline
        )
        # submitted_fs = {
        #     executor.submit(fit_client, client_proxy, ins)
        #     for client_proxy, ins in client_instructions
        # }
        # try:
        #     finished_fs, _ = concurrent.futures.as_completed(
        #         fs=submitted_fs,
        #         # timeout=None,
        #         timeout=10
        #     )
        # except concurrent.futures._base.TimeoutError:
        #     for pid, process in executor._processes.items():
        #         process.terminate()
        #     executor.shutdown()

    # Gather results
    results: List[Tuple[ClientProxy, FitRes, int]] = []
    failures: List[BaseException] = []

    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            # print(result[2].total_seconds())
            results.append(result)
    
    smartpc_elapsed_time = 0.0

    if ddl_baseline_smartpc:
        if len(results) > 0:
            results = sorted(results, key=lambda tup: tup[2].total_seconds())
            accept_until = int(min(clients_per_round, len(results)) * 0.8) #TODO: SHOULD be changed to 0.8 for smartpc
            failures += results[accept_until:]
            results = results[:accept_until]

            smartpc_elapsed_time = results[-1][2].total_seconds()

    return results, failures, smartpc_elapsed_time

#def fit_client(client: ClientProxy, ins: FitIns) -> Tuple[ClientProxy, FitRes]:
def fit_client(client: ClientProxy, parameters: Parameters, config: dict, loss_threshold: float) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    before_fit = datetime.now()
    
    # ins.config["train_time_per_epoch"] = client.per_epoch_train_time
    # ins.config["networking_time"] = client.networking_time
    # ins.sampleloss = client.current_whole_data_loss_list
    # fit_res = client.fit(ins)

    print(client, client.device_id, np.mean(client.per_epoch_train_times[-5:]), np.mean(client.per_batch_train_times[-5:]), np.mean(client.networking_times[-5:]))
    if len(client.per_epoch_train_times) >= 5:
        config["train_time_per_epoch"] = np.mean(client.per_epoch_train_times[-5:])
    else:
        config["train_time_per_epoch"] = np.mean(client.per_epoch_train_times)
    
    if len(client.per_batch_train_times) >= 5:
        config["train_time_per_batch"] = np.mean(client.per_batch_train_times[-5:])
    else:
        config["train_time_per_batch"] = np.mean(client.per_batch_train_times)
    
    if len(client.networking_times) >= 5:
        config["networking_time"] = np.mean(client.networking_times[-5:])
    else:
        config["networking_time"] = np.mean(client.networking_times)

    
    config["inference_time"] = np.mean(client.inference_times)
    config["loss_threshold"] = loss_threshold

    # ins = FitIns(parameters, config, client.current_whole_data_loss_list)

    fit_res = client.fit(FitIns(parameters, config, client.current_whole_data_loss_list))

    after_fit = datetime.now()
    return client, fit_res, after_fit - before_fit

def sample_latency_client(client: ClientProxy, sl: SampleLatency) -> Tuple[ClientProxy, SampleLatencyRes]:
    """Refine parameters on a single client."""
    sample_latency_res = client.sample_latency(sl)
    return client, sample_latency_res


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def evaluate_client(
    client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))

    clients_id_to_device_id = {}
    clients_id_to_device_id[0] = '1'
    clients_id_to_device_id[1] = '3'
    clients_id_to_device_id[2] = '5'
    clients_id_to_device_id[3] = '6'
    clients_id_to_device_id[4] = '7'
    clients_id_to_device_id[5] = '8'
    clients_id_to_device_id[6] = '11'
    clients_id_to_device_id[7] = '14'
    clients_id_to_device_id[8] = '15'
    clients_id_to_device_id[9] = '16'
    clients_id_to_device_id[10] = '17'
    clients_id_to_device_id[11] = '19'
    clients_id_to_device_id[12] = '21'
    clients_id_to_device_id[13] = '22'
    clients_id_to_device_id[14] = '23'
    clients_id_to_device_id[15] = '25'
    clients_id_to_device_id[16] = '26'
    clients_id_to_device_id[17] = '27'
    clients_id_to_device_id[18] = '28'
    clients_id_to_device_id[19] = '29'
    clients_id_to_device_id[20] = '30'

    return clients, groups, data, clients_id_to_device_id