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
"""Flower ClientManager."""


import random
import threading
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

from flwr.common.logger import log

from .client_proxy import ClientProxy
from .criterion import Criterion

import math
import numpy as np


class ClientManager(ABC):
    """Abstract base class for managing Flower clients."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available clients."""

    @abstractmethod
    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Returns:
            bool: Indicating if registration was successful
        """

    @abstractmethod
    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance."""

    @abstractmethod
    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""

    @abstractmethod
    def wait_for(self, num_clients: int, timeout: int) -> bool:
        """Wait until at least `num_clients` are available."""

    @abstractmethod
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
    
    @abstractmethod
    def fb_oort_sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        current_round: int = 0,
        oort_non_pacer_deadline: float = 0.0
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()

        self.epsilon = 0.9

    def __len__(self) -> int:
        return len(self.clients)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Block until at least `num_clients` are available or until a timeout
        is reached.

        Current timeout default: 1 day.
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def num_available(self) -> int:
        """Return the number of available clients."""
        return len(self)

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Returns:
            bool: Indicating if registration was successful. False if ClientProxy is
                already registered or can not be registered for any reason
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        
        log(INFO, "Alive clients count: " + str(len(available_cids)))

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]
    
    # TODO: Implement this
    def fb_oort_sample(
        self,
        num_clients: int,
        clients_explored: dict,
        min_num_clients: Optional[int] = None,
        current_round: int = 0,
        oort_non_pacer_deadline: float = 0.0,
        num_epochs: int = 5
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)

        log(INFO, "Alive clients count: " + str(len(available_cids)))

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        
        c_id_and_overthreshold_loss = []
        overthreshold_loss_list = []
        
        for cid in available_cids:
            overthreshold_loss_count = self.clients[cid].overthreshold_loss_count
            summ = self.clients[cid].overthreshold_loss_sum

            if overthreshold_loss_count == -1 or overthreshold_loss_count == 0:
                summ = 0
            else:
                summ = math.sqrt(summ / overthreshold_loss_count) * overthreshold_loss_count
        
            c_id_and_overthreshold_loss.append((cid, summ))
            overthreshold_loss_list.append(summ)
            self.clients[cid].client_utility = summ
        
        # Calculate the clip utility of the utility(loss sum) distribution, by 95% value
        overthreshold_loss_list.sort()
        clip_value = overthreshold_loss_list[min(int(len(overthreshold_loss_list)*0.95), len(overthreshold_loss_list)-1)]

        for tmp_idx in range(len(c_id_and_overthreshold_loss)):
            # Add incentive term for clients that have been overlooked for a long time
            cid = c_id_and_overthreshold_loss[tmp_idx][0]
            summ = min(c_id_and_overthreshold_loss[tmp_idx][1], clip_value)
            if self.clients[cid].last_selected_round != -1:
                summ += math.sqrt(0.1*math.log(current_round + 1)/(self.clients[cid].last_selected_round+1)) #To avoid zero division, we regard the training round starts from 1 only at this equation
            if len(self.clients[cid].per_epoch_train_times) >= 5:
                client_complete_time = np.mean(self.clients[cid].networking_times) +  np.mean(self.clients[cid].per_epoch_train_times[-5:]) * num_epochs
            else:
                client_complete_time = np.mean(self.clients[cid].networking_times) +  np.mean(self.clients[cid].per_epoch_train_times) * num_epochs
                            
            if client_complete_time > oort_non_pacer_deadline:
                summ *= math.pow(oort_non_pacer_deadline / client_complete_time, 0.5)
            c_id_and_overthreshold_loss[tmp_idx] = (cid, summ)
        
        sorted_c_id_and_overthreshold_loss = sorted(c_id_and_overthreshold_loss, key=lambda tup: tup[1], reverse=True)

        #Sample clients from 1-epsilon, prioritizing statistical utility
        print("NUM CLIENTS", num_clients, sorted_c_id_and_overthreshold_loss, self.epsilon)
        cutoff_loss = 0.95*(sorted_c_id_and_overthreshold_loss[int(num_clients*(1-self.epsilon))][1])
        if cutoff_loss == 0:
            self.selected_cids = np.random.choice(available_cids, num_clients, replace=False)
            for cid in self.selected_cids:
                self.clients[cid].last_selected_round = current_round
            return [self.clients[cid] for cid in self.selected_cids]
        
        c_id_over_cutoff_loss_ids = []
        c_id_over_cutoff_loss_probs = []
        c_id_over_cutoff_loss_sum = 0

        c_id_less_cutoff_loss_ids = []

        print("CUTOFF LOSS", cutoff_loss)
        
        for item in sorted_c_id_and_overthreshold_loss:
            if item[1] >= cutoff_loss:
                c_id_over_cutoff_loss_ids.append(item[0])
                c_id_over_cutoff_loss_probs.append(item[1])
                c_id_over_cutoff_loss_sum += item[1]
            else:
                c_id_less_cutoff_loss_ids.append(item[0])
        
        for probs_idx in range(len(c_id_over_cutoff_loss_probs)):
            c_id_over_cutoff_loss_probs[probs_idx] /= c_id_over_cutoff_loss_sum

        intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids = []
        intersection_btw_possible_clients_and_c_id_over_cutoff_loss_probs = []
        possible_clients_not_in_c_id_over_cutoff_loss_ids = []

        selected_clients_ids = np.random.choice(c_id_over_cutoff_loss_ids, int(num_clients*(1-self.epsilon)), replace=False, p=c_id_over_cutoff_loss_probs)

        #Randomly sample clients from epsilon - not prioritizing device speed as in Oort
        #From now on, we prioritize device speed as in Oort 
        print("c_id_less_cutoff_loss_ids len:", len(c_id_less_cutoff_loss_ids))
        if self.epsilon > 0.0:
            c_ids_tobe_removed = []
            for cid in c_id_less_cutoff_loss_ids:
                if clients_explored[self.clients[cid].device_id] != 0:
                    c_ids_tobe_removed.append(cid)
            
            for c_id in c_ids_tobe_removed:
                c_id_less_cutoff_loss_ids.remove(c_id)
            
            print("c_id_less_cutoff_loss_ids len:", len(c_id_less_cutoff_loss_ids))

            epsilon_selected_clients_ids = []
            epsilon_selected_clients_len = (num_clients - int(num_clients * (1-self.epsilon)))

            if len(c_id_less_cutoff_loss_ids) < epsilon_selected_clients_len:
                additional_c_id_less_cutoff_loss_ids = np.random.choice(c_ids_tobe_removed, min(len(c_ids_tobe_removed), int(epsilon_selected_clients_len - len(c_id_less_cutoff_loss_ids))), replace=False)         
                c_id_less_cutoff_loss_ids = [*c_id_less_cutoff_loss_ids, *additional_c_id_less_cutoff_loss_ids]
            
            remaining_clients_ids_and_device_speed = []
            
            for c_id in c_id_less_cutoff_loss_ids:
                remaining_clients_ids_and_device_speed.append((c_id, 1/np.mean(self.clients[c_id].per_epoch_train_times)))
            
            random.shuffle(remaining_clients_ids_and_device_speed)
            remaining_clients_ids_and_device_speed.sort(key=lambda x:x[1])
            
            for elem in remaining_clients_ids_and_device_speed[:epsilon_selected_clients_len]:
                epsilon_selected_clients_ids.append(elem[0])
            
            if len(epsilon_selected_clients_ids) < epsilon_selected_clients_len:
                selected_clients_ids = np.random.choice(c_id_over_cutoff_loss_ids, min(int(epsilon_selected_clients_len - len(epsilon_selected_clients_ids)), len(c_id_over_cutoff_loss_ids)), replace=False, p=c_id_over_cutoff_loss_probs)
            
            print("SELECTED CLIENTS IDS: ", len(selected_clients_ids), "EPSILON SELECTED CLIENTS IDS: ", len(epsilon_selected_clients_ids))

            selected_clients_ids = [*selected_clients_ids, *epsilon_selected_clients_ids]

        #Update client_last_selected_round
        
        for c_id in selected_clients_ids:
            self.clients[c_id].last_selected_round = current_round 
        
        if self.epsilon > 0.2:
            log(INFO, 'epsilon {}'.format(self.epsilon))
            self.epsilon = self.epsilon * 0.98

        return [self.clients[cid] for cid in selected_clients_ids]
