from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf

import json
import numpy as np

import sys

from args import parse_args

def main() -> None:
    # Create strategy

    # WHEN MODEL = CNN and INPUT SIZE = 128 * 9
    inputs = tf.keras.Input(shape=(1152,), name="digits")
    x = tf.keras.layers.Reshape((128,9))(inputs)
    x = tf.keras.layers.Conv1D(192, 16, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=6, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    x_test = None
    y_test = None

    strategy = fl.server.strategy.FedAvgAndroid(
        fraction_fit=1.0,
        fraction_eval=0.0,
        min_fit_clients=1,
        min_eval_clients=1,
        min_available_clients=1,
        eval_fn=get_eval_fn(model),
        sample_loss_fn=get_sample_loss_fn(model),
        on_fit_config_fn=fit_config,
        initial_parameters=None,

        latency_sampling=True,
        ss_baseline=False,
        fedprox=False,
        fedbalancer=False,
        fb_client_selection=False,
        ddl_baseline_fixed=False,
        ddl_baseline_fixed_value_multiplied_at_mean=1.0,
        ddl_baseline_smartpc=True,
        ddl_baseline_wfa=False,
        num_epochs=5,
        batch_size=10,
        clients_per_round=5,
        fb_p=0.0,
        lss=0.0,
        dss=0.0,
        w=0,
        total_client_num=21,
    )
    
    args = parse_args()
    client_id = args.client_id

    filename = 'log/client_'+str(client_id)+'_latency.log'

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server("[::]:"+str(8999 - client_id), config={"num_rounds": 1000}, strategy=strategy, filename=filename)

def fit_config(rnd: int, batch_size: int, num_epochs: int, deadline: float, fedprox: bool, fedbalancer: bool, fb_p: float, ss_baseline: bool):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": batch_size,
        "local_epochs": num_epochs,
        "deadline": deadline,
        "fedprox": fedprox,
        "fedbalancer": fedbalancer,
        "fb_p": fb_p,
        "ss_baseline": ss_baseline
    }
    return config

def get_eval_fn(model):
    test_f = open('../data/data/test/test_har.json')
    test = json.load(test_f)
    x_test = np.array(test['user_data']['testuser_1']['x'])
    y_test = np.array(test['user_data']['testuser_1']['y']).reshape(-1,1)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)

        results = model.evaluate(x_test, y_test)
        return results

    return evaluate

def get_sample_loss_fn(model):
    # The `evaluate` function will be called after every round
    def calculate_sample_loss(weights: fl.common.Weights, x_test:np.ndarray, y_test:np.ndarray) -> List[float]:
        model.set_weights(weights)
        y_pred = model.predict(x_test)
        results = tf.keras.losses.sparse_categorical_crossentropy(y_test.flatten(), y_pred)
        return list(results.numpy())

    return calculate_sample_loss


if __name__ == "__main__":
    main()
