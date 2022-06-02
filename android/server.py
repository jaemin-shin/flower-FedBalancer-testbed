from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf

import json
import numpy as np

import sys

from config import Config

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
    # outputs = tf.keras.layers.Dense(units=6, activation="softmax", name="predictions")(x)
    outputs = tf.keras.layers.Dense(units=6, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    x_test = None
    y_test = None

    args = parse_args()
    config_name = args.config

    cfg = Config(config_name)
    strategy = fl.server.strategy.FedAvgAndroid(
        fraction_fit=cfg.fraction_fit,
        fraction_eval=cfg.fraction_eval,
        min_fit_clients=cfg.min_fit_clients,
        min_eval_clients=cfg.min_eval_clients,
        min_available_clients=cfg.min_available_clients,

        eval_fn=get_eval_fn(model),
        sample_loss_fn=get_sample_loss_fn(model),
        on_fit_config_fn=fit_config,
        initial_parameters=None,

        ss_baseline=cfg.ss_baseline,
        fedprox=cfg.fedprox,
        fedbalancer=cfg.fedbalancer,
        ddl_baseline_fixed=cfg.ddl_baseline_fixed,
        ddl_baseline_fixed_value_multiplied_at_mean=cfg.ddl_baseline_fixed_value_multiplied_at_mean,
        ddl_baseline_smartpc=cfg.ddl_baseline_smartpc,
        ddl_baseline_wfa=cfg.ddl_baseline_wfa,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        clients_per_round=cfg.clients_per_round,
        fb_p=cfg.fb_p,
        lss=cfg.lss,
        dss=cfg.dss,
        w=cfg.w,
        total_client_num=cfg.total_client_num, 
    )

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server("[::]:8999", config={"num_rounds": 1000}, strategy=strategy, filename=cfg.output_path)

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
    test_f = open('/mnt/sting/jmshin/FedBalancer/FLASH_jm/data/har_raw/data/test/test_raw_har.json')
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
