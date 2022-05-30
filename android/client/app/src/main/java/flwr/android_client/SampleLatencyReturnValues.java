package flwr.android_client;

import java.nio.ByteBuffer;

public class SampleLatencyReturnValues {
    private float train_time_per_epoch;
    private float train_time_per_batch;
    private float train_time_per_epoch_list[];
    private float train_time_per_batch_list[];
    private ByteBuffer[] weights;
    private int size_training;

    public SampleLatencyReturnValues(float train_time_per_epoch, float train_time_per_batch, float[] train_time_per_epoch_list, float[] train_time_per_batch_list, ByteBuffer[] weights, int size_training) {
        this.train_time_per_epoch = train_time_per_epoch;
        this.train_time_per_batch = train_time_per_batch;

        this.train_time_per_epoch_list = train_time_per_epoch_list;
        this.train_time_per_batch_list = train_time_per_batch_list;

        this.weights = weights;
        this.size_training = size_training;
    }

    public float[] getTrain_time_per_batch_list() {
        return train_time_per_batch_list;
    }

    public float[] getTrain_time_per_epoch_list() {
        return train_time_per_epoch_list;
    }

    public float getTrain_time_per_batch() {
        return this.train_time_per_batch;
    }

    public float getTrain_time_per_epoch() {
        return this.train_time_per_epoch;
    }

    public int getSize_training() {
        return this.size_training;
    }

    public ByteBuffer[] getWeights() {
        return this.weights;
    }
}
