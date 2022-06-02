package flwr.android_client;

import android.content.Context;
import android.os.ConditionVariable;
import android.os.Environment;
import android.util.Pair;

import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.GatheringByteChannel;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import org.tensorflow.lite.examples.transfer.api.AssetModelLoader;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.LossConsumer;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;

/**
 * App-layer wrapper for {@link TransferLearningModel}.
 *
 * <p>This wrapper allows to run training continuously, using start/stop API, in contrast to
 * run-once API of {@link TransferLearningModel}.
 */
public class TransferLearningModelWrapper implements Closeable {

    private final TransferLearningModel model;

    private final ConditionVariable shouldTrain = new ConditionVariable();
    private volatile LossConsumer lossConsumer;
    private Context context;
    TransferLearningModelWrapper(Context context) {
        model =
                new TransferLearningModel(
                        new AssetModelLoader(context, "model"),
                        Arrays.asList("0", "1", "2", "3", "4", "5"));
        this.context = context;
    }


    public void train(int epochs, List<Integer> sampleIndexToTrain){
        new Thread(() -> {
                shouldTrain.block();
                try {
                    model.train(epochs, lossConsumer, sampleIndexToTrain).get();
                } catch (ExecutionException e) {
                    throw new RuntimeException("Exception occurred during model training", e.getCause());
                } catch (InterruptedException e) {
                    // no-op
                }
        }).start();
    }

    // This method is thread-safe.
    public Future<Void> addSample(float[] image, String className, Boolean isTraining, int sampleIndex) {
        return model.addSample(image, className, isTraining, sampleIndex);
    }

    // This method is thread-safe.
    public Future<Void> addSample_UCIHAR(float[] sample, String className, Boolean isTraining, int sampleIndex) {
        return model.addSample_UCIHAR(sample, className, isTraining, sampleIndex);
    }

    public float calculateSampleInferenceLatency(){
        return model.getSampleInferenceLatency();
    }

    public Pair<Float, Float> calculateTestStatistics(){
        return model.getTestStatistics();
    }

    // This method is thread-safe, but blocking.
    public Prediction[] predict(float[] image) {
        return model.predict(image);
    }

    public int getTrainBatchSize() {
        return model.getTrainBatchSize();
    }

    /**
     * Start training the model continuously until {@link #disableTraining() disableTraining} is
     * called.
     *
     * @param lossConsumer callback that the loss values will be passed to.
     */
    public void enableTraining(LossConsumer lossConsumer) {
        this.lossConsumer = lossConsumer;
        shouldTrain.open();
    }

    public FileChannel createChannelInstance(File file, boolean isOutput)
    {
        FileChannel fc = null;
        try
        {
            if (isOutput) {
                fc = new FileOutputStream(file).getChannel();
            } else {
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return fc;
    }

    public float getMeanEpochTrainTime() {
        return model.getMeanEpochTrainLatency();
    }

    public float getMeanBatchTrainTime() { return model.getMeanBatchTrainLatency(); }

    public float[] getEpochTrainLatencies() { return model.getEpochTrainLatencies(); }

    public float[] getBatchTrainLatencies() { return model.getBatchTrainLatencies(); }

    public int getSize_Training() {
        return model.getSize_Training();
    }

    public int getSize_Testing() { return model.getSize_Testing(); }

    public ByteBuffer[] getParameters()  {
        return model.getParameters();
    }

    public ByteBuffer[] getSavedModelParameters() { return model.getSavedModelParameters(); }

    public void updateParameters(ByteBuffer[] newParams) {
        model.updateParameters(newParams);
    }

    /**
     * Stops training the model.
     */
    public void disableTraining() {
        shouldTrain.close();
    }

    /** Frees all model resources and shuts down all background threads. */
    public void close() {
        model.close();
    }
}
