package flwr.android_client;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.os.ConditionVariable;
import android.util.Log;
import android.util.Pair;

import androidx.lifecycle.MutableLiveData;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.stream.Stream;

public class FlowerClient {

    private TransferLearningModelWrapper tlModel;
    private static final int LOWER_BYTE_MASK = 0xFF;
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private Context context;
    private final ConditionVariable isTraining = new ConditionVariable();
    private static String TAG = "Flower";
    private int local_epochs = 1;

    public FlowerClient(Context context) {
        this.tlModel = new TransferLearningModelWrapper(context);
        this.context = context;
    }

    public ByteBuffer[] getWeights() {
        return tlModel.getSavedModelParameters();
    }

    public SampleLatencyReturnValues fit(ByteBuffer[] weights, int epochs, List<Integer> sampleIndexToTrain) {

        this.local_epochs = epochs;
        tlModel.updateParameters(weights);
        isTraining.close();
        tlModel.train(this.local_epochs, sampleIndexToTrain);
        tlModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
        isTraining.block();

        SampleLatencyReturnValues slrv = new SampleLatencyReturnValues(tlModel.getMeanEpochTrainTime(), tlModel.getMeanBatchTrainTime(), tlModel.getEpochTrainLatencies(), tlModel.getBatchTrainLatencies(), getWeights(), tlModel.getSize_Training());

        return slrv;
    }

    public List<Integer> pickNRandom(List<Integer> lst, int n) {
        List<Integer> copy = new ArrayList<Integer>(lst);
        Collections.shuffle(copy);
        return n > copy.size() ? copy.subList(0, copy.size()) : copy.subList(0, n);
    }

    public List<Integer> baselineSampleSelection() {
        List<Integer> res = new ArrayList<Integer>();

        int num_data = FedBalancerSingleton.getInstance().getSamplesCount();
        int user_whole_data_len = num_data;

        int batch_size = 10;

        List<Pair<Integer, Float>> whole_data_loss_list =  new ArrayList<>();
        whole_data_loss_list.addAll(FedBalancerSingleton.getInstance().getWholeDataLossList());

        int data_len = whole_data_loss_list.size();

        Collections.sort(whole_data_loss_list, Comparator.comparing(p -> p.second));

        for (int idx = whole_data_loss_list.size() - 1; idx >= whole_data_loss_list.size() - 382; idx--) {
            res.add(whole_data_loss_list.get(idx).first);
        }

        return res;
    }

    public Pair<List<Integer>, List<Float>> fbSampleSelection(double loss_threshold, double deadline, int local_epochs, double train_time_per_batch, double fb_p) {
        List<Integer> res = new ArrayList<Integer>();

        int num_data = FedBalancerSingleton.getInstance().getSamplesCount();
        int user_whole_data_len = num_data;

        int batch_size = 10;

        List<Pair<Integer, Float>> whole_data_loss_list =  new ArrayList<>();
        whole_data_loss_list.addAll(FedBalancerSingleton.getInstance().getWholeDataLossList());

        int data_len = whole_data_loss_list.size();

        Collections.sort(whole_data_loss_list, Comparator.comparing(p -> p.second));

        List<Float> sorted_loss_list = new ArrayList<Float>();
        for (int idx = 0; idx < whole_data_loss_list.size(); idx++) {
            sorted_loss_list.add(whole_data_loss_list.get(idx).second);
        }

        int i;
        for (i = 0; i<whole_data_loss_list.size(); i++) {
            if (whole_data_loss_list.get(i).second >= loss_threshold) {
                break;
            }
        }

        int j = num_data;

        if (deadline > local_epochs * ((num_data - 1)/(batch_size + 1)) * train_time_per_batch) {
            for (int idx = 0; idx < whole_data_loss_list.size(); idx++) {
                res.add(idx);
            }
        } else if (deadline > local_epochs * ((j-i-1)/(batch_size + 1)) * train_time_per_batch) {
            int data_cnt;
            if (((int) ((deadline / (local_epochs * train_time_per_batch)) * batch_size)) > num_data) {
                data_cnt = num_data;
            } else {
                data_cnt = (int) ((deadline / (local_epochs * train_time_per_batch)) * batch_size);
            }
            int easy_data_cnt = (int) (data_cnt * fb_p);
            int hard_data_cnt = data_cnt - easy_data_cnt;

            if (easy_data_cnt > i) {
                easy_data_cnt = i;
                hard_data_cnt = data_cnt - easy_data_cnt;
            } else if (hard_data_cnt > (j - i)) {
                hard_data_cnt = j - i;
                easy_data_cnt = data_cnt - hard_data_cnt;
            }

            List<Integer> easy_data_idxs = new ArrayList<Integer>();
            List<Integer> hard_data_idxs = new ArrayList<Integer>();

            for (int idx = 0; idx < whole_data_loss_list.size(); idx++) {
                if (idx < i) {
                    easy_data_idxs.add(whole_data_loss_list.get(idx).first);
                } else {
                    hard_data_idxs.add(whole_data_loss_list.get(idx).first);
                }
            }

            List<Integer> easy_res = pickNRandom(easy_data_idxs, easy_data_cnt);
            List<Integer> hard_res = pickNRandom(hard_data_idxs, hard_data_cnt);

            res.addAll(easy_res);
            res.addAll(hard_res);
        } else {
            int easy_data_cnt = (int) ((j-i) * fb_p);
            int hard_data_cnt = (j-i) - easy_data_cnt;

            if (easy_data_cnt > i) {
                easy_data_cnt = i;
                hard_data_cnt = (j-i) - easy_data_cnt;
            }

            List<Integer> easy_data_idxs = new ArrayList<Integer>();
            List<Integer> hard_data_idxs = new ArrayList<Integer>();

            for (int idx = 0; idx < whole_data_loss_list.size(); idx++) {
                if (idx < i) {
                    easy_data_idxs.add(whole_data_loss_list.get(idx).first);
                } else {
                    hard_data_idxs.add(whole_data_loss_list.get(idx).first);
                }
            }

            List<Integer> easy_res = pickNRandom(easy_data_idxs, easy_data_cnt);
            List<Integer> hard_res = pickNRandom(hard_data_idxs, hard_data_cnt);

            res.addAll(easy_res);
            res.addAll(hard_res);
        }

         Log.e(TAG, "SORTED LOSS");
         Log.e(TAG, whole_data_loss_list.toString());
         Log.e(TAG, whole_data_loss_list.size()+"");

        return Pair.create(res, sorted_loss_list);
    }

    public SampleLatencyReturnValues sampleLatency(ByteBuffer[] weights, int epochs) {

        this.local_epochs = epochs;
        tlModel.updateParameters(weights);
        isTraining.close();
        tlModel.train(this.local_epochs, (new ArrayList<Integer>()));
        tlModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
        isTraining.block();

        SampleLatencyReturnValues slrv = new SampleLatencyReturnValues(tlModel.getMeanEpochTrainTime(), tlModel.getMeanBatchTrainTime(), tlModel.getEpochTrainLatencies(), tlModel.getBatchTrainLatencies(), getWeights(), tlModel.getSize_Training());

        return slrv;
    }

    public float sampleInferenceLatency(ByteBuffer[] weights) {
        tlModel.updateParameters(weights);
        tlModel.disableTraining();
        return tlModel.calculateSampleInferenceLatency();
    }

    public Pair<Pair<Float, Float>, Integer> evaluate(ByteBuffer[] weights) {
        tlModel.updateParameters(weights);
        tlModel.disableTraining();
        return Pair.create(tlModel.calculateTestStatistics(), tlModel.getSize_Testing());
    }

    public void setLastLoss(int epoch, float newLoss) {
        if (epoch == this.local_epochs - 1) {
            Log.e(TAG, "Training finished after epoch = " + epoch);
            lastLoss.postValue(newLoss);
            tlModel.disableTraining();
            isTraining.open();
        }
    }

    public void loadData(int device_id) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/user" + (device_id - 1) + "_train.txt")));
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i+"th training sample loaded");
                addSample_UCIHAR("data/" + line, true, i - 1);
            }
            reader.close();
            FedBalancerSingleton.getInstance().setSamplesCount(i);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void addSample_UCIHAR(String samplePath, Boolean isTraining, int sampleIndex) throws IOException {
        String sampleClass = get_class_UCIHAR(samplePath);
        float[] sample;
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open(samplePath)));
            String line = reader.readLine();
            double[] sample_in_double = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
            sample = Floats.toArray(Doubles.asList(sample_in_double));

            this.tlModel.addSample_UCIHAR(sample, sampleClass, isTraining, sampleIndex);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String get_class_UCIHAR(String path) {
        String label = path.split("/")[2];
        return label;
    }
}
