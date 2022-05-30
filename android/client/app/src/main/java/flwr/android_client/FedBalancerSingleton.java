package flwr.android_client;

import android.util.Pair;

import java.util.ArrayList;
import java.util.List;

public class FedBalancerSingleton {
    private static FedBalancerSingleton fedBalancerSingleton = new FedBalancerSingleton();
    private static List<Pair<Integer, Float>> whole_data_loss_list;
    private static boolean is_first_round;
    private static int samples_count;

    private static boolean ss_baseline;
    private static boolean is_big_client;

    private static List<Pair<Float, Float>> train_epoch_and_batch_latency_history;

    private FedBalancerSingleton() {

        is_first_round = true;
        train_epoch_and_batch_latency_history = new ArrayList<Pair<Float, Float>>();
        ss_baseline = false;
    }

    public static void resetFedBalancerSingleton() {
        fedBalancerSingleton = new FedBalancerSingleton();
    }

    public static FedBalancerSingleton getInstance() {
        return fedBalancerSingleton;
    }

    public static void setWholeDataLossList(List<Float> new_whole_data_loss_list) {
        whole_data_loss_list = new ArrayList<>();
        for(int i = 0; i < new_whole_data_loss_list.size(); i++) {
            whole_data_loss_list.add(Pair.create(i, new_whole_data_loss_list.get(i)));
        }
    }

    public static void setIndexOfWholeDataLossList(float new_loss_value, int index) {
        whole_data_loss_list.set(index, Pair.create(index, new_loss_value));
    }

    public static List<Pair<Integer, Float>> getWholeDataLossList() {
        return whole_data_loss_list;
    }

    public static void setIsFirstRound(boolean new_is_first_round) {
        is_first_round = new_is_first_round;
    }

    public static boolean getIsFirstRound() {
        return is_first_round;
    }

    public static void setSamplesCount(int new_samples_count) {
        samples_count = new_samples_count;
    }

    public static int getSamplesCount() {
        return samples_count;
    }

    public static void addTrainEpochAndBatchLatencyHistory(Pair<Float, Float> new_latency_pair) {
        train_epoch_and_batch_latency_history.add(new_latency_pair);
    }

    public static List<Pair<Float, Float>> getTrainEpochAndBatchLatencyHistory() {
        return train_epoch_and_batch_latency_history;
    }

    public static void setSSBaseline(boolean new_ss_baseline) {
        ss_baseline = new_ss_baseline;
    }

    public static boolean getSSBaseline(){
        return ss_baseline;
    }

    public static void setIsBigClient(boolean new_is_big_client) {
        is_big_client = new_is_big_client;
    }

    public static boolean getIsBigClient(){
        return is_big_client;
    }
}
