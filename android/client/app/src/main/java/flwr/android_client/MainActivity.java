package flwr.android_client;

import android.app.Activity;
import android.content.Intent;
import android.icu.text.SimpleDateFormat;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Handler;
import android.text.TextUtils;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.util.Patterns;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import  flwr.android_client.FlowerServiceGrpc.FlowerServiceBlockingStub;
import  flwr.android_client.FlowerServiceGrpc.FlowerServiceStub;
import com.google.protobuf.ByteString;

import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URL;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import javax.net.ssl.HttpsURLConnection;

public class MainActivity extends AppCompatActivity {
    private String ip;
    private String port;
    private String latency_sampling_port;

    private boolean is_latency_sampling;

    private Button loadDataButton;
    private Button connectButton;
    private Button trainButton;
    private TextView resultText;
    private EditText device_id;
    private ManagedChannel channel;
    public FlowerClient fc;
    private static String TAG = "Flower";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        FedBalancerSingleton.resetFedBalancerSingleton();

        try {
            Intent intent = this.getIntent();
            String experimentID = intent.getStringExtra("id");
            Log.e(TAG, experimentID);

            if (experimentID.equals("0")) {
                is_latency_sampling = false;
            } else {
                is_latency_sampling = true;
            }
        } catch (Exception e) {
            is_latency_sampling = false;
            e.printStackTrace();
        }

        resultText = (TextView) findViewById(R.id.grpc_response_text);
        resultText.setMovementMethod(new ScrollingMovementMethod());
        device_id = (EditText) findViewById(R.id.device_id_edit_text);
//        ip = (EditText) findViewById(R.id.serverIP);
//        port = (EditText) findViewById(R.id.serverPort);
        ip = "143.248.36.214";
        port = "8999";

        loadDataButton = (Button) findViewById(R.id.load_data) ;
        connectButton = (Button) findViewById(R.id.connect);
        trainButton = (Button) findViewById(R.id.trainFederated);

        runOnUiThread(new Runnable(){
            @Override public void run() {
                device_id.setText(Integer.toString(7));
            }
        });

        Log.e(TAG, Build.MODEL);

//        List<Integer> tmp = new ArrayList<>();
//        tmp.add(5);
//        tmp.add(3);
//        tmp.add(2);
//
//        List<Integer> tmp2 = new ArrayList<>();
//        tmp2.addAll(tmp);
//
//        Log.e(TAG, tmp.toString());
//        Log.e(TAG, tmp2.toString());
//
//        Collections.sort(tmp2);
//        Log.e(TAG, tmp.toString());
//        Log.e(TAG, tmp2.toString());

        // device_id.setText('5');
//        if (Build.MODEL.equals("SM-G991N")) {
//            Log.e(TAG, "HAHA");
//        } else if (Build.MODEL.equals("Pixel 5")) {
//            Log.e(TAG, "Pixel!!");
//        }

        fc = new FlowerClient(this);

        loadDataWithoutView();
//        connectWithoutView();
//        runGRCPWithoutView();
    }

    public static void hideKeyboard(Activity activity) {
        InputMethodManager imm = (InputMethodManager) activity.getSystemService(Activity.INPUT_METHOD_SERVICE);
        View view = activity.getCurrentFocus();
        if (view == null) {
            view = new View(activity);
        }
        imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }


    public void setResultText(String text) {
        SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");
        String time = dateFormat.format(new Date());
        runOnUiThread(new Runnable(){
            @Override public void run() {
                resultText.append("\n" + time + "   " + text);
            }
        });
//        resultText.append("\n" + time + "   " + text);
    }

    public void loadData(View view){
        if (TextUtils.isEmpty(device_id.getText().toString())) {
            Toast.makeText(this, "Please enter a client partition ID between 1 and 21 (inclusive)", Toast.LENGTH_LONG).show();
        }
        else if (Integer.parseInt(device_id.getText().toString()) > 21 ||  Integer.parseInt(device_id.getText().toString()) < 1)
        {
            Toast.makeText(this, "Please enter a client partition ID between 1 and 21 (inclusive)", Toast.LENGTH_LONG).show();
        }
        else{
            hideKeyboard(this);
            setResultText("Loading the local training dataset in memory. It will take several seconds.");
            loadDataButton.setEnabled(false);
            final Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    latency_sampling_port = Integer.toString(8999 - Integer.parseInt(device_id.getText().toString()));
                    if ((Integer.parseInt(device_id.getText().toString()) == 13) || (Integer.parseInt(device_id.getText().toString()) == 16) || (Integer.parseInt(device_id.getText().toString()) == 17) || (Integer.parseInt(device_id.getText().toString()) == 21)) {
                        FedBalancerSingleton.getInstance().setIsBigClient(true);
                    }
                    fc.loadData(Integer.parseInt(device_id.getText().toString()));
                    setResultText("Training dataset is loaded in memory.");
                    connectButton.setEnabled(true);
                }
            }, 1000);
        }
    }

    public void loadDataWithoutView(){
        if (TextUtils.isEmpty(device_id.getText().toString())) {
            Toast.makeText(this, "Please enter a client partition ID between 1 and 21 (inclusive)", Toast.LENGTH_LONG).show();
        }
        else if (Integer.parseInt(device_id.getText().toString()) > 21 ||  Integer.parseInt(device_id.getText().toString()) < 1)
        {
            Toast.makeText(this, "Please enter a client partition ID between 1 and 21 (inclusive)", Toast.LENGTH_LONG).show();
        }
        else{
            hideKeyboard(this);
            setResultText("Loading the local training dataset in memory. It will take several seconds.");
            loadDataButton.setEnabled(false);
            final Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    latency_sampling_port = Integer.toString(8999 - Integer.parseInt(device_id.getText().toString()));
                    if ((Integer.parseInt(device_id.getText().toString()) == 13) || (Integer.parseInt(device_id.getText().toString()) == 16) || (Integer.parseInt(device_id.getText().toString()) == 17) || (Integer.parseInt(device_id.getText().toString()) == 21)) {
                        FedBalancerSingleton.getInstance().setIsBigClient(true);
                    }
                    fc.loadData(Integer.parseInt(device_id.getText().toString()));
                    setResultText("Training dataset is loaded in memory.");
                    connectButton.setEnabled(true);
                    connectWithoutView();
                    runGRCPWithoutView();
                }
            }, 1000);
        }
    }

    public void restartActivity(View view)
    {
        // do your work Here
        Intent intent= new Intent(MainActivity.this, MainActivity.class);
        intent.putExtra("id", "0");
        startActivity(intent);
        finish();
    }

    public void restartLatencySamplingActivity(View view)
    {
        // do your work Here
        Intent intent= new Intent(MainActivity.this, MainActivity.class);
        intent.putExtra("id", "1");
        startActivity(intent);
        finish();
    }

    public void connect(View view) {
//        String host = ip.getText().toString();
//        String portStr = port.getText().toString();
        if (is_latency_sampling) {
            port = latency_sampling_port;
        }

        String host = ip;
        String portStr = port;
        if (TextUtils.isEmpty(host) || TextUtils.isEmpty(portStr) || !Patterns.IP_ADDRESS.matcher(host).matches()) {
            Toast.makeText(this, "Please enter the correct IP and port of the FL server", Toast.LENGTH_LONG).show();
        }
        else {
            int port = TextUtils.isEmpty(portStr) ? 0 : Integer.valueOf(portStr);
            channel = ManagedChannelBuilder.forAddress(host, port).maxInboundMessageSize(10 * 1024 * 1024).usePlaintext().build();
            hideKeyboard(this);
            trainButton.setEnabled(true);
            connectButton.setEnabled(false);
            setResultText("Channel object created. Ready to train!");
        }
    }

    public void connectWithoutView() {
        if (is_latency_sampling) {
            port = latency_sampling_port;
        }

        String host = ip;
        String portStr = port;
        if (TextUtils.isEmpty(host) || TextUtils.isEmpty(portStr) || !Patterns.IP_ADDRESS.matcher(host).matches()) {
            Toast.makeText(this, "Please enter the correct IP and port of the FL server", Toast.LENGTH_LONG).show();
        }
        else {
            int port = TextUtils.isEmpty(portStr) ? 0 : Integer.valueOf(portStr);
            channel = ManagedChannelBuilder.forAddress(host, port).maxInboundMessageSize(10 * 1024 * 1024).usePlaintext().build();
            hideKeyboard(this);
            trainButton.setEnabled(true);
            connectButton.setEnabled(false);
            setResultText("Channel object created. Ready to train!");
        }
    }

    public void runGRCP(View view){
        new GrpcTask(new FlowerServiceRunnable(), channel, this).execute();
    }

    public void runGRCPWithoutView(){
        new GrpcTask(new FlowerServiceRunnable(), channel, this).execute();
    }

    private static class GrpcTask extends AsyncTask<Void, Void, String> {
        private final GrpcRunnable grpcRunnable;
        private final ManagedChannel channel;
        private final MainActivity activityReference;

        GrpcTask(GrpcRunnable grpcRunnable, ManagedChannel channel, MainActivity activity) {
            this.grpcRunnable = grpcRunnable;
            this.channel = channel;
            this.activityReference = activity;
        }

        @Override
        protected String doInBackground(Void... nothing) {
            try {
                grpcRunnable.run(FlowerServiceGrpc.newBlockingStub(channel), FlowerServiceGrpc.newStub(channel), this.activityReference);
                return "Connection to the FL server successful \n";
            } catch (Exception e) {
                StringWriter sw = new StringWriter();
                PrintWriter pw = new PrintWriter(sw);
                e.printStackTrace(pw);
                pw.flush();
                return "Failed to connect to the FL server \n" + sw;
            }
        }

        @Override
        protected void onPostExecute(String result) {
            MainActivity activity = activityReference;
            if (activity == null) {
                return;
            }
            activity.setResultText(result);
            activity.trainButton.setEnabled(false);
        }
    }

    private interface GrpcRunnable {
        void run(FlowerServiceBlockingStub blockingStub, FlowerServiceStub asyncStub, MainActivity activity) throws Exception;
    }

    private static class FlowerServiceRunnable implements GrpcRunnable {
        private Throwable failed;
        private StreamObserver<ClientMessage> requestObserver;
        @Override
        public void run(FlowerServiceBlockingStub blockingStub, FlowerServiceStub asyncStub, MainActivity activity)
                throws Exception {
             join(asyncStub, activity);
        }

        private void join(FlowerServiceStub asyncStub, MainActivity activity)
                throws InterruptedException, RuntimeException {

            final CountDownLatch finishLatch = new CountDownLatch(1);
            requestObserver = asyncStub.join(
                            new StreamObserver<ServerMessage>() {
                                @Override
                                public void onNext(ServerMessage msg) {
                                    handleMessage(msg, activity);
                                }

                                @Override
                                public void onError(Throwable t) {
                                    failed = t;
                                    finishLatch.countDown();
                                    Log.e(TAG, t.getMessage());
                                }

                                @Override
                                public void onCompleted() {
                                    finishLatch.countDown();
                                    Log.e(TAG, "Done");
                                }
                            });
        }

        private void handleMessage(ServerMessage message, MainActivity activity) {

            try {
                ByteBuffer[] weights;
                ClientMessage c = null;
                int device_id;

                if (message.hasGetParameters()) {
                    Log.e(TAG, "Handling GetParameters");
                    activity.setResultText("Handling GetParameters message from the server.");

                    weights = activity.fc.getWeights();
                    c = weightsAsProto(weights);
                } else if (message.hasDeviceInfo()) {
                    Log.e(TAG, "Handling DeviceInfo");
                    activity.setResultText("Handling DeviceInfo.");

                    device_id = Integer.parseInt(activity.device_id.getText().toString());
                    c = deviceInfoAsProto(device_id);
                } else if (message.hasFitIns()) {
                    // long handleStartTime = System.currentTimeMillis();

                    Log.e(TAG, "Handling FitIns");
                    activity.setResultText("Handling Fit request from the server.");

                    List<ByteString> layers = message.getFitIns().getParameters().getTensorsList();

                    Scalar epoch_config = message.getFitIns().getConfigMap().getOrDefault("local_epochs", Scalar.newBuilder().setSint64(1).build());
                    int local_epochs = (int) epoch_config.getSint64();

                    Scalar deadline_config = message.getFitIns().getConfigMap().getOrDefault("deadline", Scalar.newBuilder().setDouble(1.0).build());
                    double deadline = deadline_config.getDouble();

                    Scalar fedprox_config = message.getFitIns().getConfigMap().getOrDefault("fedprox", Scalar.newBuilder().setBool(false).build());
                    boolean fedprox = fedprox_config.getBool();

                    Scalar fedbalancer_config = message.getFitIns().getConfigMap().getOrDefault("fedbalancer", Scalar.newBuilder().setBool(false).build());
                    boolean fedbalancer = fedbalancer_config.getBool();

                    Scalar ss_baseline_config = message.getFitIns().getConfigMap().getOrDefault("ss_baseline", Scalar.newBuilder().setBool(false).build());
                    boolean ss_baseline = ss_baseline_config.getBool();

                    Scalar train_time_per_epoch_config = message.getFitIns().getConfigMap().getOrDefault("train_time_per_epoch", Scalar.newBuilder().setDouble(1.0).build());
                    double train_time_per_epoch = train_time_per_epoch_config.getDouble();

                    Scalar train_time_per_batch_config = message.getFitIns().getConfigMap().getOrDefault("train_time_per_batch", Scalar.newBuilder().setDouble(1.0).build());
                    double train_time_per_batch = train_time_per_batch_config.getDouble();

                    Scalar inference_time_config = message.getFitIns().getConfigMap().getOrDefault("inference_time", Scalar.newBuilder().setDouble(1.0).build());
                    double inference_time = inference_time_config.getDouble();

                    Scalar networking_time_config = message.getFitIns().getConfigMap().getOrDefault("networking_time", Scalar.newBuilder().setDouble(1.0).build());
                    double networking_time = networking_time_config.getDouble();

                    Scalar loss_threshold_config = message.getFitIns().getConfigMap().getOrDefault("loss_threshold", Scalar.newBuilder().setDouble(0.0).build());
                    double loss_threshold = loss_threshold_config.getDouble();

                    Scalar fb_p_config = message.getFitIns().getConfigMap().getOrDefault("fb_p", Scalar.newBuilder().setDouble(0.0).build());
                    double fb_p = fb_p_config.getDouble();

                    // Scalar round_idx_config = message.getFitIns().getConfigMap().getOrDefault("round_idx", Scalar.newBuilder().setSint64(1).build());
                    // int round_idx = (int) round_idx_config.getSint64();

                    Log.e(TAG, local_epochs + " " + deadline + " " + fedprox + " " + fedbalancer + " " + train_time_per_epoch + " " + networking_time);

                    List<Float> sampleloss = message.getFitIns().getSamplelossList();

//                    if (FedBalancerSingleton.getInstance().getWholeDataLossList() != null) {
//                        Log.e(TAG, FedBalancerSingleton.getInstance().getWholeDataLossList().toString());
//                    }
                    Log.e(TAG, sampleloss.toString());
                    Log.e(TAG, sampleloss.size()+"");

                    // Our model has 10 layers -> CIFAR-10
                    // Our model has 2 layers -> UCIHAR
                    // Our new model has 4 layers -> UCIHAR_DNN
                    // Our new new model has 6 layers -> UCIHAR_CNN
                    ByteBuffer[] newWeights = new ByteBuffer[6] ;
                    for (int i = 0; i < 6; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }
                    
                    Pair<ByteBuffer[], Integer> outputs = null;

                    List<Integer> sampleIndexToTrain = new ArrayList<Integer>();
                    List<Float> sortedLoss = new ArrayList<Float>();

                    boolean isThisFirstRound = false;

                    if (fedbalancer || (FedBalancerSingleton.getInstance().getIsBigClient() && ss_baseline)) {
                        if (FedBalancerSingleton.getInstance().getIsFirstRound()) {
                            FedBalancerSingleton.getInstance().setWholeDataLossList(sampleloss);
                            FedBalancerSingleton.getInstance().setIsFirstRound(false);
                            activity.fc.sampleInferenceLatency(newWeights);
                            isThisFirstRound = true;
                        }

                        if (fedbalancer) {
                            Pair<List<Integer>, List<Float>> ssresult = activity.fc.fbSampleSelection(loss_threshold, deadline, local_epochs, train_time_per_batch, fb_p);
                            sampleIndexToTrain = ssresult.first;
                            sortedLoss = ssresult.second;
                        } else if (FedBalancerSingleton.getInstance().getIsBigClient() && ss_baseline) {
                            sampleIndexToTrain = activity.fc.baselineSampleSelection();

                        }
                    }

                    //Log.e(TAG, "SAMPLE SELECTION TIME: " + (System.currentTimeMillis() - handleStartTime));

                    SampleLatencyReturnValues slrv;

                    // boolean isTrained = false;

                    if (fedprox || fedbalancer) {
                        int ne = -1;
                        Log.e(TAG, "local_epochs, train_time_per_epoch, networking_time: "+local_epochs+" "+train_time_per_epoch+" "+networking_time);
                        if (isThisFirstRound){
                            for (int e_idx = 1; e_idx < local_epochs + 1; e_idx++) {
                                double e_time = ((sampleIndexToTrain.size()-1) / 10 + 1) * train_time_per_batch * e_idx + networking_time + inference_time;
                                if (e_time < deadline) {
                                    ne = e_idx;
                                }
                            }
                        } else {
                            for (int e_idx = 1; e_idx < local_epochs + 1; e_idx++) {
                                double e_time = ((sampleIndexToTrain.size()-1) / 10 + 1) * train_time_per_batch * e_idx + networking_time;
                                if (e_time < deadline) {
                                    ne = e_idx;
                                }
                            }
                        }
                        Log.e(TAG, "ne: "+ ne);

                        if (ne == -1) {
                            ne = 1;
                        }

                        // outputs = activity.fc.fit(newWeights, ne, sampleIndexToTrain);
                        slrv = activity.fc.fit(newWeights, ne, sampleIndexToTrain);
                    }
                    else {
                        // outputs = activity.fc.fit(newWeights, local_epochs, sampleIndexToTrain);
                        slrv = activity.fc.fit(newWeights, local_epochs, sampleIndexToTrain);
                    }

                    FedBalancerSingleton.getInstance().addTrainEpochAndBatchLatencyHistory(Pair.create(slrv.getTrain_time_per_epoch(), slrv.getTrain_time_per_batch()));

                    //Log.e(TAG, "TRAIN FINISH TIME: " + (System.currentTimeMillis() - handleStartTime));

                    float current_round_loss_min = (float) 0.0;
                    float current_round_loss_max = (float) 0.0;

                    float loss_square_summ = (float) 0.0;
                    int overthreshold_loss_count = 0;
                    float loss_summ = (float) 0.0;
                    int loss_count = 0;

                    if (fedbalancer || (FedBalancerSingleton.getInstance().getIsBigClient() && ss_baseline)) {
                        for (int idx = 0; idx < sampleIndexToTrain.size(); idx++) {
                            FedBalancerSingleton.getInstance().setIndexOfWholeDataLossList(sampleloss.get(sampleIndexToTrain.get(idx)), sampleIndexToTrain.get(idx));
                        }
                    }

                    if (fedbalancer) {
                        current_round_loss_min = (float) sortedLoss.get(0);
                        current_round_loss_max = (float) percentile(sortedLoss, 80);

                        for(int loss_idx = 0; loss_idx < sortedLoss.size(); loss_idx++) {
                            if (sortedLoss.get(loss_idx) > loss_threshold) {
                                loss_square_summ += (float) (sortedLoss.get(loss_idx) * sortedLoss.get(loss_idx));
                                overthreshold_loss_count ++;
                            }
                            loss_summ += (float) sortedLoss.get(loss_idx);
                            loss_count ++;
                        }
                    }

                    weights = activity.fc.getWeights();

                    //Log.e(TAG, "LOSS UPDATE AND SORTED LOSS PROCESING TIME: " + (System.currentTimeMillis() - handleStartTime));

                    c = fitResAsProto(weights, slrv.getSize_training(), current_round_loss_min, current_round_loss_max, loss_square_summ, overthreshold_loss_count, loss_summ, loss_count, slrv.getTrain_time_per_epoch(), slrv.getTrain_time_per_batch(), slrv.getTrain_time_per_epoch_list(), slrv.getTrain_time_per_batch_list());
                } else if (message.hasSampleLatency()) {
                    SimpleDateFormat formatter= new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
                    Date date = new Date(System.currentTimeMillis());
                    String msg_receive_time = formatter.format(date);

                    Log.e(TAG, "Handling SampleLatency");
                    activity.setResultText("Handling Fit request from the server.");

                    List<ByteString> layers = message.getSampleLatency().getParameters().getTensorsList();

                    Scalar epoch_config = message.getSampleLatency().getConfigMap().getOrDefault("local_epochs", Scalar.newBuilder().setSint64(1).build());

                    int local_epochs = (int) epoch_config.getSint64();

                    // Our model has 10 layers -> CIFAR-10
                    // Our model has 2 layers -> UCIHAR
                    // Our new model has 4 layers -> UCIHAR_DNN
                    // Our new new model has 6 layers -> UCIHAR_CNN
                    ByteBuffer[] newWeights = new ByteBuffer[6] ;
                    for (int i = 0; i < 6; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }

                    // Pair<Float, Float> outputs = activity.fc.sampleLatency(newWeights, local_epochs);

                    SampleLatencyReturnValues slrv = activity.fc.sampleLatency(newWeights, local_epochs);

                    float inference_time = activity.fc.sampleInferenceLatency(newWeights);

                    // Log.e(TAG, "Train epoch latency "+ slrv.getTrain_time_per_epoch()+"");
                    // Log.e(TAG, "Inference time "+ inference_time+"");

                    date = new Date(System.currentTimeMillis());
                    String msg_sent_time = formatter.format(date);
                    c = sampleLatencyResAsProto(msg_receive_time, msg_sent_time, slrv.getTrain_time_per_epoch(), slrv.getTrain_time_per_batch(), inference_time, slrv.getWeights(), slrv.getSize_training(), slrv.getTrain_time_per_epoch_list(), slrv.getTrain_time_per_batch_list());
                } else if (message.hasEvaluateIns()) {
                    Log.e(TAG, "Handling EvaluateIns");
                    activity.setResultText("Handling Evaluate request from the server");

                    List<ByteString> layers = message.getEvaluateIns().getParameters().getTensorsList();

                    // Our model has 10 layers
                    ByteBuffer[] newWeights = new ByteBuffer[10] ;
                    for (int i = 0; i < 10; i++) {
                        newWeights[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
                    }
                    Pair<Pair<Float, Float>, Integer> inference = activity.fc.evaluate(newWeights);

                    float loss = inference.first.first;
                    float accuracy = inference.first.second;
                    activity.setResultText("Test Accuracy after this round = " + accuracy);
                    int test_size = inference.second;
                    c = evaluateResAsProto(loss, test_size);
                }
                requestObserver.onNext(c);
                activity.setResultText("Response sent to the server");
                c = null;
            }
            catch (Exception e){
                Log.e(TAG, e.getMessage());
                e.printStackTrace();
            }
        }
    }

    public static Float percentile(List<Float> inputList, double percentile) {
        int index = (int) Math.ceil(percentile / 100.0 * inputList.size());
        return inputList.get(index-1);
    }

    private static ClientMessage weightsAsProto(ByteBuffer[] weights){
        List<ByteString> layers = new ArrayList<ByteString>();
        for (int i=0; i < weights.length; i++) {
            layers.add(ByteString.copyFrom(weights[i]));
        }
        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.ParametersRes res = ClientMessage.ParametersRes.newBuilder().setParameters(p).build();
        return ClientMessage.newBuilder().setParametersRes(res).build();
    }

    private static ClientMessage fitResAsProto(ByteBuffer[] weights, int training_size, float loss_min, float loss_max, float loss_square_summ, int overthreshold_loss_count, float loss_summ, int loss_count, float train_time_per_epoch, float train_time_per_batch, float[] train_time_per_epoch_list, float[] train_time_per_batch_list){
        List<ByteString> layers = new ArrayList<ByteString>();
        for (int i=0; i < weights.length; i++) {
            Log.e(TAG, weights[i]+"");
            layers.add(ByteString.copyFrom(weights[i]));
        }
        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.FitRes.Builder res_builder = ClientMessage.FitRes.newBuilder().setParameters(p).setNumExamples(training_size).setLossMin(loss_min).setLossMax(loss_max).setLossSquareSum(loss_square_summ).setOverthresholdLossCount(overthreshold_loss_count).setLossSum(loss_summ).setLossCount(loss_count).setTrainTimePerEpoch(train_time_per_epoch).setTrainTimePerBatch(train_time_per_batch);

        for (int i=0; i < train_time_per_epoch_list.length; i++){
            res_builder = res_builder.addTrainTimePerEpochList(train_time_per_epoch_list[i]);
        }

        for (int i=0; i < train_time_per_batch_list.length; i++){
            res_builder = res_builder.addTrainTimePerBatchList(train_time_per_batch_list[i]);
        }

        ClientMessage.FitRes res = res_builder.build();
        return ClientMessage.newBuilder().setFitRes(res).build();
    }

    private static ClientMessage sampleLatencyResAsProto(String msg_receive_time, String msg_sent_time, float train_time_per_epoch, float train_time_per_batch, float inference_time, ByteBuffer[] weights, int training_size, float[] train_time_per_epoch_list, float[] train_time_per_batch_list){
        List<ByteString> layers = new ArrayList<ByteString>();
        for (int i=0; i < weights.length; i++) {
            layers.add(ByteString.copyFrom(weights[i]));
        }
        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.SampleLatencyRes.Builder res_builder = ClientMessage.SampleLatencyRes.newBuilder().setMsgReceiveTime(msg_receive_time).setMsgSentTime(msg_sent_time).setTrainTimePerEpoch(train_time_per_epoch).setTrainTimePerBatch(train_time_per_batch).setInferenceTime(inference_time).setParameters(p).setNumExamples(training_size);

        for (int i=0; i < train_time_per_epoch_list.length; i++){
            res_builder = res_builder.addTrainTimePerEpochList(train_time_per_epoch_list[i]);
        }

        for (int i=0; i < train_time_per_batch_list.length; i++){
            res_builder = res_builder.addTrainTimePerBatchList(train_time_per_batch_list[i]);
        }

        ClientMessage.SampleLatencyRes res = res_builder.build();

        return ClientMessage.newBuilder().setSampleLatencyRes(res).build();
    }

    private static ClientMessage deviceInfoAsProto(int device_id){
        ClientMessage.DeviceInfoRes res = ClientMessage.DeviceInfoRes.newBuilder().setDeviceId(device_id).build();
        return ClientMessage.newBuilder().setDeviceInfoRes(res).build();
    }

    private static ClientMessage evaluateResAsProto(float accuracy, int testing_size){
        ClientMessage.EvaluateRes res = ClientMessage.EvaluateRes.newBuilder().setLoss(accuracy).setNumExamples(testing_size).build();
        return ClientMessage.newBuilder().setEvaluateRes(res).build();
    }
}
