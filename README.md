# Object-Detection-on-Mobile-Cam-for-Blind
Multi-Class Object Detection on Mobile Live-Video Stream using Deep Learning Convnets, to assist the Visually Challenged, or to signal an incoming threat.

![](Output_Highlights.gif)

This is a reincarnation of [this](https://github.com/kumarkan/Food_Detection) project to showcase Object Detection of custom shapes trained offline and deployed on Android mobile. You can find the background theme of the project in [this](https://becominghuman.ai/third-eye-in-your-hand-850b77e1d45a) blog.

To detect an object of your choice, we need to follow these steps:

- **Data Generation:** Gather images of similar objects.
- **Image Annotation:** Label the objects with bounding box.
- **API Installation:** Install TensorFlow Object Detection API.
- **Train & Validate Model:** Using annotated images.
- **Freeze the Model:** To enable mobile deployment.
- **Deploy and Run:** In mobile or virtual environment.

## Data Generation
Lets assume, the object you want to detect is a flying fighter jet. Find out and save images of flight from https://images.google.com/ and download some videos of jets flying the skies, to make the offline video input.
Use Lossless Cut to cut out relevant portions from video and MP4Joiner to join video-bits without loss. The video thus created becomes the test data. Now extract some frames from created video using  Video to JPG Converter. The extracted frames, along with saved images from google, are batch processed by IrfanView to make the filenames consistent and  image dimensions similar, which  becomes the train data.

## Image Annotation

By now, we have the train and test images. But the exact location and type of objects in the images has to be explicitly labelled. The bounding boxes can be drawn using Label Box or Label Image software, the output of which are saved as XML files, corresponding to each image.

For multi-class classification, give different label names for different objects in the image. This information is saved in the generated XML files. Add all the categories to label_map.pbtxt in \data folder and modify NUM_CLASSES variable in code, accordingly.

Label Image Software being used to annotate objects in the imageFor the purpose of this blog, I have downloaded around 125 random images of chair and took 75 images of chair using my mobile cam. Around 100 images of fighter jets are also download from multiple sources. The whole data-set of 300 images are manually annotated to specify object location and type.

Now we need to convert the generated XML files to a format suitable for training. Download the project from here and use FoodDetection.ipynb to convert the generated XML files to CSV. Generate TFRecord files using code adapted from this raccoon detector to optimize the data feed. The train &test data are separately handled in the code. Modify the train folder name in the TFRecord generator .py file, if you wish to train other data-sets.

TFRecord is TensorFlows binary storage format. It reduces the training time of your model, as binary data takes up less space and disk read more efficient.

```
ipython notebook FoodDetection.ipynb
python generate_tfrecord.py
mv test.record data
mv train.record data
```

## API Installation
We will use MobileNet model for the neural network architecture and Single Shot Detection to locate the bounding boxes. Mobilenet-SSD architecture is designed to use in mobile applications. 

To install TensorFlow Object Detection API, download and unzip TensorFlow Models from the repository here and execute the commands below.

```
cd models/research/
pip install protobuf-compiler
protoc object_detection/protos/*.proto - python_out=.
set PYTHONPATH=<cwd>\models\research;<cwd>\models\research\slim
cd ../../
```
  
## Train & Validate Model
Download the pre-trained Mobileset SSD model from [here](https://medium.com/r/?url=http%3A%2F%2Fdownload.tensorflow.org%2Fmodels%2Fobject_detection%2Fssd_mobilenet_v1_coco_11_06_2017.tar.gz) and retrain it with your dataset to replace the classes as you desire. Re-training is done to reduce the training time.

Once the environment variable is set, execute the train.py file with a config parameter. Let it train till MAX_STEPS=20,000 or until loss is stabilized.

```
python train.py -logtostderr -train_dir=data\ -pipeline_config_path=data\ssd_mobilenet_v1_custom.config
```

## Freeze the Model
To serve a model in production, we only need the graph and its weights. We don't need the metadata saved in .meta file, which is mostly useful to retrain the model. TF has a built-in helper function to extract what is needed for inference and create frozen graph_def.

To export the graph for inference, use the latest checkpoint file number stored inside "data" folder. The frozen model file, frozen_inference_graph.pb is generated inside output directory, to be deployed in mobile.
```
rm -rf object_detection_graph
python models/research/object_detection/export_inference_graph.py -input_type image_tensor -pipeline_config_path data/ssd_mobilenet_v1_custom.config -trained_checkpoint_prefix data/model.ckpt-19945 -output_directory object_detection_graph
```

## Deploy and Run
Download TensorFlow Android examples from [here](https://github.com/tensorflow/tensorflow). Using Android Studio, open the project in [this](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android) path and follow the steps below.
Update the tensorflow/WORKSPACE file in root directory with the API level and location of the SDK and NDK.
```
android_sdk_repository (
name = "androidsdk",
api_level = 23,
build_tools_version = "28.0.3",
path = "C:\Users\Anand\AppData\Local\Android\Sdk",
)
 
android_ndk_repository(
name = "androidndk",
path = "C:\Users\Anand\AppData\Local\Android\Sdk\ndk-bundle",
api_level = 19,
)
```

- Set "def nativeBuildSystem" in build.gradle to 'none' using Android Studio
- Download quantized Mobilenet-SSD TF Lite model from here and unzip mobilenet_ssd.tflite to assets folder: tensorflow/examples/android/assets/
- Copy frozen_inference_graph.pb generated in the previous step and label_map.pbtxt in \data folder to the "assets" folder above.  Edit the label file to reflect the classes to be identified.
- Update the variables TF_OD_API_MODEL_FILE and TF_OD_API_LABELS_FILE in DetectorActivity.java to above filenames with prefix "file:///android_asset/"
- Build the bundle as an APK file using Android Studio.
- Locate APK file and install on your Android mobile. Execute TF-Detect app to start object detection. The camera would turn on and detect objects real-time.
