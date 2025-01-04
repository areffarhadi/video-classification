# video-classification
Finetuning Facebook's [TimeSformer pre-trained model](https://huggingface.co/facebook/timesformer-base-finetuned-k600) for any task related to video classification, such as emotion recognition, action recognition, etc.


We fine-tuned the model for a two-class face reaction classification, but you can easily modify it for any scenario regarding video classification.

## How to do
-------------------
1. We used the Pytorch version of the MTCNN method to create a new video file that contains faces. To process all your videos, you can use `extract_face.py`. In this code, just change the 'input_folder` and 'output_fodler` parameters.
2. Use `make_manifest.py` to create the required manifests for train and test in .csv format. You can see the CSV files to understand the format of the manifest. In the manifest, the folder name is the label. However, you can modify and utilize `manifest_5Fold.py` to make a 5-fold evaluation.
3. after preprocessing the video and preparing the manifests, run `video_classification_finetune.py` to fine-tune the TimeSformer model.
4. use `video_classification_test.py` to test the saved model using the test.csv manifest.
5. to tune these two scripts, run the `run.sh` file.
   
