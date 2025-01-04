# video-classification
Finetuning Facebook's [TimeSformer pre-trained model](https://huggingface.co/facebook/timesformer-base-finetuned-k600) for any task related to video classification, such as emotion recognition, action recognition, etc.


We fine-tuned the model for a two-class face reaction classification, but you can easily modify it for any scenario regarding video classification.

## How to do
-------------------
1. we used pytorch version of MTCNN method to make a new video file that just contains face. to process all your videos you can use ``.

To run the code:
