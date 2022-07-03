# Setup

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
Place the following 3 folders in a single folder.
```
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
First you need to specify the arugments in `configuration.py` 

you need to specify the following arguments:
```
self.pre_train_bert_path       // the model path(or model name) of the pretrain bert 
self.backbone                  // the model path(or model name) of the pretrain vit
self.dir                       // the coco dataset folder
self.checkpoint_save_folder    // the folder to save the checkpoints
```


To train the model on a single GPU for several epochs run:
```
$ python main.py
```


## Testing (Get text augmentation)
To test CATR with your own images.
```
$ python augment_text.py --checkpoint aFullCheckPointPath  
                         --coco_val_path theFullPathOf[annotations/captions_val2017.json]
```
