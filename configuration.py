import time

import datasets
class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 5e-5 # 除了backbone之外的学习率

        # Epochs
        self.epochs = 30
        self.pretrain_epochs = 1 # 用caption任务来进行预训练
        self.lr_drop = 1
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # bert
        self.pre_train_bert_path = "/data/transformers/bert-base-uncased"  # huggingface的 bert 的路径

        # Backbone
        # /data/tywang/vision_transformer/facebook_vit-mae-base
        # '/data/tywang/vision_transformer/google_vit-base-patch16-224'
        self.backbone = '/data/tywang/vision_transformer/google_vit-base-patch16-224' # huggingface的 vit 的路径
        self.dual_model = False  # 是否使用VisionTextDualEncoder
        self.use_res = False # 是否使用残差

        
        # Basic
        self.device = 'cuda:0'
        self.seed = 42
        self.batch_size = 32
        self.num_workers = 8
        
        self.checkpoint = ''
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 768
        self.pad_token_id = 0
        self.max_position_embeddings = 64
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dataset = "cc12m"  # choice in ['cc12m','coco']
        if self.dataset=="cc12m":
            self.image_folder = "/data/tywang/dataset/CC12M/small_try"  # 所有的图片都存放在这个文件夹
            self.train_annotation_path = "/data/tywang/dataset/CC12M/CC12M_train.pkl"  # image folder 对应的标注文件, List[{'caption':str,'image':str}]
            self.dev_annotation_path = "/data/tywang/dataset/CC12M/CC12M_dev.pkl"  # image folder 对应的标注文件, List[{'caption':str,'image':str}]
        elif self.dataset=="coco":
            self.dir = '/home/tywang/myURE/text-align-aug/data' # self.dir 下要有从coco数据集下载得到的 annotations/  train2017/  val2017/ 三个文件夹
        self.checkpoint_save_folder = "/data/tywang/vision_translation/catr_ckpt"
        self.limit = -1


        # loss
        self.temp = 0.9  # contrastive loss的参数
    def __str__(self) -> str:
        # 打印所有的参数
        attrs = vars(self)
        TIME=time.strftime("%m-%d-%H:%M:%S", time.localtime())
        return 'Time: '+str(TIME)+'\n'+'\n'.join("config.%s = %s" % item for item in attrs.items())