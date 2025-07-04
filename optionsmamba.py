
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')  
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
     
        
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.001, help='initial learning rate') 
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='1', help='GPUs')

    
        parser.add_argument('--arch', type=str, default ='CT_Mamba',  help='archtechture') 
        
        # saving 
        parser.add_argument('--save_dir', type=str, default ='./logs_/',  help='save dir')
        parser.add_argument('--checkpoint', type=int, default=1, help='checkpoint') 

        
        parser.add_argument('--resume', action='store_true',default=False)  
        parser.add_argument('--pretrain_weights',type=str, default='', help='path of pretrained_weights')
        parser.add_argument('--pretrain_weights_SS2D',type=str, default='', help='path of pretrained_weights')
        parser.add_argument('--pretrain_weights_SS2D1',type=str, default='', help='path of pretrained_weights')
        parser.add_argument('--pretrain_weights_uFeatureNet',type=str, default='', help='path of pretrained_weights')
        parser.add_argument('--pretrain_weights_uFeatureNet1',type=str, default='', help='path of pretrained_weights')

        
        parser.add_argument('--patch_size', type=int, default=64) 
        parser.add_argument('--patch_n', type=int, default=4)

        parser.add_argument('--train_dir', type=str, default ='./TrainData',  help='dir of train data') 
        parser.add_argument('--val_dir', type=str, default ='./TestData',  help='dir of train data') 
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup') 
        
        return parser
