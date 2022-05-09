```
#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#                  		  神兽保佑
#                        代码无BUG!
```







## 关键代码

``` python 
# 1 导入库
import cv2
 
# 2 加载视频文件
capture = cv2.VideoCapture("video/video.mp4")
 
# 3 读取视频
ret,frame = capture.read()
while ret:
    # 4 ret 是否读取到了帧，读取到了则为True
    cv2.imshow("video",frame)
    ### save（frame）
    ret,frame = capture.read()
 
    # 5 若键盘按下q则退出播放
    if cv2.waitKey(20) & 0xff == ord('q'):
        break
 
# 4 释放资源
capture.release()
 
# 5 关闭所有窗口
cv2.destroyAllWindows()
```



```python
class PPYOLO(torch.nn.Module):
    def __init__(self, backbone, head):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, im_size, eval=True, gt_box=None, gt_label=None, gt_score=None, targets=None):
        body_feats = self.backbone(x)
        if eval:
            out = self.head.get_prediction(body_feats, im_size)
        else:
            out = self.head.get_loss(body_feats, gt_box, gt_label, gt_score, targets)
        return out

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.backbone.add_param_group(param_groups, base_lr, base_wd)
        self.head.add_param_group(param_groups, base_lr, base_wd)

```



``` python
class Resnet50Vd(torch.nn.Module):
    def __init__(self, norm_type='bn', feature_maps=[3, 4, 5], dcn_v2_stages=[5], downsample_in3x3=True, freeze_at=0, freeze_norm=False, norm_decay=0., lr_mult_list=[1., 1., 1., 1.]):
        super(Resnet50Vd, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert len(lr_mult_list) == 4, "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        self.stage1_conv1_1 = Conv2dUnit(3,  32, 3, stride=2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_1')
        self.stage1_conv1_2 = Conv2dUnit(32, 32, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_2')
        self.stage1_conv1_3 = Conv2dUnit(32, 64, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_3')
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage2
        self.stage2_0 = ConvBlock(64, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], stride=1, downsample_in3x3=downsample_in3x3, is_first=True, block_name='res2a')
        self.stage2_1 = IdentityBlock(256, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], block_name='res2b')
        self.stage2_2 = IdentityBlock(256, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], block_name='res2c')

        # stage3
        use_dcn = 3 in dcn_v2_stages
        self.stage3_0 = ConvBlock(256, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res3a')
        self.stage3_1 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3b')
        self.stage3_2 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3c')
        self.stage3_3 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3d')

        # stage4
        use_dcn = 4 in dcn_v2_stages
        self.stage4_0 = ConvBlock(512, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res4a')
        self.stage4_1 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4b')
        self.stage4_2 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4c')
        self.stage4_3 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4d')
        self.stage4_4 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4e')
        self.stage4_5 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4f')

        # stage5
        use_dcn = 5 in dcn_v2_stages
        self.stage5_0 = ConvBlock(1024, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res5a')
        self.stage5_1 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name='res5b')
        self.stage5_2 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name='res5c')

    def forward(self, input_tensor):
        x = self.stage1_conv1_1(input_tensor)
        x = self.stage1_conv1_2(x)
        x = self.stage1_conv1_3(x)
        x = self.pool(x)

        # stage2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
        # stage3
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
        # stage4
        x = self.stage4_0(s8)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        s16 = self.stage4_5(x)
        # stage5
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)

        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs

    def get_block(self, name):
        layer = getattr(self, name)
        return layer

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.stage1_conv1_1.freeze()
            self.stage1_conv1_2.freeze()
            self.stage1_conv1_3.freeze()
        if freeze_at >= 2:
            self.stage2_0.freeze()
            self.stage2_1.freeze()
            self.stage2_2.freeze()
        if freeze_at >= 3:
            self.stage3_0.freeze()
            self.stage3_1.freeze()
            self.stage3_2.freeze()
            self.stage3_3.freeze()
        if freeze_at >= 4:
            self.stage4_0.freeze()
            self.stage4_1.freeze()
            self.stage4_2.freeze()
            self.stage4_3.freeze()
            self.stage4_4.freeze()
            self.stage4_5.freeze()
        if freeze_at >= 5:
            self.stage5_0.freeze()
            self.stage5_1.freeze()
            self.stage5_2.freeze()

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.stage1_conv1_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage1_conv1_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage1_conv1_3.add_param_group(param_groups, base_lr, base_wd)
        self.stage2_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage2_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage2_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage3_3.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_2.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_3.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_4.add_param_group(param_groups, base_lr, base_wd)
        self.stage4_5.add_param_group(param_groups, base_lr, base_wd)
        self.stage5_0.add_param_group(param_groups, base_lr, base_wd)
        self.stage5_1.add_param_group(param_groups, base_lr, base_wd)
        self.stage5_2.add_param_group(param_groups, base_lr, base_wd)
```

#  预处理相关 
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=True,
            with_cutmix=False,
        )
        # MixupImage
        self.mixupImage = dict(
            alpha=1.5,
            beta=1.5,
        )
        # ColorDistort
        self.colorDistort = dict()
        # RandomExpand
        self.randomExpand = dict(
            fill_value=[123.675, 116.28, 103.53],
        )
        # RandomCrop
        self.randomCrop = dict()
        # RandomFlipImage
        self.randomFlipImage = dict(
            is_normalized=False,
        )
        # NormalizeBox
        self.normalizeBox = dict()
        # PadBox
        self.padBox = dict(
            num_max_boxes=50,
        )
        # BboxXYXY2XYWH
        self.bboxXYXY2XYWH = dict()
        # RandomShape
        self.randomShape = dict(
            sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
            random_inter=True,
        )
        # NormalizeImage
        self.normalizeImage = dict(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True,
            is_channel_first=False,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )
        # Gt2YoloTarget
        self.gt2YoloTarget = dict(
            anchor_masks=[[3, 4, 5], [0, 1, 2]],
            anchors=[[10, 14], [23, 27], [37, 58],
                     [81, 82], [135, 169], [344, 319]],
            downsample_ratios=[32, 16],
            num_classes=self.num_classes,
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=608,
            interp=2,
        )
    
        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decodeImage')
        self.sample_transforms_seq.append('mixupImage')
        self.sample_transforms_seq.append('colorDistort')
        self.sample_transforms_seq.append('randomExpand')
        self.sample_transforms_seq.append('randomCrop')
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('normalizeBox')
        self.sample_transforms_seq.append('padBox')
        self.sample_transforms_seq.append('bboxXYXY2XYWH')
        self.batch_transforms_seq = []
        self.batch_transforms_seq.append('randomShape')
        self.batch_transforms_seq.append('normalizeImage')
        self.batch_transforms_seq.append('permute')
        self.batch_transforms_seq.append('gt2YoloTarget')







### 实现

[(63条消息) PP-YOLO_MarDino的博客-CSDN博客_pp yolo](https://blog.csdn.net/weixin_44106928/article/details/107589319)

[优化器(Optimizer) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/261695487)

[通俗易懂讲解梯度下降法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/335191534)













