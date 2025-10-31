import timm
import torch
import torch.nn as nn


if __name__ == '__main__':
    ### basic
    # print(timm.__version__)  # 1.0.21
    # print(len(timm.list_models()))  # 1279
    # print(len(timm.list_models(pretrained=True)))  # 1689
    # print(len(timm.list_models('*efficientnetv2*', pretrained=True)))  # 21
    # print(timm.list_models('*efficientnetv2*', pretrained=True))

    ### model
    # model = timm.create_model('tf_efficientnetv2_s.in1k',
    #                           pretrained=True,
    #                           cache_dir=r"E:\Git\pytorch-image-models\models")
    # model = timm.create_model('resnet50d', pretrained=True,
    #                           num_classes=10,
    #                           cache_dir=r"E:\Git\pytorch-image-models\models")
    # print(model)
    # print(model.get_classifier())  # Linear(in_features=2048, out_features=10, bias=True)
    # print(model.global_pool)  # SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))

    # pool_types = ['avg', 'max', 'avgmax', 'catavgmax', '']
    #
    # for pool in pool_types:
    #     model = timm.create_model('resnet50d', pretrained=True,
    #                               num_classes=0, global_pool=pool,
    #                               cache_dir=r"E:\Git\pytorch-image-models\models")
    #     model.eval()
    #     feature_output = model(torch.randn(1, 3, 224, 224))
    #     print(feature_output.shape)

    # model = timm.create_model('resnet50d', pretrained=True,
    #                           num_classes=10, global_pool='catavgmax',
    #                           cache_dir=r"E:\Git\pytorch-image-models\models")
    # print(model.get_classifier())
    # num_in_features = model.get_classifier().in_features
    # print(num_in_features)
    # 修改分类头
    # model.fc = nn.Sequential(
    #     nn.BatchNorm1d(num_in_features),
    #     nn.Linear(in_features=num_in_features, out_features=512, bias=False),
    #     nn.ReLU(),
    #     nn.BatchNorm1d(512),
    #     nn.Dropout(0.4),
    #     nn.Linear(in_features=512, out_features=10, bias=False)
    # )
    # print(model.get_classifier())
    # model.eval()
    # output = model(torch.randn(1, 3, 224, 224))
    # print(output.shape)

    ### data
    from timm.data.transforms_factory import create_transform
    # print(create_transform(224,))
    # print(create_transform(224, is_training=True))
    # create_transform(224, is_training=True, auto_augment='rand-m9-mstd0.5')
    # from timm.data.auto_augment import rand_augment_transform
    # tfm = rand_augment_transform(config_str='rand-m9-mstd0.5', hparams={'img_mean': (124, 116, 104)})
    # print(tfm)

    # from PIL import Image
    #
    # img = Image.open(r"E:\Data\TrainSet\13_HS_CaF2_cls\1029_a1b8\images\train\1\3_44.bmp")
    #
    # from timm.data.transforms import RandomResizedCropAndInterpolation
    #
    # tfm = RandomResizedCropAndInterpolation(size=224, interpolation='random')
    #
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    #
    # for i in range(2):
    #     for idx, im in enumerate([tfm(img) for i in range(4)]):
    #         ax[i, idx].imshow(im)
    #
    # fig.tight_layout()
    # plt.show()

