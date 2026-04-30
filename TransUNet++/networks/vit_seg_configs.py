
import ml_collections


def get_b16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "seg"
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = "/kaggle/working/project-transunet/model/vit_checkpoint/imagenet21k/imagenet21k_ViT-B_16.npz"
    config.patch_size = 16
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = "softmax"
    return config


def get_testing():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


def get_r50_b16_config():
    config = get_b16_config()
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 6, 3)
    config.resnet.width_factor = 1
    config.classifier = "seg"
    config.pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = "softmax"
    return config


def get_r50_b16_plus_config():
    """TransUNet++ with ResNet50 backbone — nested UNet++ decoder."""
    config = get_r50_b16_config()
    config.use_nested_decoder = True
    return config


def get_convnext_b16_config():
    config = get_b16_config()
    config.patches.grid = (14, 14)
    config.convnext = ml_collections.ConfigDict()
    config.convnext.pretrained = True
    config.classifier = "seg"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [192, 96, 96, 0]
    config.n_classes = 9
    config.n_skip = 3
    config.activation = "softmax"
    return config


def get_convnext_plus_b16_config():
    """TransUNet++ with ConvNeXt backbone — nested UNet++ decoder."""
    config = get_convnext_b16_config()
    config.use_nested_decoder = True
    return config


def get_efficientnet_b3_config():
    config = get_b16_config()
    config.patches.grid = (14, 14)
    config.efficientnet = ml_collections.ConfigDict()
    config.efficientnet.model_type = "b3"
    config.efficientnet.pretrained = True
    config.classifier = "seg"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [48, 32, 24, 0]
    config.n_classes = 9
    config.n_skip = 3
    config.activation = "softmax"
    return config


def get_efficientnet_b3_plus_config():
    """TransUNet++ with EfficientNet-B3 backbone — nested UNet++ decoder."""
    config = get_efficientnet_b3_config()
    config.use_nested_decoder = True
    return config


def get_efficientnet_b4_config():
    config = get_b16_config()
    config.patches.grid = (14, 14)
    config.efficientnet = ml_collections.ConfigDict()
    config.efficientnet.model_type = "b4"
    config.efficientnet.pretrained = True
    config.classifier = "seg"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [56, 32, 24, 0]
    config.n_classes = 9
    config.n_skip = 3
    config.activation = "softmax"
    return config


def get_b32_config():
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = "../model/vit_checkpoint/imagenet21k/ViT-B_32.npz"
    return config


def get_l16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None
    config.classifier = "seg"
    config.resnet_pretrained_path = None
    config.pretrained_path = "../model/vit_checkpoint/imagenet21k/ViT-L_16.npz"
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = "softmax"
    return config


def get_r50_l16_config():
    config = get_l16_config()
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    config.classifier = "seg"
    config.resnet_pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = "softmax"
    return config


def get_l32_config():
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config
