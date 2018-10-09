# VGG11-19

Unit.cfg    配置文件说明
-----------------------

[Image data]        图像数据

    image_size          图像大小
    train_ratio         训练集占比
    num_classes         分类数目

[Train data]        训练数据

    num_train           训练集图像数量
    learning_rate       学习率
    step                训练迭代步数
    batch_size          训练批度

[Test data]         测试数据

    num_test            测试集图像数量
    batch_size          测试批度

--------------------------------------

    FlowIO.py   数据流控制器

    主要用于数据封装、读取等操作

--------------------------------------

    train.py    训练

--------------------------------------

    Evaluation.py   精度测试

--------------------------------------

    useModel.py     使用模型预测

--------------------------------------

    VGG_models.py   VGG11~19前趋关系
