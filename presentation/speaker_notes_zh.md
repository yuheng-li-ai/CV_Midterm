# CV Assignment 2 汇报手稿（第三版，讲解型）

使用方式：不要逐字机械朗读。每页先讲“这页要说明什么”，再用自己的语速解释。PPT 上已经有结构和数字，手稿的作用是把概念讲清楚。

## Slide 1 Title

各位老师好，我是 Yuheng Li，学号 23307130334。这次汇报我会围绕三个任务展开：宠物分类、宠物分割，以及 VisDrone 检测和视频跟踪。我的重点不是只报分数，而是说明图像进入模型以后到底经历了什么，以及每个扩展实验具体改了哪里。

## Slide 2 What This Presentation Explains

这页相当于汇报路线图。分类任务回答的是：一张宠物照片怎么变成 37 个类别分数。分割任务回答的是：模型怎么给每一个像素分成前景、边界和背景。检测和跟踪任务回答的是：无人机视角下的目标框怎么进一步变成视频里的 ID 和越线计数。后面每个任务我都会先讲数据流，再讲模型结构，最后讲结果和扩展实验。

## Slide 3 Final Results First

先把最终结果放在前面。分类任务最好的模型是 ImageNet 预训练 ResNet-34，测试准确率 90.60%。分割任务最好的 required setting 是 U-Net 加 Dice loss，测试 mIoU 77.90%。检测任务中，YOLO11m 在 960 输入尺寸下达到 0.5356 mAP50 和 0.3310 mAP50-95。后面我会解释这些数字背后的原因，而不是只停留在分数上。

## Slide 4 Harness: Experimental Control

这一页讲实验控制。因为这次作业有很多模型、loss 和扩展实验，如果每次手动改脚本，很容易出现比较不公平的问题。所以我用 YAML config 固定每次实验的模型、学习率、loss、保存路径和输出路径。训练过程中，每个任务都保存验证集最好的 checkpoint。这样最后测试的不是最后一轮模型，而是验证集表现最好的模型，结果更稳定。

## Slide 5 Classification Data Flow

分类任务中，输入是一张 RGB 宠物图片。这里的 horizontal flip 指随机左右翻转图片，比如狗朝左或朝右都应该被识别成同一类；color jitter 指随机调整亮度、对比度和饱和度，让模型不要只适应一种拍摄光照。normalize 是用 ImageNet 的均值和方差把像素分布调到预训练模型熟悉的范围。经过这些处理后，图片进入 ResNet，最后输出 37 个 logits。logits 可以理解为 softmax 之前的原始类别分数，哪个分数最大，模型就预测哪个品种。

## Slide 6 ResNet Baseline Architecture From Code

这页是分类 baseline 的真实结构。输入是 `[B,3,224,224]`，B 代表 batch size。经过第一层大卷积和池化后，空间尺寸从 224 逐步降到 56。之后四个 residual layer 继续提取特征，channel 从 64 增加到 512，空间尺寸最后变成 7 by 7。global average pooling 的作用是把每个 channel 的 7 by 7 空间信息平均成一个数，所以 `[B,512,7,7]` 变成 `[B,512]`。最后新换的 FC 层把 512 维特征变成 37 类分数。

## Slide 7 Classification Extensions: Where They Enter

这里讲分类扩展具体改了哪里。Random init 没改结构，只是取消 ImageNet 预训练，用来证明预训练到底有多重要。ResNet-34 是把 ResNet-18 换成更深的 backbone，空间维度变化一样，但 residual blocks 更多，所以表达能力更强。Label smoothing 是把原来 100% 属于正确类的 hard label 改成稍微平滑的标签，避免模型过度自信。Freeze-unfreeze 是前三轮只训练新分类头，后面再解冻 backbone，这样可以先让新 head 适应任务，再温和调整预训练特征。

## Slide 8 Classification Hyperparameter Grid Search

这一页是为了专门回应超参数分析要求。我没有把 ResNet-34 或 SE 混进 grid search，因为那样变量太多，不是真正干净的超参数搜索。这里固定模型为 ImageNet 预训练 ResNet-18，固定 batch size、图片大小、backbone learning rate 和 weight decay，只扫描两个训练超参数：label smoothing 和新分类头的 learning rate。热力图里颜色越亮表示验证准确率越高。最好的是 label smoothing 0.05 加 head learning rate 1e-3，但优势不大。这说明分类任务对这两个超参数有一定敏感性，但真正决定性能的还是预训练和 backbone 表达能力。

## Slide 9 SE Block: The Attention Extension

SE block 的作用可以理解为“给 channel 打分”。ResNet layer4 输出 `[B,512,7,7]`，这里 512 个 channel 可以看成 512 种视觉特征。SE 先对每个 channel 做 global average pooling，得到 512 个全局统计值；再通过一个小 MLP，先压缩到 32 维，再恢复到 512 维；最后用 sigmoid 得到 0 到 1 的权重。模型把这些权重乘回原 feature map，相当于自动增强重要 channel、压低不重要 channel。这个模块加在 layer4 之后、average pooling 之前。

## Slide 10 Classification Results

分类结果说明预训练是最关键的因素。随机初始化 ResNet-18 只有 40.15%，而同样结构加 ImageNet 预训练后达到 89.40%。这说明模型不是只靠最后一层分类器，而是大量依赖 ImageNet 中学到的边缘、纹理、形状等通用视觉特征。ResNet-34 进一步到 90.60%，说明更深 backbone 对细粒度品种区分有帮助。但 SE 没有明显提升，也说明不是所有复杂模块都会自动有效。

## Slide 11 Classification Failure Cases

从错误样例看，模型的错误比较合理。它不是把完全不相干的类别混在一起，而是在外观很相似的品种之间混淆。比如一些 terrier 和 bulldog 类都有短毛、相似头型和体态；一些猫的面部色块和毛色也很接近。这说明当前瓶颈已经不是“能不能认出宠物”，而是“能不能抓住非常细的品种差异”。

## Slide 12 Segmentation Data Flow

分割任务和分类不同，输入不只是图像，还有每张图对应的 mask。这里的 mask 原始值是 1、2、3，代码里把它转成 0、1、2，分别表示三类像素。对 mask resize 时必须用 nearest interpolation，也就是最近邻插值，因为 mask 是类别标签，不能像图片那样插出中间颜色。图像进入 U-Net 后，不是输出一个类别，而是输出 `[B,3,224,224]`，也就是每个像素都有三个类别分数，最后对每个像素取 argmax 得到分割图。

## Slide 13 U-Net Architecture From Code

这个 U-Net 是从零训练的。左边 encoder 负责逐步压缩空间、提取语义信息：224 到 112、56、28，最后 bottleneck 到 14；同时 channel 从 32 增加到 512。右边 decoder 负责把空间尺寸一步步恢复回 224。关键是 skip connection，也就是把 encoder 中高分辨率的细节特征直接拼接到 decoder 对应层。这样模型既有 bottleneck 的全局语义，也保留了边界和轮廓细节。最后 1x1 convolution 把 32 个 channel 映射成 3 个类别 logits。

## Slide 14 How the Segmentation Losses Are Applied

Cross-Entropy 在这里是逐像素分类：每个像素都有三个 logits，真实 mask 告诉它这个像素属于哪一类。Dice loss 的思路不同，它更像是在问“预测出来的区域和真实区域重叠多少”。代码里先对 logits 做 softmax，得到每个像素属于三类的概率；再把真实 mask 转成 one-hot mask；然后算交集和总面积。Dice 越大说明重叠越好，所以 loss 用 1 减 Dice。CE+Dice 就是把像素级监督和区域重叠监督合起来。

## Slide 15 Segmentation Extensions: Loss-Level Changes

分割的扩展实验没有改 U-Net，只改 loss。Dice smooth 从 1.0 改到 0.1，smooth 是 Dice 分子和分母里的平滑项，主要防止数值不稳定；改小后，loss 对真实重叠变化更敏感。Focal Loss 是另一个思路：先算每个像素的 CE，如果一个像素很容易，模型对它已经很有把握，它的权重就会被压低；如果一个像素很难，权重就会变高。它原本希望模型更关注边界或少数类这样的困难像素。

## Slide 16 Segmentation Results

结果显示 Dice 最好，test mIoU 是 77.90%。mIoU 本质上也是看预测区域和真实区域的交并比，所以 Dice 和评价指标更一致。CE 稳定但不是直接优化区域重叠，所以略低。Dice smooth 0.1 和原始 Dice 很接近，说明这个小改动没有稳定提升。Focal Loss 也没有超过 Dice，这说明当前主要问题不只是简单的类别不平衡。

## Slide 17 Segmentation Qualitative Explanation

可视化里可以看到，模型大部分时候能抓住宠物主体，但边界仍然不够精确。比如耳朵、腿、毛发边缘，以及前景背景颜色很接近的地方，都是容易出错的。这也解释了为什么 Focal Loss 不一定有效：很多 hard pixels 并不是被模型忽略的清晰样本，而是本身就模糊、甚至标注边界也不稳定的像素。强调这些像素不一定会让轮廓更干净。

## Slide 18 VisDrone Label Conversion

检测任务先要把 VisDrone 原始标注转成 YOLO 能读的格式。VisDrone 给的是左上角坐标 x、y 和框的宽高 w、h。YOLO 需要的是中心点坐标 cx、cy 加宽高，而且都要除以图片宽高归一化到 0 到 1。代码还只保留 category 1 到 10 的有效类别，比如 pedestrian、car、bus 等。转换后，每张图片对应一个 txt label 文件，再由 visdrone.yaml 告诉 YOLO 训练集、验证集和类别名在哪里。

## Slide 19 YOLO Detection Training Pipeline

检测训练主要调用 Ultralytics YOLO API，但我用 config 固定模型、数据、输入尺寸和训练参数。YOLOv8n 是 baseline，输入尺寸 640。YOLOv8s 扩展只把模型变大，输入还是 640，所以它更像一个容量对比。YOLO11m@960 则同时换成更新的模型族，并把输入尺寸提高到 960。提高输入尺寸对 VisDrone 很重要，因为小目标在原图里像素很少，放大输入后，小目标在特征图里能留下更多信息。

## Slide 20 YOLOv8n vs YOLOv8s vs YOLO11m

这一页专门比较三个 YOLO 模型。YOLOv8n 和 YOLOv8s 属于同一个结构族，主要模块是 C2f、SPPF 和 Detect。它们的区别主要是宽度系数不同：n 是 0.25，参数量大约 3.16M；s 是 0.5，参数量大约 11.17M。所以 v8n 到 v8s 的提升主要来自容量增加。YOLO11m 则使用 C3k2、SPPF、C2PSA 和 Detect，参数量约 20.11M，结构也更新。因此 YOLO11m@960 的提升不是单一因素，而是更强结构、更大模型和更高分辨率共同作用。

## Slide 21 YOLO11m Architecture

YOLO11m 仍然是单阶段检测器，也就是一张图输入后直接预测框和类别，不需要先生成候选区域。backbone 负责提取不同层级的特征，C3k2 可以理解为更高效的特征提取模块；SPPF 提供更大感受野；C2PSA 引入注意力式的上下文建模。neck 做多尺度融合，把深层语义和浅层细节结合起来。最后 P3、P4、P5 三个检测头分别偏向小、中、大目标。VisDrone 里小目标特别多，所以 P3 和高输入分辨率都很重要。

## Slide 22 Detection Extensions: What Changed

这里再明确三组检测实验的差别。YOLOv8n 是轻量 baseline，640 输入。YOLOv8s 只换模型大小，不改输入尺寸，所以可以相对干净地看容量带来的提升。YOLO11m@960 同时换模型族和输入尺寸，因此不能说提升完全来自某一个因素；更准确的说法是，这个 practical combination 更适合 VisDrone 这种小目标、密集、遮挡多的场景。

## Slide 23 Detection Results

结果上，YOLOv8n 的 mAP50-95 是 0.1665，YOLOv8s 提升到 0.2211，说明容量确实有帮助。YOLO11m@960 达到 0.3310，mAP50 达到 0.5356。mAP50 是 IoU 阈值 0.5 下的平均精度，相对宽松；mAP50-95 会在多个 IoU 阈值下平均，更严格，也更能反映定位质量。这个结果说明更高分辨率和更强模型明显改善了 VisDrone 检测。

## Slide 24 Tracking and Line-Crossing Algorithm

跟踪阶段不是重新训练模型，而是使用检测器每一帧输出的 boxes、类别和置信度。ByteTrack 会把相邻帧中可能属于同一个目标的框关联起来，形成 track ID。越线计数用的是 signed side，也就是判断目标中心点在计数线的哪一侧。如果同一个 ID 从线的一侧变到另一侧，而且之前没被计过数，就认为它完成一次越线。这比单帧检测更难，因为遮挡或漏检会导致 ID 断裂。

## Slide 25 Tracking Demo

这一页我会播放视频。视频里有雨天道路、车辆、行人、伞、反光和遮挡，目标也比较密集，所以它不是一个特别简单的展示场景。播放时重点看两件事：第一，检测框能不能跟住密集小目标；第二，ID 在遮挡和交错运动中是否稳定。如果视频播放失败，我会用这张关键帧说明计数线、目标 ID 和 line count 的含义。

## Slide 26 Overall Discussion

整体来看，三个任务的瓶颈不一样。分类的核心瓶颈是细粒度表示，所以 ImageNet 预训练和更深的 ResNet-34 有效。分割的瓶颈是区域重叠和边界质量，所以 Dice 有效，但 Focal Loss 没有解决模糊边界。检测的瓶颈是小目标可见性和密集定位，所以更强的 YOLO11m 和更高输入分辨率有效。负结果同样重要，因为它们说明不是所有看起来复杂的方法都适合当前问题。

## Slide 27 Conclusion

最后总结，本项目完成了分类、分割、检测和跟踪，并且每个任务都有 controlled extensions。最终结果都来自 validation-best checkpoint，而不是随便取最后一轮。分类达到 90.60%，分割 mIoU 达到 77.90%，检测 mAP50-95 达到 0.3310。我的核心结论是：一个扩展是否有效，不取决于它听起来有多高级，而取决于它有没有击中当前任务真正的瓶颈。

## Slide 28 Appendix

这一页是备用页，不主动讲。这里放了 GitHub、ModelScope、YOLO11 官方文档和视频路径。如果被问到为什么 YOLO11m 没有训满，我会回答：因为时间限制手动停在 epoch 36，但它已经是所有检测实验中验证指标最好的模型，而且明显超过 YOLOv8n 和 YOLOv8s。

## 计时建议

- Slides 1-4：约 55 秒。
- Slides 5-10：分类，约 1 分 30 秒。
- Slides 11-16：分割，约 1 分 45 秒。
- Slides 17-24：检测和跟踪，约 2 分 20 秒。
- Slides 25-26：讨论和总结，约 50 秒。
- 如果超时，压缩 Slide 10 和 Slide 16；不要压缩 Slide 5、12、19、20 这些结构解释页。
