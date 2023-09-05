http://681314.com/A/bzkBdpOrR8

### 01 /cfg/baseline
其中depth_multiple、 width_multiple是特征图的缩放比例，达到s、m、x等模型的选择。
### 02 /data/hyp.scratch.tiny.yaml
还有学习率等超参数。
关于数据增强的参数，一般来说：模型越小泛化能力越若，增强程度应越低。数据越少，增强程度应越大，防止过拟合。
### 03
UserWarning: torch.meshgrid: in an upcoming release, it will be required to....
https://blog.csdn.net/m0_74890428/article/details/131042830