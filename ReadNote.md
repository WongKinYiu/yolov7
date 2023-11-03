http://681314.com/A/bzkBdpOrR8

### 01 /cfg/baseline
其中depth_multiple、 width_multiple是特征图的缩放比例，达到s、m、x等模型的选择。
### 02 /data/hyp.scratch.tiny.yaml
还有学习率等超参数。
关于数据增强的参数，一般来说：模型越小泛化能力越若，增强程度应越低。数据越少，增强程度应越大，防止过拟合。
### 03
UserWarning: torch.meshgrid: in an upcoming release, it will be required to....
https://blog.csdn.net/m0_74890428/article/details/131042830

### yolov7-tiny网络结构讲的非常清晰 
https://blog.csdn.net/qq_41398619/article/details/129742953?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169511107416800197071264%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=169511107416800197071264&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-129742953-null-null.nonecase&utm_term=yolov7&spm=1018.2226.3001.4450
freeze 50
### 关于tensorrt部署的c++
https://zhuanlan.zhihu.com/p/580268047?utm_id=0