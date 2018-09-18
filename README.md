# Center_loss
mnist classify using center loss
在分类领域原始经典的是softmax损失，但是softmax只优化类间距离，而不优化类内距离，center loss的出发点就是优化类间距离
# mnist classify using only softmax loss
# 20 epochs test Accuracy=0.990
# feature display
![image](https://github.com/lovekittynine/Center_loss/blob/master/images/19.png)
# mnist classify using softmax loss + center loss
# 20 epochs test Accuracy=0.995
# feature display
![image](https://github.com/lovekittynine/Center_loss/blob/master/center_loss_images/19.png)
## 可以看到不加center_loss的时候，类内距离非常大，由于类内距离分散大，导致类间距离也很小。而加入了center_loss后每个类的特征向中心收缩，所以类内距离减小，导致类间距离也相应变大，最后效果也更好。
