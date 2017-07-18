Q1.Difference between demo_cifar n demo_cifar_2?
	由于前者只挑选并保存了恶意样本的bottleneck值，而出于实验的需要还要保存对应正常样本的瓶颈值，
	为了节省时间资源，避免训练from scratch，后者只保存了对应正常样本的瓶颈值，
	仍在统一文件夹下，编号顺延。