＃checkCirclePresicion
双目相机通过标定，标定参数和以左相机为世界坐标系的投影矩阵放置在xml文件中
对勾靶标圆心距真值12mm
程序包括：通过模板匹配提取靶标区域；寻找轮廓精细的椭圆；提取圆心特征点；排序；重建圆心三维坐标；求取圆心距；最大误差0.36，标准差0.15以内
