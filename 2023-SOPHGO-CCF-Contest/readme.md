环境：mlir: v2.2 image + 08.15 release sdk
CCF压缩包中含有sail

# 创建mlir docker容器
docker run --restart always --privileged -v /dev:/dev -td -v <工作目录>:/workspace --name <指定容器名字如 mlir-qx> sophgo/tpuc_dev:v2.2 bash
(最好映射到CCF路径下)

# 进入容器
docker exec -it <上述容器名> bash
 
# 安装运行库，一个容器只需执行一次
# 注意这里不能安装驱动 sophon-driver_<version>_amd64.deb
sudo dpkg -i ./sophon-libsophon_<version>_amd64.deb ./sophon-libsophon-dev_<version>_amd64.deb

# 导入运行环境
source /etc/profile.d/libsophon-bin-path.sh
# 检查容器内运行环境是否正常
bm-smi

# 导入工具链环境
source path/tpu-mlir_vx.y.z-<hash>-<date>/envsetup.sh

# 编译bmodel
cd models
sh convert.sh

#进行推理
退出容器后，重新进入容器

# 导入运行环境
source /etc/profile.d/libsophon-bin-path.sh

#安装sail
pip3 install sophon-0.0.0-py3-none-any.whl


