# 一、Termius下载和汉化

## 1.1 Termius下载

 Termius是一款现代SSH客户端工具，支持桌面和移动端使用。

来到官网[Termius – Modern SSH Client for Desktop and Mobile](https://termius.com/)

这里下载的是免费版的window版本

![image-20260915085941340](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915085941340.png)

默认直接下载到C盘里，在官网上需要登录

## 1.2 汉化

由于这个软件内部没有语言设置（我没找到），上网查找后，找到了汉化脚本

链接：[Release v9.21.2 · ArcSurge/Termius-Pro-zh_CN · GitHub](https://github.com/ArcSurge/Termius-Pro-zh_CN/releases/tag/v9.21.2)

在下方选择对应版本，**注意软件会自动更新，更新后要替换新的汉化**

![image-20260915090020230](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915090020230.png)

我选择的是`app-windows-localize.asar` ，下载后重命名为`app.asar`

下载完后来到==C:\Users\你的用户名\AppData\Local\Programs\Termius\resources==目录下

把你下载完的内容 更换文件下的`app.asar`即可

成功显示中文：

![image-20260915090039257](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915090039257.png)



# 二、虚拟机VMware安装





# 三、Docker的安装

[Docker-02.Docker的安装_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1S142197x7?spm_id_from=333.788.videopod.episodes&vd_source=24c1e92bdfe1c6a0f1b228cda0583ac9&p=23)

 官方文档：https://docs.docker.com/engine/install/centos/

黑马飞书文档：[⁠‌⁠‬‌‬﻿⁠‍‬‬‌‌‬‬‬﻿‍‌‌‌‌﻿安装Docker - 飞书云文档](https://b11et3un53m.feishu.cn/wiki/Rfocw7ctXij2RBkShcucLZbrn2d?fromScene=spaceOverview)

虚拟机安装的`VMware`，过程：

系统为CentOS 7 64位

ssh工具为`Termius`，过程：



## 3.1 安装docker

### 3.1.1 卸载旧版本

在ssh终端输入如下命令：

```bash
sudo dnf remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
```

如果没有旧版本会显示如下：

![image-20260915090332290](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915090332290.png)



### 3.1.2 配置docker的yum库

先要安装yum工具，代码如下：

```bash
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
```

- `yum-utils` 提供 `yum-config-manager`，可用于管理软件仓库。
- `device-mapper-persistent-data` 和 `lvm2` 是 Docker 存储驱动（devicemapper）的依赖项。

### 3.1.2.1 安装出现问题（可跳）

报错，问题说是无法连接到服务器

![image-20260915090618977](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915090618977.png)



看飞书评论区的方法

1、使用阿里云更新yum源，代码如下：

```bash
curl -o /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)2、清除yum：

```bash
yum clean all
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)3、更新缓存

```bash
yum makecache
```

过程如图：

![image-20260915090701829](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915090701829.png)

然后我们再次执行最初的下载命令：

```bash
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)成功界面如图，最后会显示 “完毕！”：

![image-20260915090741196](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915090741196.png)

### 3.1.3 配置源

配置docker的yum源，执行命令：

```bash
sudo yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

sudo sed -i 's+download.docker.com+mirrors.aliyun.com/docker-ce+' /etc/yum.repos.d/docker-ce.repo
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)更新yum缓存：

```bash
sudo yum makecache fast
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)运行过程：

![image-20260915090815793](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915090815793.png)



### 3.1.4 安装docker

执行命令：

```bash
yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)结果如图：

![image-20260915091023782](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091023782.png)

### 3.1.5 启动和校验

启动和校验代码如下：

```bash
#查看docker版本
docker -v

# 启动Docker
systemctl start docker

# 停止Docker
systemctl stop docker

# 重启
systemctl restart docker

# 设置开机自启
systemctl enable docker

# 执行docker ps命令，如果不报错，说明安装启动成功
docker ps
```

启动后，输入docker images不会报错

![image-20260915091317921](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091317921.png)



## 3.2 配置镜像加速

来到阿里云网站：[阿里云-计算，为了无法计算的价值](https://www.aliyun.com/)

找到阿里云的 **容器镜像服务ACR** ：

![image-20260915091352003](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091352003.png)

点击管理控制台

![image-20260915091410006](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091410006.png)

点击 **镜像加速器**，复制加速器地址：

![image-20260915091421995](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091421995.png)

下面有具体使用方法。

回到ssh终端，依次输入如下代码：

```bash
# 创建目录
mkdir -p /etc/docker

# 复制内容，注意把其中的镜像加速地址改成你自己的
tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://你的镜像地址.mirror.aliyuncs.com"]
}
EOF

# 重新加载配置
systemctl daemon-reload

# 重启Docker
systemctl restart docker
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)可以用如下代码检验是否成功替换镜像：

```bash
cat /etc/docker/daemon.json
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)显示我们刚刚替换的地址：

![image-20260915091456717](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091456717.png)

重新启动我们的docker（我是关了再开）：

![image-20260915091524544](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091524544.png)











# 四、Anaconda安装

Anaconda官网https://www.anaconda.com/download

推荐在国内镜像网站下载：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

选择对应操作系统，这里选择2025.10-1-Windows-x86_64.exe

![image-20260915091634499](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091634499.png)

下载完后选择Just Me

![image-20260915091651013](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091651013.png)

选择合适的安装路径

![image-20260915091703533](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091703533.png)

勾选1，3，4

![image-20260915091726418](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915091726418.png)

安装完成之后，验证是否成功，打开cmd命令行，输入以下代码

```bash
conda --version
```

可以输出 Conda 版本号，说明安装成功。



创建并激活新环境（推荐，避免污染 base）:

```bash
conda create -n torch_env python=3.9
```

输入 `y` 确认后，再执行：

```bash
conda activate torch_env
```



# 五、PyTorch安装

打开Anaconda Prompt

输入代码

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 安装完成后验证

```bash
python -c "import torch; print(torch.__version__)"
```

如果输出版本号，说明安装成功。



# 六、CCSwitch下载







# 七、ClaudeCode安装

## 7.1 ClaudeCode安装



## 7.2 接入deepseek 和 中转站



## 7.3 cc-Haha下载

cc-Haha是ClaudeCode的一个可视化软件。

来到cc-Haha的github地址：https://github.com/NanmiCoder/cc-haha/releases

选择对应版本，我这里选择windows版本，分x64和arm64，根据你的CPU来选择

![image-20260902110440308](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260902110440308.png)

下载完之后直接安装即可。

打开应用在设置里面添加服务商即可使用。

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260902121144241.png" alt="image-20260902121144241" style="zoom:67%;" />





# 八、Codex安装

## 8.1 codex下载

### 8.1.1 桌面端下载

来到官网：https://openai.com/zh-Hans-CN/codex/

点击下载

![](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915170010460.png)



### 8.1.2 命令行下载

打开命令行，输入指令：

```bash
npm install -g @openai/codex
```

不报错就是安装成功，用校园网安装

查看版本，验证一下是否安装成功：

```bash
codex --version
```

![image-20260915222427178](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915222427178.png)

启动codex：

```bash
codex
```



## 8.2 接入API

进入CCSwitch，选择codex，默认选择的是claude

![image-20260915223357785](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915223357785.png)



点击右上角的添加，找到deepseek

![image-20260915223832741](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915223832741.png)

去deepseek官网：https://platform.deepseek.com/api_keys

创建一个api-key

![image-20260915224020655](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915224020655.png)

回到CCSwitch，在下面把api复制进去就行

![image-20260915224557975](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915224557975.png)

菜单显示名是你在codex里面可以进行切换

可以点击右上角 `获取模型列表`，实际请求模型填写对应的就行，填完保存

![image-20260915224915966](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915224915966.png)



新建工作目录，我的是`E:\CodexWork`

正常显示，可以使用

![image-20260915230524605](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260915230524605.png)



# 九、Obsidian安装

## 9.1 下载与安装

官网：https://obsidian.md/

打开官网直接安装

![image-20260916074953266](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260916074953266.png)

双击安装包进行安装，选择合适的下载位置：

![image-20260916082257707](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260916082257707.png)

下载好后打开

## 9.2 简单配置

点击创建仓库

![image-20260916082734670](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260916082734670.png)

选择合适的位置

![image-20260916082833851](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260916082833851.png)



进去之后选择设置

![image-20260917161638943](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917161638943.png)







# 十、codex + obsidian + zetero实现论文辅助

## 10.1 前置安装

codex的安装参考**八、**

obsidian的安装参考**九、**

zetero的安装参考B站：https://www.bilibili.com/video/BV1RkyBBVE3f/?spm_id_from=333.337.search-card.all.click&vd_source=24c1e92bdfe1c6a0f1b228cda0583ac9



`MinerU`：是一个开源的 PDF 解析工具，可以本地部署

打开命令行，执行以下命令创建一个沙箱：

```bash
conda create -n MinerU python=3.11
conda activate MinerU
```

然后执行安装命令：

```bash
pip install -U pip uv
uv pip install -U "mineru[all]"
```

![image-20260918122350044](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260918122350044.png)

检查是否安装成功：

```bash
pip show mineru
mineru --version
```



![image-20260918122844313](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260918122844313.png)

要让 Codex 调用这个本地 MinerU，需要通过 **MCP (Model Context Protocol)** 进行集成。以下是完整步骤：

在MinerU 沙箱中：

```bash
pip install mineru-mcp-server
```



![image-20260918123427498](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260918123427498.png)



## 10.2 skill下载

我们要用到的两个skill都来自开发者 `cheneternity` 的仓库

这两个技能都**强烈依赖一个特定的本地目录结构**，默认路径为 `D:\ResearchVault`

如果在下载前修改路径会比较麻烦，所以我们先在D盘创建相对应的路径

```bash
mkdir D:\ResearchVault
mkdir D:\ResearchVault\note
mkdir D:\ResearchVault\模板
```



![image-20260917220106898](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917220106898.png)



打开`命令行`，执行：

```bash
npx skills add https://github.com/cheneternity/Zotero-Analytical-Workflow-Skills --all -g -a codex
```

这个命令安装的skill会下载到`C:\Users\<你的用户名>\.codex\skills\`，可以全局执行

安装如下：

![image-20260917220817416](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917220817416.png)

skill安装成功，失败的是 **Eve** 和 **PromptScript** 这两个 agent，原因是它们**不支持全局技能安装**。这和你使用的 Codex 完全无关，不影响 Codex 使用这些技能。



## 10.3 工具配置

### 10.3.1 Obsidian

然后来到`Obsidian`，选择我们最开始创建的 `D:\ResearchVault`

![image-20260917221112446](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917221112446.png)

![image-20260917221217519](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917221217519.png)



### 10.3.2 Zetero

来到Zetero，打开设置，其实这个目录不影响，但是默认存C盘有些占空间

![image-20260917221822746](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917221822746.png)

选择你要换的文件夹，换完之后会要求你重启

![image-20260917221911636](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917221911636.png)



### 10.3.3 Codex

打开codex，点击左上角，切换成Codex（因为GPT和Codex公用一个桌面端），点击项目创建

![image-20260917222832512](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917222832512.png)

选择我们之前创建的 `D:\ResearchVault`

![image-20260917223102623](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917223102623.png)





## 10.4 skill使用

先将论文导入到Zetero中，可以创建一个分类，导入一个期刊文章，这里做测试

![image-20260917223631879](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917223631879.png)

然后导入我们的pdf文件

![image-20260917223757993](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917223757993.png)

打开翻译软件，可以先对纯英文的论文进行翻译，得到中文版本

![image-20260917223955974](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917223955974.png)

![image-20260917225929883](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917225929883.png)



得到中文版本，和双语对照版本的pdf

![image-20260917230342960](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917230342960.png)



然后在Codex中输出。效果如下：

![image-20260918104701729](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260918104701729.png)



由于之前忘记装插件了，装好插件后，再次用codex整理

![image-20260918130603541](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260918130603541.png)



目录说明：

- **全文归档**：`D:\ResearchVault\03fulltext\智能语音\AQIFVCXT.md`
- **精读笔记**：`D:\ResearchVault\note\智能语音\模型压缩和知识蒸馏的语音模型.md`





## 10.5 插件版本过低解决

Zetero上来给我自动更新了....打开Zetero说插件版本太低

![image-20260917224627513](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917224627513.png)

来到插件商店：https://zotero-chinese.com/plugins/#search=Zotero+Pdf2zh

下载最新的

![image-20260917224744903](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917224744903.png)

来到你的插件位置：`E:\zotero\2025_10_pdf2zh`

将你刚下载的内容把这里的替换掉

![image-20260917224941671](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917224941671.png)



然后来检查一下更新就行了

![image-20260917225839250](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260917225839250.png)















# 九十九、疑难杂症

## 99.1 VsCode相关问题

### 99.1.1 vscode路径

1、vscode中写相对路径 ./是你左边打开的文件夹，而不是你当前代码所在文件夹！——2026-07-05

修改之后，现在./就是当前代码所在文件夹——2026-07-31



### 99.1.2 VsCode终端中文显示问题

在代码中添加如下代码

```python
#VS Code 终端默认编码不是 UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字
```



### 99.1.3 VsCode无法输入中文问题

1. 在 VSCode 中按下 `Ctrl + ,` (逗号) 打开设置。
2. 在搜索框中输入 `editor.editContext`。
3. 在搜索结果中，**取消勾选** “**Editor: Edit Context**” 选项。
4. 关闭设置页面，**无需重启**，立即生效。





## 99.2 电脑验机——天选6

1、看外包装破损，右边的配置型号是否正确，OEAE是灰色 看三码是否一致

2、开箱，看电源适配器是否二次加封，插头是否有被使用过

3、检查电脑外观，是否有磕碰，检查背部螺丝和胶垫，检查按键回弹情况，屏幕是否被擦拭，划痕，转轴情况。

4、连接电源，开启电脑，指示灯为橙色正常，白色大概率有问题

5、进去 shift + F10,  输入代码start ms-cxh:locaonly跳过联网，用户名输入英文fgy，不用密码，两下回车

6、查看电池通电记录，win+r打开命令行，输入powercfg/batteryreport得到地址
	复制地址（不加末尾句号），打开浏览器搜索，往下翻，应该只有一条通电记录
	右键win,打开事件查看器，选择windows日志，点击应用程序，查看记录，与通电记录对的上

7、打开命令行，slmgr.vbs -vpr激活状态，显示通知模式

8、u盘拷贝图吧工具箱，查看
	硬件信息，看硬件是否正确
	磁盘工具，选择第二个，查看健康状态
	屏幕工具，坏点与漏光测试

9、菜单，搜索加密，点击设备加密设置，关闭设备加密
	点击奥创（下方应用最右边），点击跳过，下方 增强，点击电脑，打开GPU性能，选择独显输出

10、图吧工具箱
	烤机工具，点击64，点击传感器，打开后点击上方小火苗，左边选项只留第二个，start开始单烤，看CPU温度和功耗
	烤机工具，点击第一个，分辨率2500*1600，GPU点击go



