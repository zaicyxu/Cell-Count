# 细胞计数工作总结报告

# 第一阶段：

## 环境搭建

Jetson Xavier NX 环境搭建，主要为系统重装，jetpack4.6.2适配cuda10.2, cudnn8.2，安装miniconda，构建虚拟环境搭建paddlepaddlegpu以及搭载TensorRT加速框架，具体操作细节及问题处理方式请见如下章节：

[Paddle FastDeploy on Jetson NX](https://www.notion.so/Paddle-FastDeploy-on-Jetson-NX-dec7fbf6ed3146998e6c90e18e588c5c?pvs=21) 

[Jetson NX Plus SSD card](https://www.notion.so/Jetson-NX-Plus-SSD-card-20799fc071f64117887c2c5d10b7762d?pvs=21) 

[****Jetson Xavier NX风扇控制****](https://www.notion.so/Jetson-Xavier-NX-a58cf0d4c58b490fb16f33a97a5ee245?pvs=21) 

[Jetson NX update CMake](https://www.notion.so/Jetson-NX-update-CMake-14819762b67943d5a77c6ae7b6079577?pvs=21) 

系统重装，需要18.0.4ubuntu的虚拟环境，该虚拟环境在alien的电脑上由VMware构建，虚拟机的构建步骤参考如下帖子：

[https://zhuanlan.zhihu.com/p/41940739](https://zhuanlan.zhihu.com/p/41940739)

18.0.4的iso系统镜像文件地址：https://drive.google.com/file/d/1N2jpdSahMXIo5cRaCGjzKPr1Ih7cc6if/view?usp=sharing

我的开机密码为：123456

重装系统步骤参考如下帖子：

[https://blog.csdn.net/weixin_50060664/article/details/121722660](https://blog.csdn.net/weixin_50060664/article/details/121722660)

## PaddlePaddlegpu编译

paddle官网提供的paddlepaddlegpu版本需要严格与jetpack版本匹配，当前适用版本为4.6.2jetpack与paddlepaddle+gpu-2.3.2，注意编译版本为.aarch64, 以下是我已经编译好的.whl文件：https://drive.google.com/file/d/1d-o9LVGv37DE3Vba9vNvMxVmI78ODn_S/view?usp=sharing

## 打包环境搭建

需要安装在paddlepaddle_gou的同一个虚拟环境中，需要安装node 16.14.0 npm 8.3.1 electron 13.6.9，如果electron安装不成功，可以先只安装npm, node，然后进入cellcount文件夹运行npm run, 

打包的时候出现如下错误：

![Untitled](%E7%BB%86%E8%83%9E%E8%AE%A1%E6%95%B0%E5%B7%A5%E4%BD%9C%E6%80%BB%E7%BB%93%E6%8A%A5%E5%91%8A%20437f455bab4141959647af7e71913633/Untitled.jpeg)

解决方案如下：

在构建之前运行：

```jsx
sudo apt-get update
sudo gem install fpm
```

打包命令如下:

```jsx
npm run electron:linux
```

不打包试运行调试程序命令如下：

```jsx
npm run electron:serve
```

打包完成之后，进入dist_electron文件夹，里面已经生成了对应的.deb文件只需要”sudo dpkg -i xxx.deb”就可以安装文件，之后在application里面可以找到已安装的程序。

# 第二阶段：

## 分析参数算法：

```
survival_ratio = "{:.2f}%".format((cell_pos / (cell_pos + cell_neg)) * 100)
cell_all_concentration = "{:.2e} /mL".format(round(((cell_pos + cell_neg) / current_volum), 2))
cell_pos_concentration = "{:.2e} /mL".format(round((cell_pos / current_volum), 2))
cell_neg_concentration = "{:.2e} /mL".format(round((cell_neg / current_volum), 2))
```

细胞直径：

直径需要根据以下函数中对目标细胞画框然后选取框框比较短的一边作为直径，

```
@staticmethod
def result_visualize(im, results):
    cell_pos = 0
    cell_neg = 0
    diameters = []
    image = im.copy()

    for res in results:
        xmin, ymin, wid, hei = res[0], res[1], res[2] - res[0], res[3] - res[1]
        diameter = min(wid, hei)
        diameters.append(diameter)

        color = (0, 0, 255)
        if int(res[5]) == 2:
            color = (255, 0, 0)
            cell_neg += 1
        else:
            cell_pos += 1
        cv2.circle(image, (int(xmin + wid / 2), int(ymin + hei / 2)), int(np.sqrt(wid * wid + hei * hei) / 2 + 0.3),
                   thickness=2, color=color)
    return image, cell_pos, cell_neg, diameters
```

注意这里的直径单位是像素值，需要根据视野面积和面积内所有的像素值转化为单位微米，计算方式如下：

*`# current = best_code * 0.0288023`*

*`# current_area = -0.0086 * current + 4.0710`*

*`# diameters = [round(d * 0.55, 2) for d in diameters]`*

细胞团聚率需要在进行细胞NMS操作过程中识别有重叠的框的部分，然后累加计算重叠细胞个数。

```jsx
@staticmethod
def nms_patch(results, overlap):
        boxes = [[m[2], m[3], m[4], m[5], m[1]] for m in results]
        if len(boxes) == 0:
            pick = np.ones((len(boxes)), np.uint16) * -1
        else:
            trial = np.zeros((len(boxes), 5), dtype=np.float64)
            trial[:] = boxes[:]
            x1 = trial[:, 0]
            y1 = trial[:, 1]
            x2 = trial[:, 2]
            y2 = trial[:, 3]
            score = trial[:, 4]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            I = np.argsort(score)
            index1 = 0
            pick = np.ones((len(boxes)), np.uint16) * -1
            processed_boxes = set()
            aggregation = 0  # 初始化聚合率
            # aggregated_cells = set()
            while I.size != 0:
                last = I.size
                i = I[last - 1]
                pick[index1] = i
                index1 += 1
                index2 = 0
                suppress = np.ones((len(boxes)), np.int16) * -1
                suppress[index2] = last - 1
                cluster_boxes = [i]
                for pos in range(last - 1):
                    j = I[pos]
                    xx1 = max(x1[i], x1[j])
                    yy1 = max(y1[i], y1[j])
                    xx2 = min(x2[i], x2[j])
                    yy2 = min(y2[i], y2[j])
                    w = xx2 - xx1 + 1
                    h = yy2 - yy1 + 1
                    if w > 0 and h > 0:
                        o = w * h / area[j]
                        if o > overlap:
                            index2 += 1
                            suppress[index2] = pos
                            # 检测框有重叠，增加聚合率
                            aggregation += 1
                            # cluster_boxes.append(j)
                suppress = suppress[:index2 + 1]
                I = np.delete(I, suppress)
                processed_boxes.add(i)
                # 添加当前框到已处理列表
            pick = pick[:index1]
            aggregation = len(processed_boxes)
            # print(aggregation)
        return np.array(results)[pick], aggregation
```

为了解决没有细胞无法计算的情况出现，可视化计算函数需要修改如下：

```jsx
def image_predict(self, im, best_code):
        st = time()
        patches, locations, x_junction, y_junction = self.get_patches(im=im,
                                                                      patch_size=self.patch_size,
                                                                      overlay=self.overlay)
        results_all = []
        aggregations = 0
        for patch in patches:
            results = self.run_model(self.predictor, [np.array([patch]), self.scale_factor])
            results_nms, patch_aggregation = self.nms_patch(results[0], self.iou_thre)  # patch level nms
            results_all.append(results_nms)
            aggregations += patch_aggregation

        results_new = self.get_results(results_all, locations)  # convert whole slide coordinates and filter results
        results_final = self.nms_slide(results_new, self.iou_thre, x_junction, y_junction)  # nms for overlay areas

        im_v, cell_pos, cell_neg, diameters = self.result_visualize(im, results_final)
        # current = best_code * 0.0288023
        # current_area = -0.0086 * current + 4.0710
        # diameters = [round(((d * 1000000) / 6291456) * 1.27, 2) for d in diameters]
        # diameters = [round(d * 0.55, 2) for d in diameters]
        # current_area = ((-0.0067 * (best_code * 0.0288023) + 3.7428) * 0.1) / 1000
        current_volum = (3.5 * 0.1) / 1000
        # len_pixel = (current_area * 1000000) / 6291456
        # survival_ratio = "{:.2f}%".format((cell_pos / (cell_pos + cell_neg)) * 100)
        # cell_all_concentration = "{:.2e} /mL".format(round(((cell_pos + cell_neg) / current_volum), 2))
        # cell_pos_concentration = "{:.2e} /mL".format(round((cell_pos / current_volum), 2))
        # cell_neg_concentration = "{:.2e} /mL".format(round((cell_neg / current_volum), 2))
        # cell_average_diameters = f"{round(sum(diameters) / len(diameters), 2)} µm"
        # cell_aggregation_rate = "{:.2f}%".format(aggregations / (cell_pos + cell_neg))
        # cell_plt_image = self.plt_image(diameters)
        ed = time()
        if cell_pos + cell_neg > 0:
            diameters = [round(d * 0.55, 2) for d in diameters]
            survival_ratio = "{:.2f}%".format((cell_pos / (cell_pos + cell_neg)) * 100)
            cell_all_concentration = "{:.2e} /mL".format(round(((cell_pos + cell_neg) / current_volum), 2))
            cell_pos_concentration = "{:.2e} /mL".format(round((cell_pos / current_volum), 2))
            cell_neg_concentration = "{:.2e} /mL".format(round((cell_neg / current_volum), 2))
            cell_average_diameters = f"{round(sum(diameters) / len(diameters), 2)} µm"
            cell_aggregation_rate = "{:.2f}%".format(aggregations / (cell_pos + cell_neg))
            cell_plt_image = self.plt_image(diameters)
            print("活细胞总数：", cell_pos)
            print("死细胞总数：", cell_neg)
            print("细胞活率：", survival_ratio)
            print("总细胞浓度：", cell_all_concentration)
            print("活细胞浓度：", cell_pos_concentration)
            print("死细胞浓度：", cell_neg_concentration)
            print("平均细胞直径：", cell_average_diameters)
            print("细胞聚合率：", cell_aggregation_rate)
        else:
            print("未检出细胞！")
            diameters = [round(d * 0.55, 2) for d in diameters]
            survival_ratio = 0
            cell_all_concentration = 0
            cell_pos_concentration = 0
            cell_neg_concentration = 0
            cell_average_diameters = 0
            cell_aggregation_rate = 0
            cell_plt_image = None
            print("活细胞总数：", cell_pos)
            print("死细胞总数：", cell_neg)
            print("细胞活率：", survival_ratio)
            print("总细胞浓度：", cell_all_concentration)
            print("活细胞浓度：", cell_pos_concentration)
            print("死细胞浓度：", cell_neg_concentration)
            print("平均细胞直径：", cell_average_diameters)
            print("细胞聚合率：", cell_aggregation_rate)
        print("分析耗时（秒）：", round(ed - st, 2))
        return Image.fromarray(im_v), cell_pos, cell_neg, survival_ratio, cell_all_concentration, cell_pos_concentration, cell_neg_concentration, cell_average_diameters, cell_aggregation_rate, cell_plt_image
```

应甲方需求视野面积目前为固定值，如果需要根据实际情况确定面积的话，需要给best_code赋值，并且修改面积计算，计算方法为注释掉的部分，注意如果修改电流值，计算公式也会随之发生改变，需要重新计算。

需要注意传参个数与函数传入的个数需要对应上。

## 系统设置

### 开机自启

需要运行以下代码构建一个.sh文件：

```jsx
sudo vim xxx.sh
```

文件内容如下：

```jsx
#!/bin/bash

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate your_conda_environment_name
python /path/to/your/python_script.py
```

- **`/path/to/miniconda3/`** 替换为你的miniconda安装路径。
- **`your_conda_environment_name`** 替换为你的conda环境名。
- **`/path/to/your/python_script.py`** 替换为你的Python脚本的路径。

打开终端并运行以下命令来给你的脚本添加执行权限：

```bash

chmod +x /path/to/start_python_script.sh
```

创建一个新的systemd服务文件，例如**`my_python_script.service`**，在**`/etc/systemd/system/`**目录下，内容如下：

```bash

[Unit]
Description=My Python Script Service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/working/directory
ExecStart=/path/to/start_python_script.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

- **`your_username`** 替换为你的用户名。
- **`/path/to/working/directory`** 替换为你的Python脚本的工作目录。
- **`/path/to/start_python_script.sh`** 替换为你在步骤1中创建的bash脚本的路径。

现在需要重新加载systemd守护进程，使其识别新服务，然后启用和启动该服务。运行以下命令：

```bash

sudo systemctl daemon-reload
sudo systemctl enable my_python_script.service
sudo systemctl start my_python_script.service

```

使用下面的命令来检查服务的状态：

```bash
sudo systemctl status my_python_script.service
```

现在Python脚本应该在启动时自动运行在miniconda虚拟环境中。如果遇到任何问题，可以检查systemd日志以获得更多信息：

```bash
journalctl -u my_python_script.service
```

如果需要关闭开机自启，只需要打开最开始构造的.sh文件，将里面的内容注释，保存，然后运行如下代码：

```jsx
sudo systemctl restart xxx.service
```

即可，后端修改完成请记得重新取消注释掉的.sh文件，并重新运行如下代码，确保开启后端服务：

```jsx
sudo systemctl start my_python_script.service
sudo systemctl status my_python_script.service
```

### 取消锁屏和息屏

全部在setting里面操作。

在Ubuntu中取消锁屏唤醒密码的设置通常可以在图形用户界面(GUI)中或通过命令行完成。

1. **设置屏幕锁定选项**
    - 打开“设置”（Settings）。
    - 点击“隐私”（Privacy）。
    - 在左侧边栏中，点击“屏幕锁定”（Screen Lock）。
    - 关闭“自动锁定”（Automatic Screen Lock）选项或将“锁定屏幕时要求密码”（Require a password when waking from suspend）选项设置为“从不”（Never）。

1. **设置不锁屏**
    - 打开"设置"（可以在应用菜单中找到或通过点击右上角的系统托盘图标并选择“设置”来访问）。
    - 选择"电源"类别。
    - 在“电源保护”或者“屏幕保护”部分，你会找到“关闭屏幕”或者类似的选项，设置为“从不”。
2. **设置不休眠**
    - 在"设置"中选择"电源"类别。
    - 找到“自动挂起”或者类似的选项，设置“从不”或者选择一个适合你使用习惯的时间。

还可以通过下面的方式设置：

1. **禁用屏幕保护**
你可以使用**`gsettings`**来改变屏幕保护的设置：
    
    ```bash
    bashCopy code
    gsettings set org.gnome.desktop.session idle-delay 0
    
    ```
    
    这个命令将屏幕空闲时间设置为0，也就是说系统将不会进入屏幕保护模式。
    
2. **禁用自动锁屏**
    
    ```bash
    bashCopy code
    gsettings set org.gnome.desktop.screensaver lock-enabled false
    
    ```
    
    这个命令禁用了屏幕保护激活后自动锁屏的功能。
    
3. **禁用自动休眠**
    
    ```bash
    bashCopy code
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
    
    ```
    
    上述两个命令分别禁用了在接通电源和使用电池时的自动休眠功能。
    
    ## 多通道拍摄修改方案
    
    ### 多通道电机移动及聚焦修改
    
    修改思路为，将每一个通道内的电机移动指令由单一赋值改为list赋值，采用循环的的方式，并且对单通道内电机的移动进行计数，赋值给聚焦函数，使得每一个通道内的拍摄只聚焦一次。修改方案如下：
    
    ```python
    def pool_process():
        global Cell_Mode, Usb_mode, TU_cam
        pool_name = request.args.get('type')
        print(pool_name)
        if pool_name == '1':
            move_steps = [1869000, 1914500, 1962000]
        elif pool_name == '2':
            move_steps = [1485000, 1541500, 1568000]
        elif pool_name == '3':
            move_steps = [1120000, 1168500, 1200000]
        elif pool_name == '4':
            move_steps = [748000, 795500, 843000]
        elif pool_name == '5':
            move_steps = [377000, 422500, 470000]
        elif pool_name == '6':
            move_steps = [4000, 49500, 100000]
    
        save_option = request.args.get('store')
        # if save_option == '1':
        imgs = []
        move_len = len(move_steps)
        pool_best_code = None
        for i in range(0, move_len):
            move_step = move_steps[i]
            usb_type, motor_status, motor_num = Usb_mode.motor_control(0, move_step, 1)
            if motor_status == 1:
                # if move_step == 795500 or 422500 or 49500:
                #     sleep(0.5)
                # pool init best code
                if i == 0:
                    img, best_code = auto_focus(Usb_mode, TU_cam, pool_name)
                    pool_best_code = best_code
                else:
                    # direct get image
                    img = TU_cam.frame
                    sleep(0.2)
                    img = TU_cam.frame
                if img is None:
                    return jsonify({'error': 'fail image get'})
            else:
                return jsonify({'error': 'motor fail move'})
            if save_option == '0':
                save_channel_im(img)
            imgs.append((img.copy(), pool_best_code))
            # value = mode_predict(Cell_Mode, img, pool_name, best_code)
            # print(value)
            # img = cv2.imread(r"D:\aworks\cell_count_final\cell_count_jetson\test\A2780.tif")
            # sleep(2)
        thread = threading.Thread(target=mode_predict, args=(Cell_Mode, imgs, pool_name))
        thread.setDaemon(True)
        thread.start()
        # mode_predict(Cell_Mode, img, pool_name)
        return jsonify({'message': pool_name})
    ```
    
    同时，由于现在一个通道需要计算三张图片的参数，显示的参数需要是三张图片的平均值，显示的图片则需要是三张图片，所以修改思路是在计算阶段传入包含三张图片的list, 计算的得到其平均值，返还给函数，但是通过处理得到的图片则需要另外存储成一个包含图片的list，修改意见如下：
    
    ```python
    def image_predict(self, im_list):
            st = time()
            total_cell_pos = 0
            total_cell_neg = 0
            total_aggregations = 0
            im_v_list = None
            cell_total_list = []
            for ind, (img, best_code) in enumerate(im_list):
                if im_v_list is None:
                    h, w, c = img.shape
                    scale = min(1920 * 0.8 / w, 1080 * 0.8 / h)
                    im_v_list = np.zeros((len(im_list), int(h * scale), int(w * scale), c), np.uint8)
                patches, locations, x_junction, y_junction = self.get_patches(img,
                                                                              patch_size=self.patch_size,
                                                                              overlay=self.overlay)
                results_all = []
                aggregations = 0
                for patch in patches:
                    results = self.run_model(self.predictor, [np.array([patch]), self.scale_factor])
                    results_nms, patch_aggregation = self.nms_patch(results[0], self.iou_thre)  # patch level nms
                    results_all.append(results_nms)
                    aggregations += patch_aggregation
    
                results_new = self.get_results(results_all, locations)  # convert whole slide coordinates and filter results
                results_final = self.nms_slide(results_new, self.iou_thre, x_junction, y_junction)  # nms for overlay areas
    
                im_v, cell_pos, cell_neg, diameters = self.result_visualize(img, results_final)
                # im_v_list.append(im_v)
                im_v_list[ind] = cv2.resize(im_v, (int(w * scale), int(h * scale)))
                total_cell_neg += cell_neg
                total_cell_pos += cell_pos
                diameters.extend(diameters)
                total_aggregations += aggregations
    
                total_cells_for_single_image = cell_pos + cell_neg
                cell_total_list.append(total_cells_for_single_image)
    
            cell_pos = total_cell_pos // len(im_list)
            cell_neg = total_cell_neg // len(im_list)
            aggregations = total_aggregations / len(im_list)
            print(cell_total_list)
            # im_v_list = [Image.fromarray(im) for im in im_v_list]
            # current = best_code * 0.0288023
            # current_area = -0.0086 * current + 4.0710
            # diameters = [round(((d * 1000000) / 6291456) * 1.27, 2) for d in diameters]
            # diameters = [round(d * 0.55, 2) for d in diameters]
            # current_area = ((-0.0067 * (best_code * 0.0288023) + 3.7428) * 0.1) / 1000
            current_volum = (3.5 * 0.1) / 1000
    
            ed = time()
            if cell_pos + cell_neg > 0:
                diameters = [round(d * 0.55, 2) for d in diameters]
                survival_ratio = "{:.2f}%".format((cell_pos / (cell_pos + cell_neg)) * 100)
                cell_all_concentration = "{:.2e} /mL".format(round(((cell_pos + cell_neg) / current_volum), 2))
                cell_pos_concentration = "{:.2e} /mL".format(round((cell_pos / current_volum), 2))
                cell_neg_concentration = "{:.2e} /mL".format(round((cell_neg / current_volum), 2))
                cell_average_diameters = f"{round(sum(diameters) / len(diameters), 2)} µm"
                cell_aggregation_rate = "{:.2f}%".format(aggregations / (cell_pos + cell_neg))
                cell_plt_image = self.plt_image(diameters)
                print("活细胞总数：", cell_pos)
                print("死细胞总数：", cell_neg)
                print("细胞活率：", survival_ratio)
                print("总细胞浓度：", cell_all_concentration)
                print("活细胞浓度：", cell_pos_concentration)
                print("死细胞浓度：", cell_neg_concentration)
                print("平均细胞直径：", cell_average_diameters)
                print("细胞聚合率：", cell_aggregation_rate)
            else:
                print("未检出细胞！")
                diameters = [round(d * 0.55, 2) for d in diameters]
                survival_ratio = 0
                cell_all_concentration = 0
                cell_pos_concentration = 0
                cell_neg_concentration = 0
                cell_average_diameters = 0
                cell_aggregation_rate = 0
                cell_plt_image = None
                print("活细胞总数：", cell_pos)
                print("死细胞总数：", cell_neg)
                print("细胞活率：", survival_ratio)
                print("总细胞浓度：", cell_all_concentration)
                print("活细胞浓度：", cell_pos_concentration)
                print("死细胞浓度：", cell_neg_concentration)
                print("平均细胞直径：", cell_average_diameters)
                print("细胞聚合率：", cell_aggregation_rate)
            print("分析耗时（秒）：", round(ed - st, 2))
            return im_v_list, cell_total_list, cell_pos, cell_neg, survival_ratio, cell_all_concentration, cell_pos_concentration, cell_neg_concentration, cell_average_diameters, cell_aggregation_rate, cell_plt_image
    ```
    
    最后，由于需要传输给前端包含每个通道三张图片以及各自的细胞数目的list，所以在与前端交互过程中的信息交互部分也需要修改，修改意见如下：
    
    ```python
    def mode_predict(model_infer, images_list, pool_num):
        try:
            # img_cp = images.copy()
            # img, best_code = imgs[0][:]
            im_v_list, cell_total_list, cell_pos, cell_neg, survival_ratio, cell_all_concentration, cell_pos_concentration, \
            cell_neg_concentration, cell_average_diameters, cell_aggregation_rate, cell_plt_image = model_infer.image_predict(images_list)
    
            base64_img_list = []
            base64_img_ori_list = [] # 创建一个空列表，用于存放base64编码的图片
            res_len = len(images_list)
            for i in range(0, res_len):
                # result_image into base64
                im_v = Image.fromarray(im_v_list[i])
                img_width, img_height = im_v.size
                img_buffer = BytesIO()
                im_v.save(img_buffer, format='JPEG')
                image_bytes1 = img_buffer.getvalue()
                encoded_image = base64.b64encode(image_bytes1).decode('ascii')
                base64_img_list.append(encoded_image)
    
                # ori image into base64
                image = images_list[i]
                image = cv2.resize(image, (img_width, img_height))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ori_image = Image.fromarray(image_rgb)
                img_ori = BytesIO()
                ori_image.save(img_ori, format='JPEG')
                image_ori = img_ori.getvalue()
                encoded_image = base64.b64encode(image_ori).decode('ascii')
                base64_img_ori_list.append(encoded_image)
            # for im_v in im_v_list:
            #     img_buffer = BytesIO()
            #     im_v.save(img_buffer, format='JPEG')
            #     image_bytes1 = img_buffer.getvalue()
            #     encoded_image = base64.b64encode(image_bytes1).decode('ascii')  # 使用新的变量名避免重复
            #     base64_img_list.append(encoded_image)
            # # combined_list = [(base64_img, cell_total) for base64_img, cell_total in zip(base64_img_list, cell_total_list)]
            # # keys = ['img1', 'img2', 'img3']
            # # combined_dict = dict(zip(keys, base64_img_list))
            #
            # # base64_img_ori_list = []  # 修复后的变量名，避免冲突
            # for (image, best_code) in images_list:
            #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     ori_image = Image.fromarray(image_rgb)
            #     img_ori = BytesIO()
            #     ori_image.save(img_ori, format='JPEG')
            #     image_ori = img_ori.getvalue()
            #     encoded_image = base64.b64encode(image_ori).decode('ascii')  # 使用新变量名避免冲突
            #     base64_img_ori_list.append(encoded_image)
            cell_info = dict()
            cell_info['pool_num'] = pool_num
            cell_info['pos_cell'] = cell_pos
            cell_info['neg_cell'] = cell_neg
            cell_info['survival_ratio'] = survival_ratio
            cell_info['cell_all_concentration'] = cell_all_concentration
            cell_info['cell_pos_concentration'] = cell_pos_concentration
            cell_info['cell_neg_concentration'] = cell_neg_concentration
            cell_info['cell_average_diameter'] = cell_average_diameters
            cell_info['cell_aggregation_rate'] = cell_aggregation_rate
            cell_info['cell_plt_image'] = cell_plt_image
            cell_info['cell_number'] = cell_total_list
            cell_info['cell_img'] = base64_img_list
            cell_info['original_image'] = base64_img_ori_list
            # print(cell_info)
            results.put(cell_info)
        except:
            traceback.print_exc()
    ```
    
    ## 内存及计算资源不足修改方案