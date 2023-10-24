# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author         :  Zhang Chukang / Rui Xu
@Version        :  5.1.0
------------------------------------
@File           :  infer_paddle_class.py
@Description    :  Added additional parameter calculations and corrected the calculation logic when there are no cells.
@CreateTime     :  2023/10
"""

from glob import glob
from time import time
import numpy as np
from paddle.inference import Config
from paddle.inference import Predictor
from paddle.inference import PrecisionType
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import io
from io import BytesIO
import random
import base64


cur_dir = os.path.dirname(os.path.abspath(__file__))

class CellInfer:
    def __init__(self):
        self.model_file = cur_dir + '/ppyoloe_s_new_data_update/model.pdmodel'
        self.params_file = cur_dir + '/ppyoloe_s_new_data_update/model.pdiparams'
        self.run_mode = 'false'
        self.input_size = 640
        self.patch_size = 640
        self.overlay = 32
        self.score_thre = 0.3
        self.iou_thre = 0.2
        self.best_code = None
        self.scale_factor = np.array([self.input_size / self.patch_size,
                                      self.input_size / self.patch_size]).reshape((1, 2)).astype(np.float32)
        self.predictor = self.init_predictor(self.model_file, self.params_file, self.run_mode)

    @staticmethod
    def init_predictor(model_file, params_file, run_mode):
        config = Config(model_file, params_file)
        config.enable_memory_optim()
        config.enable_use_gpu(1000, 0)
        if run_mode == "trt_fp32":
            config.enable_tensorrt_engine(workspace_size=1 << 30,
                                          max_batch_size=1,
                                          min_subgraph_size=5,
                                          precision_mode=PrecisionType.Float32,
                                          use_static=True,
                                          use_calib_mode=False)

        elif run_mode == "trt_fp16":
            config.enable_tensorrt_engine(workspace_size=1 << 30,
                                          max_batch_size=1,
                                          min_subgraph_size=5,
                                          precision_mode=PrecisionType.Half,
                                          use_static=True,
                                          use_calib_mode=False)

        elif run_mode == "trt_int8":
            config.enable_tensorrt_engine(workspace_size=1 << 30,
                                          max_batch_size=1,
                                          min_subgraph_size=5,
                                          precision_mode=PrecisionType.Int8,
                                          use_static=True,
                                          use_calib_mode=False)

        config.disable_glog_info()
        print("TensorRT Enabled: {}".format(config.tensorrt_engine_enabled()))
        predictor = Predictor(config)
        return predictor

    @staticmethod
    def run_model(predictor, img):
        # copy img data to input tensor
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)
            input_tensor.copy_from_cpu(img[i].copy())

        # do the inference
        predictor.run()

        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results

    @staticmethod
    def normalize(img, mean, std):
        img = img / 255.0
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        img -= mean
        img /= std
        return img

    @staticmethod
    def resize(img, target_size):
        if not isinstance(img, np.ndarray):
            raise TypeError('image type is not numpy.')
        im_shape = img.shape
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=2)
        return img

    def preprocess(self, img, img_size):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = self.resize(img, img_size)
        # img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
        img = self.normalize(img, mean, std)
        img = img.transpose((2, 0, 1))  # hwc -> chw
        img = img.astype('float32')
        return img[np.newaxis, :]

    def get_patches(self, im, patch_size, overlay):
        raw = cv2.cvtColor(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

        total_patch_num = 0
        locations = []
        x_junction = []
        y_junction = []
        for sh in range(0, raw.shape[0], patch_size - overlay):
            for sw in range(0, raw.shape[1], patch_size - overlay):
                # Extract patch overlay areas for slide level NMS
                if sw > 0:
                    x_range = list(range(sw - overlay, sw + overlay))
                    x_junction += x_range
                if sh > 0:
                    y_range = list(range(sh - overlay, sh + overlay))
                    y_junction += y_range
                # Extract patch coordinates
                locations.append([sh, sw])
                total_patch_num += 1

        patches = np.zeros((total_patch_num, 3, patch_size, patch_size)).astype(np.float32)
        cur_patch_num = 0
        for sh in range(0, raw.shape[0], patch_size - overlay):
            for sw in range(0, raw.shape[1], patch_size - overlay):
                base = np.zeros((patch_size, patch_size, 3)).astype(np.float32)
                patch = raw[sh:sh + patch_size, sw:sw + patch_size, ].astype(np.float32)
                if patch.shape != [patch_size, patch_size]:
                    base[:patch.shape[0], :patch.shape[1], ] = patch
                    base = self.preprocess(base, self.input_size)
                    patches[cur_patch_num] = base
                else:
                    patch = self.preprocess(patch, self.input_size)
                    patches[cur_patch_num] = patch
                cur_patch_num += 1
        return patches, locations, set(x_junction), set(y_junction)

    def get_results(self, results, locations):
        total_results = []
        for i, result in enumerate(results):
            sh = locations[i][0]
            sw = locations[i][1]
            for j in result:
                result_new = {}
                x_min = j[2] + sw
                y_min = j[3] + sh
                x_max = j[4] + sw
                y_max = j[5] + sh
                result_new['category'] = str(j[0] + 1)
                result_new['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                result_new['score'] = j[1]
                total_results.append(result_new)
            total_results = [m for m in filter(lambda x: x['score'] > self.score_thre, total_results)]
            total_results = [m for m in filter(lambda x: 1 / 2 < x['bbox'][2] / x['bbox'][3] < 2, total_results)]
        return total_results

    @staticmethod
    # def nms_patch(results, overlap):
    #     boxes = [[m[2], m[3], m[4], m[5], m[1]] for m in results]
    #     if len(boxes) == 0:
    #         pick = np.ones((len(boxes)), np.uint16) * -1
    #     else:
    #         trial = np.zeros((len(boxes), 5), dtype=np.float64)
    #         trial[:] = boxes[:]
    #         x1 = trial[:, 0]
    #         y1 = trial[:, 1]
    #         x2 = trial[:, 2]
    #         y2 = trial[:, 3]
    #         score = trial[:, 4]
    #         area = (x2 - x1 + 1) * (y2 - y1 + 1)
    #         I = np.argsort(score)
    #         index1 = 0
    #         pick = np.ones((len(boxes)), np.uint16) * -1
    #         while I.size != 0:
    #             last = I.size
    #             i = I[last - 1]
    #             pick[index1] = i
    #             index1 += 1
    #             index2 = 0
    #             suppress = np.ones((len(boxes)), np.int16) * -1
    #             suppress[index2] = last - 1
    #             for pos in range(last - 1):
    #                 j = I[pos]
    #                 xx1 = max(x1[i], x1[j])
    #                 yy1 = max(y1[i], y1[j])
    #                 xx2 = min(x2[i], x2[j])
    #                 yy2 = min(y2[i], y2[j])
    #                 w = xx2 - xx1 + 1
    #                 h = yy2 - yy1 + 1
    #                 if w > 0 and h > 0:
    #                     o = w * h / area[j]
    #                     if o > overlap:
    #                         index2 += 1
    #                         suppress[index2] = pos
    #             suppress = suppress[:index2 + 1]
    #             I = np.delete(I, suppress)
    #         pick = pick[:index1]
    #     return np.array(results)[pick]
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

    @staticmethod
    def nms_slide(results, overlap, x_junction, y_junction):
        boxes = [[m['bbox'][0], m['bbox'][1], m['bbox'][0] + m['bbox'][2], m['bbox'][1] + m['bbox'][3], m['score'],
                  m['category']] for m in results
                 if int(m['bbox'][0]) in x_junction or int(m['bbox'][1]) in y_junction]
        boxes_keep = [[m['bbox'][0], m['bbox'][1], m['bbox'][0] + m['bbox'][2], m['bbox'][1] + m['bbox'][3], m['score'],
                       m['category']] for m in results
                      if int(m['bbox'][0]) not in x_junction and int(m['bbox'][1]) not in y_junction]
        if len(boxes) == 0:
            boxes_all = np.array(boxes_keep).astype(np.float64)
        else:
            trial = np.zeros((len(boxes), 6), dtype=np.float64)
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
            while I.size != 0:
                last = I.size
                i = I[last - 1]
                pick[index1] = i
                index1 += 1
                index2 = 0
                suppress = np.ones((len(boxes)), np.int16) * -1
                suppress[index2] = last - 1
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
                suppress = suppress[:index2 + 1]
                I = np.delete(I, suppress)
            pick = pick[:index1]

            boxes_all = np.concatenate((np.array(boxes)[pick].astype(np.float64),
                                        np.array(boxes_keep).astype(np.float64)), axis=0)
        return boxes_all

    @staticmethod
    def result_visualize(im, results):
        cell_pos = 0
        cell_neg = 0
        diameters = []
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

    @staticmethod
    def plt_image(diameters):
        min_diameter = min(diameters)
        max_diameter = max(diameters)
        num_bins = int(max_diameter - min_diameter) + 1
        hist, bins = np.histogram(diameters, bins=np.arange(min_diameter, max_diameter+2), range=(min_diameter, max_diameter+1))
        plt.bar(np.arange(min_diameter, max_diameter+1), hist, align='edge', edgecolor='black', width=1)

        # 设置图表标签和标题
        plt.xlabel('cell diameter(μm)')
        plt.ylabel('cell number')
        plt.title('Cell diameter distribution chart')

        # 在每个柱子顶部显示细胞数量
        for i, v in enumerate(hist):
            plt.text(i + min_diameter + 0.5, v + 1, str(v), ha='center', va='bottom', fontsize=8)
        # plt.savefig('cell_plt_image.png', format='png', bbox_inches='tight')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_data = buffer.read()

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        plt.close()
        return image_base64


if __name__ == '__main__':
    model_infer = CellInfer()
    # imgs = glob("./test/*.tif")
    img_paths = glob("./test/*.tif")
    imgs = [(np.array(Image.open(path)), random.randint(0, 100)) for path in img_paths]
    # best_code = None
    im_v_list, cell_total_list, cell_pos, cell_neg, survival_ratio, cell_all_concentration, cell_pos_concentration, cell_neg_concentration, cell_average_diameters, cell_aggregation_rate, cell_plt_image = model_infer.image_predict(imgs)
    print(cell_total_list)
    # im_v.save(raw.replace('.tif', '_paddle_infer_int8.png'))

