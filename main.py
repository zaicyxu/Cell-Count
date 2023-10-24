# !/usr/bin/python3
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
# from usb_run import USBDevice
import traceback
import numpy as np
# import ctypes
from CameraControl import CameraControl
from usb_run import USBDevice
from Normal_Utils import brenner
import threading
import queue
from cell_count_jetson.infer_paddle_class import CellInfer
import cv2
import os
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime
from time import time, sleep
from Configurations import *

# app #
app = Flask(__name__)
Cell_Mode = CellInfer()
# Usb_mode = None
# 使用队列来存储线程结果
results = queue.Queue(8)


def tmp_save(im, name):
	os.makedirs(os.path.join(SAVE_ROOT, 'tmp'), exist_ok=True)
	cv2.imwrite(os.path.join(SAVE_ROOT, 'tmp', name + '.png'), im)


def save_channel_im(im):
	try:
		save_path = '/home/ailab/python_sdkcam/samples/test_img'
		os.makedirs(save_path, exist_ok=True)
		current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		img_filename = f"{current_timestamp}.tif"
		img_save_path = os.path.join(save_path, img_filename)
		if get_dir_size(save_path) < 15 * (1024 ** 3):
			cv2.imwrite(img_save_path, im)
		else:
			print("Directory size exceeds 15GB, image nolonger saved.")
	except:
		pass


@app.route('/thoroughfare', methods=['GET'])
def pool_process():
	global Usb_mode, TU_cam
	pool_name = request.args.get('type')
	print(pool_name)
	move_steps = POOL_STEPS[pool_name]
	save_option = request.args.get('store')
	imgs_gray = None
	# 在一个通道中拍几张图
	move_len = len(move_steps)
	pool_best_code = None
	for i in range(0, move_len):
		# 该通道的第i张图
		move_step = move_steps[i]
		usb_type, motor_status, motor_num = Usb_mode.motor_control(0, move_step, 1)
		# 如果获取到图像
		if motor_status == 1:
			# 并且为第一张图
			if i == 0:
				img, best_code = auto_focus(Usb_mode, TU_cam, pool_name)
				pool_best_code = best_code
			else:
				_ = TU_cam.frame
				sleep(0.2)
				img = TU_cam.frame
			if img is None:
				return jsonify({'error': 'fail image get'})
			else:
				# 保存用来显示的原图
				tmp_save(img, 'vis_org_' + pool_name + '_' + str(i))
				# 将图像从彩色变成黑白
				img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				VIS_BUFFER_ORG[i] = img
				IM_BUFFER[i] = img_gray
		else:
			return jsonify({'error': 'motor fail move'})
		# if save_option == '0':
		# 	# 存的时候存彩色图
		# 	save_channel_im(img)
	# 拿到这3张图之后进行计算，进去的是单通道图像
	# thread = threading.Thread(target=mode_predict, args=(Cell_Mode, imgs, pool_name))
	# thread.setDeamon(True)
	# thread.start()
	mode_predict(Cell_Mode, IM_BUFFER, VIS_BUFFER_ORG, pool_name, pool_best_code)
	return jsonify({'message': pool_name})


@app.route('/get_result', methods=['GET'])
def get_result():
	if not results.empty():
		return jsonify(results.get())
	else:
		return 'none'


def mode_predict(model_infer, im_stack, vis_stack, pool_name, best_code):
	"""im_stack传进来的是单通道
	"""
	try:
		im_vis_stack, cell_total_list, cell_pos, cell_neg, survival_ratio, cell_all_concentration, \
		cell_pos_concentration, cell_neg_concentration, cell_average_diameters, \
		cell_aggregation_rate, cell_vis_his = model_infer.image_predict(im_stack, vis_stack, best_code)

		base64_img_list = []
		base64_img_ori_list = []  # 创建一个空列表，用于存放base64编码的图片
		res_len = len(im_stack)
		for i in range(0, res_len):
			# result_image into base64
			im_v = im_vis_stack[i]
			if DISK_TYPE:
				tmp_save(im_v, 'vis_res_' + pool_name + '_' + str(i))
			else:
				h, w, _ = im_v.shape
				im_v = Image.fromarray(im_v)
				img_buffer = BytesIO()
				im_v.save(img_buffer, format='JPEG')
				image_bytes1 = img_buffer.getvalue()
				encoded_image = base64.b64encode(image_bytes1).decode('ascii')  # 使用新的变量名避免重复
				base64_img_list.append(encoded_image)
				# ori image into base64
				image = im_stack[i]
				image = cv2.resize(image, (w, h))
				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				ori_image = Image.fromarray(image_rgb)
				img_ori = BytesIO()
				ori_image.save(img_ori, format='JPEG')
				image_ori = img_ori.getvalue()
				encoded_image = base64.b64encode(image_ori).decode('ascii')  # 使用新变量名避免冲突
				base64_img_ori_list.append(encoded_image)

		cell_info = dict()
		cell_info['pool_num'] = pool_name
		cell_info['pos_cell'] = cell_pos
		cell_info['neg_cell'] = cell_neg
		cell_info['survival_ratio'] = survival_ratio
		cell_info['cell_all_concentration'] = cell_all_concentration
		cell_info['cell_pos_concentration'] = cell_pos_concentration
		cell_info['cell_neg_concentration'] = cell_neg_concentration
		cell_info['cell_average_diameter'] = cell_average_diameters
		cell_info['cell_aggregation_rate'] = cell_aggregation_rate
		cell_info['cell_number'] = cell_total_list
		if not DISK_TYPE:
			cell_info['cell_plt_image'] = cell_vis_his
			cell_info['cell_img'] = base64_img_list
			cell_info['original_image'] = base64_img_ori_list
		else:
			tmp_save(cell_vis_his, 'vis_his' + pool_name + '_' + str(i))
		# print(cell_info)
		results.put(cell_info)
	except:
		traceback.print_exc()


@app.route('/usb_connect', methods=['POST'])
def usb_connect():
	global Usb_mode, TU_cam
	try:
		data_dic = request.get_json()
		vid = data_dic['vid']
		pid = data_dic['pid']
		inferce_num = data_dic['interface_number']
		Usb_mode = USBDevice(vendor_id=vid, product_id=pid)
		Usb_mode.connect(inferce_num)
		usb_type, motor_status, motor_num = Usb_mode.motor_control(0, 0, 0)
		if motor_status == 1:
			print('success reset')
		else:
			raise ValueError('motor fail')
		TU_cam = CameraControl()
		ret = TU_cam.onOpen()
		ret = TU_cam.openCamera()
		return jsonify({'message': 'success'})
	except Exception as e:
		print(traceback.print_exc())
		return jsonify({'error': str(e)})


@app.route('/usb_motor_control', methods=['POST'])
def motor_control():
	global Usb_mode
	try:
		data_dic = request.get_json()
		motor_num = data_dic['motor_num']
		move_step = data_dic['move_step']
		action = data_dic['action']
		usb_type, motor_status, motor_num = Usb_mode.motor_control(motor_num, move_step, action)
		# usb_type = 0
		# motor_status = 1
		# motor_num = 2314
		return jsonify({'usb_type': usb_type, 'status': motor_status, 'motor_num': motor_num})
	except Exception as e:
		print(traceback.print_exc())
		return jsonify({'error': str(e)})


@app.route('/usb_led_light', methods=['POST'])
def led_light():
	global Usb_mode, TU_cam
	try:
		data_dic = request.get_json()
		light_val = data_dic['led_light']
		TU_cam = CameraControl()
		# usb_type, status = Usb_mode.light_control(light_val)
		img = auto_focus(Usb_mode, TU_cam)
	# return jsonify({'usb_type': usb_type, 'status': status})
	except Exception as e:
		print(traceback.print_exc())
		return jsonify({'error': str(e)})


def get_dir_size(path='.'):
	total = 0
	with os.scandir(path) as it:
		for entry in it:
			if entry.is_file():
				total += entry.stat().st_size
			elif entry.is_dir():
				total += get_dir_size(entry.path)
	return total


def auto_focus(Usb_mode, TU_cam, pool_name):
	try:
		st = time()
		# cam = Tu_Camera()
		# ret = cam.camera_initial()
		# ret = cam.open()
		info_list = []
		prev_grad = float('-inf')
		dir_path = r"/home/ailab/python_sdkcam/samples/test_img"
		dir_path = dir_path + '_' + pool_name
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)
		grad_list = []
		if pool_name == '1':
			point = 500
		elif pool_name == '2':
			point = -800
		elif pool_name == '3':
			point = -1000
		elif pool_name == '4':
			point = -950
		elif pool_name == '5':
			point = -500
		elif pool_name == '6':
			point = 0
		start_point = -1000 + point
		end_point = 1000 + point
		best_cu_grad = 0
		best_cu_code = None
		for code in range(start_point, end_point, 200):
			if code > 0:
				direction = 1
			else:
				direction = 0
			len_code = abs(code)
			usb_type, status = Usb_mode.liquid_lens_control(len_code, direction)
			if status == 1:
				# 取图bgr
				_ = TU_cam.frame
				sleep(0.2)
				img = TU_cam.frame
				im_downsample = cv2.resize(img, None, fx=0.25, fy=0.25)
				grad = brenner(im_downsample)
				# grad_list.append(grad)
				if SAVE_FOCUS_IM:
					img_path = os.path.join(dir_path, str(code) + '.tif')
					cv2.imwrite(img_path, img)
				# 如果当前的grad小于前一个grad
				# if grad < prev_grad:
				# #     # 保留当前grad和前两个grad
				#     if len(info_list) >= 2:
				#         info_list = info_list[-2:] + [code]
				#     else:
				#         info_list = [info_list[-1]] + [code]
				#     break
				if grad > best_cu_grad:
					best_cu_grad = grad
					best_cu_code = code
			# info_list.append((code, grad))
			# prev_grad = grad
			else:
				raise ValueError('control lens error')
		# best_val = max(info_list, key=lambda x: x[1])[0]
		# ind = np.argmax(np.array(np.array(info_list)[:, 1]))
		# best_val = info_list[max(0, ind)][0]
		# print('cujujiao code', np.array(info_list)[:, 0])
		# print('cujujiao value', np.array(info_list)[:, 1])
		print('bestcode', best_cu_code)
		# prev_grad = (0, None)
		# prev_prev_grad = (0, None)
		total_list = list()
		# best_img = None
		best_grad = 0
		best_img = None
		best_code = None
		for code in range(best_cu_code - 200, best_cu_code + 200 + 1, 50):
			if code > 0:
				direction = 1
			else:
				direction = 0
			len_code = abs(code)
			usb_type, status = Usb_mode.liquid_lens_control(len_code, direction)
			if status == 1:
				# 取图bgr
				_ = TU_cam.frame
				sleep(0.2)
				img = TU_cam.frame
				im_downsample = cv2.resize(img, None, fx=0.25, fy=0.25)
				grad = brenner(im_downsample)
				if SAVE_FOCUS_IM:
					img_path = os.path.join(dir_path, str(code) + '.tif')
					cv2.imwrite(img_path, img)
				# if prev_prev_grad[0] < prev_grad[0] > grad:
				#     if abs(prev_grad[0] - prev_prev_grad[0]) > 8000000 and abs(prev_grad[0] - grad) > 8000000:
				#         best_img = prev_grad[1]
				#         break
				# if grad > prev_grad[0]:
				#     candidate_best_img = img
				#     candidate_best_code = code
				# elif candidate_best_img is not None:
				#     best_img = candidate_best_img
				#     best_code = candidate_best_code
				#     print('best best code', best_code)
				#     break
				# img_path = os.path.join(dir_path, str(code)+'.tif')
				if grad > best_grad:
					best_grad = grad
					best_code = code
					best_img = img
			# prev_grad = (code, grad, img.copy())
			# cv2.imwrite(img_path, img)
			# if grad < prev_grad[0]:
			#     best_img = prev_grad[1]
			#     print(code)
			#     break
			# prev_grad = (code, grad, img)
			# total_list.append(prev_grad)
			else:
				raise ValueError('control lens error')
		# best_best_val = max(total_list, key=lambda x: x[1])
		# best_img = best_best_val[2]
		# best_code = best_best_val[0]
		# print('xijujiao code', np.array(total_list)[:, 0])
		# print('xijujiao value', np.array(total_list)[:, 1])
		print('bestbest code', best_code)
		if best_img is None:
			raise ValueError('no best focus ！！！')
		# img_stack = [m[1] for m in total_list]
		# grad_stack = [m[0] for m in total_list]
		# max_grad_idx = grad_stack.index(max(grad_stack))
		# best_img = img_stack[max_grad_idx]
		print(time() - st)
		# best_code = int(code - 50)
		return best_img, best_code
	except:
		print('error', code)


# return best_img


@app.route('/usb_lens_control', methods=['POST'])
def liquid_control():
	global Usb_mode
	try:
		data_dic = request.get_json()
		code_val = data_dic['code']
		direction = data_dic['direction']
		usb_type, status = Usb_mode.liquid_lens_control(code_val, direction)
		return jsonify({'usb_type': usb_type, 'status': status})
	except Exception as e:
		print(traceback.print_exc())
		return jsonify({'error': str(e)})


@app.route('/usb_lens_status', methods=['GET'])
def check_lens():
	try:
		usb_type, status = Usb_mode.liquid_lens_status()
		return jsonify({'usb_type': usb_type, 'status': status})
	except Exception as e:
		print(traceback.print_exc())
		return jsonify({'error': str(e)})


@app.route('/usb_lens_val', methods=['GET'])
def get_lens():
	try:
		usb_type, val = Usb_mode.liquid_lens_value()
		return jsonify({'usb_type': usb_type, 'electric_val': val})
	except Exception as e:
		print(traceback.print_exc())
		return jsonify({'error': str(e)})


if __name__ == "__main__":
	try:
		vid = 4404
		pid = 22134
		inferce_num = 0
		Usb_mode = USBDevice(vendor_id=vid, product_id=pid)
		Usb_mode.connect(inferce_num)
	except:
		print("USB connection failed!")

	TU_cam = None
	TU_cam = CameraControl()
	ret = TU_cam.onOpen()
	ret = TU_cam.openCamera()
	app.run(host="0.0.0.0", port=11001)
         #     break
            # prev_grad = (code, grad, img)
            # total_list.append(prev_grad)
            else:
                raise ValueError('control lens error')
        # best_best_val = max(total_list, key=lambda x: x[1])
        # best_img = best_best_val[2]
        # best_code = best_best_val[0]
        # print('xijujiao code', np.array(total_list)[:, 0])
        # print('xijujiao value', np.array(total_list)[:, 1])
        print('bestbest code', best_code)
        if best_img is None:
            raise ValueError('no best focus ！！！')
        # img_stack = [m[1] for m in total_list]
        # grad_stack = [m[0] for m in total_list]
        # max_grad_idx = grad_stack.index(max(grad_stack))
        # best_img = img_stack[max_grad_idx]
        print(time() - st)
        # best_code = int(code - 50)
        return best_img, best_code
    except:
        print('error', code)


# return best_img


@app.route('/usb_lens_control', methods=['POST'])
def liquid_control():
    global Usb_mode
    try:
        data_dic = request.get_json()
        code_val = data_dic['code']
        direction = data_dic['direction']
        usb_type, status = Usb_mode.liquid_lens_control(code_val, direction)
        return jsonify({'usb_type': usb_type, 'status': status})
    except Exception as e:
        print(traceback.print_exc())
        return jsonify({'error': str(e)})


@app.route('/usb_lens_status', methods=['GET'])
def check_lens():
    try:
        usb_type, status = Usb_mode.liquid_lens_status()
        return jsonify({'usb_type': usb_type, 'status': status})
    except Exception as e:
        print(traceback.print_exc())
        return jsonify({'error': str(e)})


@app.route('/usb_lens_val', methods=['GET'])
def get_lens():
    try:
        usb_type, val = Usb_mode.liquid_lens_value()
        return jsonify({'usb_type': usb_type, 'electric_val': val})
    except Exception as e:
        print(traceback.print_exc())
        return jsonify({'error': str(e)})


def initialize_project():
    cell_info_path = os.path.join(SAVE_ROOT, 'tmp', "cell_info.txt")
    if os.path.exists(cell_info_path):
        os.remove(cell_info_path)


if __name__ == "__main__":
    try:
        vid = 4404
        pid = 22134
        inferce_num = 0
        Usb_mode = USBDevice(vendor_id=vid, product_id=pid)
        Usb_mode.connect(inferce_num)
    except:
        print("USB connection failed!")
    TU_cam = None
    TU_cam = CameraControl()
    initialize_project()
    ret = TU_cam.onOpen()
    ret = TU_cam.openCamera()
    app.run(host="0.0.0.0", port=11001)
