import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from utils import Path_utils
from utils import image_utils
import cv2
import models
from interact import interact as io
import matplotlib.pyplot as plt
import matplotlib.image as img

def main(args, device_args):
    io.log_info ("Running trainer.\r\n")

    previews = None
    loss_history = None
    selected_preview = 0
    update_preview = False
    is_showing = False
    is_waiting_preview = False
    show_last_history_iters_count = 0
    iter = 0

    training_data_src_path = Path( args.get('training_data_src_dir', '') )
    training_data_dst_path = Path( args.get('training_data_dst_dir', '') )
    model_path = Path( args.get('model_path', '') )
    model_name = args.get('model_name', '')
    save_interval_min = 15
    debug = args.get('debug', '')

    if not model_path.exists():
        model_path.mkdir(exist_ok=True)

    model = models.import_model(model_name)(
                                    model_path,
                                    training_data_src_path=training_data_src_path,
                                    training_data_dst_path=training_data_dst_path,
                                    debug=debug,
                                    device_args=device_args)

    is_reached_goal = model.is_reached_iter_goal()
    is_upd_save_time_after_train = False

    def model_save():
        if not debug and not is_reached_goal:
            model.save()
            is_upd_save_time_after_train = True

    def send_preview(title):
        clear_output()
#        imdata = img.imread(img_path+str(iter).zfill(6)+'.jpg')
        imdata=model.get_static_preview()
#        print(imdata)
        plt.title(title)
        plt.figure(figsize=(10,10))
        plt.axis('off')
        plt.imshow(imdata)
        plt.show()

        if model.is_first_run():
            model_save()

    if model.get_target_iter() != 0:
        if is_reached_goal:
            print('Model already trained to target iter. You can use preview.')
        else:
            print('Starting. Target iter: %d. Press "Enter" to stop training and save model.' % ( model.get_target_iter()  ) )

    last_save_time = time.time()

    for i in itertools.count(0,1):
        if not debug:
            if not is_reached_goal:

                try:
                    loss_string = model.train_one_iter()
#                    send_preview(model.iter)
                except KeyboardInterrupt:
                    model_save()
                    model.finalize()

                if is_upd_save_time_after_train:
                    #save resets plaidML programs, so upd last_save_time only after plaidML rebuild them
                    last_save_time = time.time()

#                from google.colab import output
                io.log_info('\r'+loss_string, end='')
#                output.clear(wait=True)

                if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                    print ('Reached target iter.')
                    model_save()
                    is_reached_goal = True
                    break

                if not is_reached_goal and (time.time() - last_save_time) >= save_interval_min*60:
                    last_save_time = time.time()
                    model_save()
                    #send_preview(model.iter)

                if i==0:
                    if is_reached_goal:
                        model.pass_one_iter()

                if debug:
                    time.sleep(0.005)

    model.finalize()
