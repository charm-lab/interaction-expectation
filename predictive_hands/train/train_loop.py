import os
import pickle
import random
from pathlib import Path

import numpy as np
import seaborn as sns;
import torch
import yaml
import yappi
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from predictive_hands.data_loading.DataContainer import DataContainer
from predictive_hands.data_loading.GRABInstance import GRABInstance
from predictive_hands.training_methods.DenseNetLSTM import DenseNetLSTM
from predictive_hands.utilities.visualize_data import VisualizeHands

sns.set_theme()
import matplotlib.pyplot as plt
from copy import deepcopy

config_method_dict = {'denselstm': DenseNetLSTM}

def distance_error_base(predicted, target, device_type,):
    return torch.nn.functional.mse_loss(predicted, target)
def first_contact_error_base(predicted, target, device_type,):
    return torch.nn.functional.mse_loss(predicted, target)
#This currently ONLY works for batch size 1!
def first_contact_timing_base(predicted, target, device_type, threshold = .5, test_range = None):
    if test_range is not None:
        predicted_test = predicted[:,:, test_range:test_range+1]
        target_test = target[:,:,test_range:test_range+1]
    else:
        predicted_test = predicted
        target_test = target
    target_max = torch.max(target_test)
    predicted_timing = torch.nonzero(predicted_test > threshold)
    target_timing = torch.nonzero(target_test == target_max)
    if len(predicted_timing) > 0:
        if len(target_timing) > 0:
            t_timing = target_timing[0][1] + target_timing[0][2]
            p_timing = predicted_timing[0][1] + predicted_timing[0][2]
            if test_range is not None:
                prediction_range = torch.Tensor([test_range])
            else:
                prediction_range = predicted_timing[0][2]
            return (t_timing - p_timing, prediction_range)
        else:
            return None, None
    else:
        if len(target_timing) == 0:
            return torch.tensor(0.0).to(device_type)
        else:
            return None, None

def run_training(cfg_name):

    cfg = yaml.full_load(open(cfg_name,'r'))

    device_type = torch.device(f'cuda:{cfg["gpu_num"]}')

    torch.cuda.set_device(int(cfg["gpu_num"]))

    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    if cfg['profile']:
        yappi.set_clock_type("cpu") # Use set_clock_type("wall") for wall time
        yappi.start()

    if cfg['epoch_draw']:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        hand_vis = VisualizeHands(cfg, offscreen=True, obj_view = False)

    first_contact_timing = lambda predicted, target, device_type, threshold, test_range: first_contact_timing_base(predicted, target, device_type, threshold, test_range)

    writer = SummaryWriter(cfg['results_dir'])

    train_meta = val_meta = test_meta = None

    train_dataset = DataContainer(cfg, cfg['train_conditions'], device_type=device_type, meta_files_in=train_meta, randomized_start = cfg['random_start'])
    val_dataset = DataContainer(cfg, cfg['val_conditions'], device_type=device_type, meta_files_in=val_meta, randomized_start = True)
    test_dataset = DataContainer(cfg, cfg['test_conditions'], device_type=device_type, meta_files_in=test_meta, randomized_start = True)

    print(f"len train: {len(train_dataset)}")
    print(f"len val: {len(val_dataset)}")
    print(f"len test: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=0)
    obj_counts = {}
    for in_dict, target_dict, meta_idx, meta_data in test_loader:
        if in_dict == 0:
            continue
        obj_name = meta_data['obj_name'][0]

        if obj_name not in obj_counts.keys():
            obj_counts[obj_name] = 0
        obj_counts[obj_name] += 1
    print(obj_counts)


    # we want input for all valid indices, we deal with test/train split below
    file_object = open(Path(cfg['models_dir']).joinpath("meta_files").with_suffix(".pkl"), 'wb')
    pickle.dump((train_dataset.meta_files, val_dataset.meta_files, test_dataset.meta_files), file_object)
    file_object.close()

    if cfg['parallel_load'] and not cfg['profile']:
        num_workers = int(os.cpu_count()-1)
        print(f'Num workers: {num_workers}')
    else:
        num_workers = 0


    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                        shuffle=True, num_workers=num_workers)
    
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                        shuffle=False, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=num_workers)


    # pass in single datapoint so network can use dimensionality to initialize
    single_train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                        shuffle=True, num_workers=num_workers)
    for in_dict, target_dict, idx, meta_data in single_train_loader:
        if in_dict == 0:
            continue
        else:
            training_network = config_method_dict[cfg['network_type']](cfg, in_dict, target_dict, device_type)
            break
    start_epoch = -1
    if Path(cfg['models_dir']).exists():
        file_nums = [int(x.stem.split('_')[-1]) for x in Path(cfg['models_dir']).glob("model_*.pt")]
        if len(file_nums) > 0:
            start_epoch = max(file_nums)
            checkpoint = torch.load(Path(cfg['models_dir']).joinpath("model_" + str(start_epoch)).with_suffix(".pt"))
            training_network.model.load_state_dict(checkpoint['model_state_dict'])
            training_network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            training_network.epoch = checkpoint['epoch']
            training_network.loss = checkpoint['loss']

    best_val = float('inf')
    for epoch in tqdm(range(start_epoch + 1, cfg['epochs']), initial=start_epoch + 1,):
        training_network.model.train()
        train_loss = 0
        for loader_out in train_loader:
            if loader_out[0] == 0:
                continue
            else:
                in_dict, target_dict, idx, meta_data = loader_out
            batch_size = idx.shape[0]
            train_loss += training_network.train_epoch(in_dict, target_dict)*batch_size
        train_loss /= len(train_dataset)
        
        val_loss = 0
        all_vals = {}
        first_contact_thresholds = list(np.arange(0, 1.01, .01))
        first_contact_timings = {}
        first_contact_totals = {}
        first_contact_fails = {}
        first_contact_ranges = {}
        for test_range in cfg['test_ranges']:
            first_contact_timings[test_range] = np.zeros((len(first_contact_thresholds)))
            first_contact_totals[test_range] = np.zeros((len(first_contact_thresholds)))
            first_contact_fails[test_range] = np.zeros((len(first_contact_thresholds)))
            first_contact_ranges[test_range] = np.zeros((len(first_contact_thresholds)))
            all_vals[test_range] = []
        for i in range(len(first_contact_thresholds)):
            all_vals[test_range].append({'error': torch.zeros((0)).cuda(), 'fail': 0})

        training_network.model.eval()
        counter = 0
        with torch.no_grad():
            for loader_out in val_loader:
                counter += 1
                if loader_out[0] == 0:
                    continue
                else:
                    in_dict, target_dict, idx, meta_data = loader_out
                batch_size = meta_idx.shape[0]
                val_loss_cur, val_predicted_batch, predicted_dict = training_network.val_epoch(in_dict, target_dict)
                val_loss += val_loss_cur * batch_size
                for hand in cfg['hands']:
                    for test_range in cfg['test_ranges']:
                        for i in range(len(first_contact_thresholds)):
                            if cfg['single_times']:
                                t_range = None
                            else:
                                t_range = test_range
                            f_timing, f_range = first_contact_timing(predicted_dict['contacts'][hand].to(device_type),
                                                                             target_dict['contacts'][hand].to(device_type),
                                                                             device_type,
                                                                             first_contact_thresholds[
                                                                                 i], t_range)

                            first_contact_totals[test_range][i] += 1
                            if f_timing is not None and f_timing > -max(cfg['test_ranges']):
                                first_contact_timings[test_range][i] += torch.abs(f_timing) * batch_size / len(cfg['hands'])
                                first_contact_ranges[test_range][i] += torch.abs(f_range) * batch_size / len(cfg['hands'])
                                all_vals[test_range][i]['error'] = torch.cat(
                                    (all_vals[test_range][i]['error'], torch.abs(f_timing).unsqueeze(0)))
                            else:
                                first_contact_fails[test_range][i] += 1

        val_loss /= len(val_dataset)
        for test_range in cfg['test_ranges']:
            for i in range(len(first_contact_thresholds)):
                first_contact_timings[test_range][i] /= (first_contact_totals[test_range][i] - first_contact_fails[test_range][i])
                first_contact_ranges[test_range][i] /= (first_contact_totals[test_range][i] - first_contact_fails[test_range][i])
                first_contact_fails[test_range][i] /= first_contact_totals[test_range][i]

        best_first_contact_timing = {}
        best_first_contact_fail = {}
        best_first_contact_thresh = {}
        best_first_contact_range = {}
        for test_range in cfg['test_ranges']:
            best_first_contact_timing[test_range] = float("inf")
            best_first_contact_fail[test_range] = 1
            best_first_contact_thresh[test_range] = 1
            best_first_contact_range[test_range] = float("inf")
            for i in range(len(first_contact_thresholds)):
                if first_contact_fails[test_range][i] <= .05 and first_contact_timings[test_range][i] < best_first_contact_timing[test_range]:
                    best_first_contact_timing[test_range] = first_contact_timings[test_range][i]
                    best_first_contact_fail[test_range] = first_contact_fails[test_range][i]
                    best_first_contact_thresh[test_range] = first_contact_thresholds[i]
                    best_first_contact_range[test_range] = first_contact_ranges[test_range][i]

        test_loss = 0
        first_contact_timing_test = {}
        first_contact_total_test = {}
        first_contact_fail_test = {}
        first_contact_range_test = {}
        for test_range in cfg['test_ranges']:
            first_contact_timing_test[test_range] = 0
            first_contact_total_test[test_range] = 0
            first_contact_fail_test[test_range] = 0
            first_contact_range_test[test_range] = 0
        if best_first_contact_timing[test_range] < best_val:
            obj_vals = {}
            subj_vals = {}
            all_vals_test = {'error': torch.zeros((0)).cuda(), 'fail': 0}
            with torch.no_grad():
                for loader_out in test_loader:
                    if loader_out[0] == 0:
                        continue
                    else:
                        in_dict, target_dict, idx, meta_data = loader_out
                    obj_name = meta_data['obj_name'][0]
                    subj_id = meta_data['subj_id'][0]
                    if obj_name not in obj_vals.keys():
                        obj_vals[obj_name] = {'error': torch.zeros((0)).cuda(), 'fail': 0}
                    if subj_id not in subj_vals.keys():
                        subj_vals[subj_id] = {'error': torch.zeros((0)).cuda(), 'fail': 0}
                    batch_size = meta_idx.shape[0]
                    test_loss_cur, test_predicted_batch, predicted_dict = training_network.val_epoch(in_dict, target_dict)
                    test_loss += test_loss_cur * batch_size
                    for hand in cfg['hands']:
                        for test_range in cfg['test_ranges']:
                            if cfg['single_times']:
                                t_range = None
                            else:
                                t_range = test_range
                            f_timing, f_range = first_contact_timing(
                                predicted_dict['contacts'][hand].to(device_type),
                                target_dict['contacts'][hand].to(device_type),
                                device_type,
                                best_first_contact_thresh[test_range], t_range)
                            first_contact_total_test[test_range] += 1
                            if f_timing is not None and f_timing > -max(cfg['test_ranges']):
                                first_contact_timing_test[test_range] += torch.abs(f_timing) * batch_size / len(
                                    cfg['hands'])
                                first_contact_range_test[test_range] += torch.abs(f_range) * batch_size / len(
                                    cfg['hands'])
                                obj_vals[obj_name]['error'] = torch.cat(
                                    (obj_vals[obj_name]['error'], torch.abs(f_timing).unsqueeze(0)))
                                subj_vals[subj_id]['error'] = torch.cat(
                                    (subj_vals[subj_id]['error'], torch.abs(f_timing).unsqueeze(0)))
                                all_vals_test['error'] = torch.cat(
                                    (all_vals_test['error'], torch.abs(f_timing).unsqueeze(0)))
                            else:
                                first_contact_fail_test[test_range] += 1
                                obj_vals[obj_name]['fail'] += 1
                                subj_vals[subj_id]['fail'] += 1
                                all_vals_test['fail'] += 1
            best_val = best_first_contact_timing[test_range]
            test_loss /= len(test_dataset)
            for test_range in cfg['test_ranges']:
                first_contact_timing_test[test_range] /= (
                            first_contact_total_test[test_range] - first_contact_fail_test[test_range])
                first_contact_range_test[test_range] /= (
                            first_contact_total_test[test_range] - first_contact_fail_test[test_range])
                first_contact_fail_test[test_range] /= first_contact_total_test[test_range]

            model = training_network.model
            optimizer = training_network.optimizer
            loss = training_network.loss
            for child in Path(cfg['models_dir']).glob('*'):
                if child.is_file():
                    child.unlink()
            torch.save({
            'epoch': epoch,
            'model_state_dict': deepcopy(model.state_dict()),
            'optimizer_state_dict': deepcopy(optimizer.state_dict()),
            'loss': loss
            }, Path(cfg['models_dir']).joinpath("model_" + str(epoch)).with_suffix(".pt"))

            for test_range in cfg['test_ranges']:
                writer.add_scalars("contact_metrics_test", {f'timing_error_{test_range}':
                                                           first_contact_timing_test[test_range],
                                                       f'timing_range_{test_range}':
                                                           first_contact_range_test[test_range],
                                                       f'fail_rate_{test_range}':
                                                           first_contact_fail_test[test_range],
                                                       f'threshold_{test_range}':
                                                           best_first_contact_thresh[test_range]
                                                       },
                                   epoch)

            writer.add_scalars("losses", {
                                          'test':
                                              test_loss
                                          },
                               epoch)
        writer.add_scalars("losses", {'train':
                            train_loss,
                            'val':
                            val_loss,
                            },
                            epoch)
        for test_range in cfg['test_ranges']:
            writer.add_scalars("contact_metrics", {f'timing_error_{test_range}':
                                              best_first_contact_timing[test_range],
                                          f'timing_range_{test_range}':
                                              best_first_contact_range[test_range],
                                          f'fail_rate_{test_range}':
                                              best_first_contact_fail[test_range],
                                          f'threshold_{test_range}':
                                              best_first_contact_thresh[test_range]
                                          },
                               epoch)


        if cfg['profile']:
            yappi.get_func_stats().print_all()
            yappi.get_thread_stats().print_all()
        if epoch % cfg['image_frequency'] == 0:
            with sns.axes_style("white"):
                max_times_ahead = max(cfg['times_ahead'])
                fig = plt.figure()
                mask = np.zeros(target_dict['contacts']['right'][0, -3 * max_times_ahead:, :].cpu().shape)
                mask[target_dict['contacts']['right'][0, -3 * max_times_ahead:, :].cpu() == 1] = True
                ax = sns.heatmap(predicted_dict['contacts']['right'][0, -3*max_times_ahead:, :].cpu(), cbar=True, vmin=None, vmax=None, mask=mask, xticklabels = cfg['times_ahead'])
                fig.add_axes(ax)
                writer.add_figure(f'contact_predictions_zoom', fig, global_step=epoch)

                fig = plt.figure()
                mask = np.zeros(target_dict['contacts']['right'][0, 0:, :].cpu().shape)
                mask[target_dict['contacts']['right'][0, 0:, :].cpu() == 1] = True
                ax = sns.heatmap(predicted_dict['contacts']['right'][0, 0:, :].cpu(), cbar=True,
                                 vmin=None, vmax=None, mask=mask, xticklabels = cfg['times_ahead'])
                fig.add_axes(ax)
                writer.add_figure(f'contact_predictions_full', fig, global_step=epoch)
                print(np.array2string(predicted_dict['contacts']['right'][0, -3*max_times_ahead:, :].cpu().numpy().flatten(), separator=','))
                print(np.array2string(target_dict['contacts']['right'][0, -3*max_times_ahead:, :].cpu().numpy().flatten(), separator=','))


        if epoch == 0:
            fig = plt.figure()
            ax = sns.heatmap(target_dict['contacts']['right'][0, -3*max_times_ahead:, :].cpu(), cbar=True, vmin=None, vmax=None, xticklabels = cfg['times_ahead'])
            fig.add_axes(ax)
            writer.add_figure(f'contact_target', fig, global_step=epoch)


        if cfg['epoch_draw'] and (epoch % cfg['epoch_draw_checkmarks'] == 0 or epoch == cfg['epochs']-1):
            if epoch == 0 and not cfg['draw_first_epoch']:
                pass
            else:
                save_names, save_frames = writeImage(hand_vis, val_dataset, meta_idx, in_dict, predicted_dict, target_dict, cfg)
                for i in range(len(save_names)):
                    img = Image.open(save_names[i])
                    writer.add_image(f'epoch_{epoch}', np.array(img), dataformats="HWC", global_step=save_frames[i])
    writer.close()

def writeImage(hand_vis, val_dataset, meta_idx, input_dict, predicted_dict, target_dict, cfg):
    with torch.no_grad():
        show_joints = cfg['data_type'] == 'joints'
        show_vertices = cfg['data_type'] == 'vertices'
        num_frames = predicted_dict['hand_points'][cfg['hands'][0]].shape[1]
        
        #make it "true" for all time after initial contact
        predicted_contacts = {}
        target_contacts = {}
        for hand in cfg['hands']:
            predicted_contacts[hand] = predicted_dict['contacts'][hand][0,:,-1] > .5
            for i in range(1, len(predicted_contacts[hand])):
                predicted_contacts[hand][i] = predicted_contacts[hand][i] or predicted_contacts[hand][i - 1]
            target_contacts[hand] = target_dict['contacts'][hand][0,:,-1] == 1
            for i in range(1, len(target_contacts[hand])):
                target_contacts[hand][i] = target_contacts[hand][i] or target_contacts[hand][i - 1]

        meta_file = val_dataset.meta_files[meta_idx]
        meta_data = np.load(meta_file, allow_pickle=True).item()
        seq = GRABInstance.metaToSeq(meta_file, cfg)
        data_holder = GRABInstance(cfg, sequence=seq)
        data_holder.loadObjBaseMesh(reduced='full')
        faces = data_holder.loadHandFaces(cfg['hands'])
        data_holder.generateObjFaces(reduced = 'full')
        faces['obj'] = data_holder.obj_faces['full']
        data_holder.generateObjVerts(reduced = 'full')

        input_verts = {}
        target_verts = {}
        predicted_verts = {}
        input_joints = {}
        target_joints = {}
        predicted_joints = {}
        input_points = {hand: data_holder.forwardTransform(input_dict['hand_points'][hand][0].cpu(),
                                                          input_dict['hand_points_range'][hand]) for hand in
                       input_dict['hand_points'].keys()}
        target_points = {hand: data_holder.forwardTransform(target_dict['hand_points'][hand][0].cpu(),
                                                           target_dict['hand_points_range'][hand]) for hand in
                        target_dict['hand_points'].keys()}
        predicted_points = {hand: data_holder.forwardTransform(predicted_dict['hand_points'][hand][0].cpu(),
                                                              target_dict['hand_points_range'][hand]) for hand in
                           predicted_dict['hand_points'].keys()}

        if cfg['data_type'] == 'vertices':
            input_verts = input_points
            target_verts = target_points
            predicted_verts = predicted_points
        if cfg['data_type'] == 'joints':
            input_joints = input_points
            target_joints = target_points
            predicted_joints = predicted_points

        input_verts['obj'] = data_holder.forwardTransform(data_holder.obj_verts['full'][input_dict['hand_points_range']['right'][0]:input_dict['hand_points_range']['right'][1], :], input_dict['hand_points_range']['right'])
        target_verts['obj'] = data_holder.forwardTransform(data_holder.obj_verts['full'][target_dict['hand_points_range']['right'][0]:target_dict['hand_points_range']['right'][1], :],
                                                           target_dict['hand_points_range']['right'])
        input_dict = {'verts': input_verts, 'joints': input_joints, 'contacts': None, 'faces': faces}
        predicted_dict = {'verts': predicted_verts, 'joints': target_joints, 'contacts': predicted_contacts, 'faces': faces}
        target_dict = {'verts': target_verts, 'joints': predicted_joints, 'contacts': target_contacts, 'faces': faces}
        save_names = hand_vis.saveSequence(f'{cfg["exp_dir_base"]}/image_temp', seq, input_dict, predicted_dict, target_dict, num_frames, history = range(0,30,10), show_vertices = show_vertices, show_joints = show_joints)
        return save_names
