import multiprocessing
import os
import pathlib
import shutil
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
import yappi
from joblib import Parallel, delayed
from torch.utils.data import Dataset

from predictive_hands.data_loading.GRABInstance import GRABInstance
from predictive_hands.utilities import utils


class DataContainer(Dataset):

    def __init__(self, cfg, filter_conditions, device_type, meta_files_in = None, randomized_start = False):
        self.filter_conditions = filter_conditions
        self.cfg = cfg
        if meta_files_in is None:
            self.meta_files = self.getFilteredFiles()
        else:
            self.meta_files = meta_files_in
        self.device_type = device_type
        self.randomized_start = randomized_start
    
    def __len__(self):
        return len(self.meta_files)
    
    def __getitem__(self, idx):
        cur_file = self.meta_files[idx]
        in_dict, target_dict, meta_data = self.generateInputTarget(cur_file)
        return in_dict, target_dict, idx, meta_data

    def getFilteredFiles(self):
        subj_filter = self.filter_conditions['subj_id']
        intent_filter = self.filter_conditions['intent']
        obj_filter = self.filter_conditions['obj_name']
        subj_filter = utils.subject_names if subj_filter is None else subj_filter
        intent_filter = [''] if intent_filter is None else intent_filter
        obj_filter = [''] if obj_filter is None else obj_filter
        meta_files = []
        for subj in subj_filter:
            for intent in intent_filter:
                for obj in obj_filter:
                    for hand in self.cfg['hands']:
                        meta_files.extend(Path(self.cfg['data_path']).joinpath(subj).glob(f'meta_data/{obj}*_{intent}*.npy'))
        return meta_files
    
    def generateInputTarget(self, meta_file):
        times_ahead = self.cfg['times_ahead']
        max_times_ahead = max(times_ahead)
        meta_data = np.load(meta_file, allow_pickle=True).item()
        grab_instance = GRABInstance(self.cfg, self.device_type, meta_data=meta_data)
        contacts = grab_instance.loadContacts(self.cfg['hands'])
        first_correct_vectors = {}
        first_index_flats = {}
        for hand in self.cfg['hands']:
            contact_info = np.expand_dims(contacts[hand],axis=1)
            diff_vector = np.concatenate(([False], np.logical_xor(contact_info[1:,0], contact_info[:-1,0])), axis=0)
            first_only_vector = np.zeros(diff_vector.shape)
            first_index = np.where(diff_vector)
            if len(first_index[0]) > 0:
                first_index_flat = first_index[0][0]
                first_only_vector[first_index[0][0]] = True
            else:
                return 0, 0, 0
            first_index_flats[hand] = first_index_flat

            ones_array = np.zeros((self.cfg['first_correct_width']))
            ones_array[int(self.cfg['first_correct_width']/2)] = 1
            if self.cfg['smoothing'] == 'gaussian':
                conv_kernel = scipy.ndimage.filters.gaussian_filter(ones_array, self.cfg['first_correct_sigma'])
                first_correct_vectors[hand] = np.expand_dims(np.convolve(conv_kernel, first_only_vector, mode='same'),axis=1)
            else:
                first_correct_vectors[hand] = np.expand_dims(first_only_vector, axis=1)
            if np.max(first_correct_vectors[hand]) > 0:
                first_correct_vectors[hand] = first_correct_vectors[hand]/np.max(first_correct_vectors[hand])

        if self.cfg['data_type'] == 'vertices':
            dists = grab_instance.loadPointDists(self.cfg['hands'])
            hand_points = grab_instance.loadHandVerts(self.cfg['hands'])
        elif self.cfg['data_type'] == 'joints':
            dists = grab_instance.loadPointDistsJoints(self.cfg['hands'])
            hand_points = grab_instance.loadHandJoints(self.cfg['hands'])


        if not self.cfg['use_dists']:
            dists = {key: np.zeros((value.shape[0], value.shape[1], 0))for key, value in hand_points.items()}
        dists_range = {}
        hand_points_range = {}
        for hand in hand_points.keys():
            start_time = 0
            if self.randomized_start:
                max_time = max(1, first_index_flats[hand] - 2 * max_times_ahead)
                start_time = np.random.randint(0, max_time)
            end_time = np.shape(hand_points[hand])[0]
            if self.cfg['until_contact']:
                end_time = 2*max_times_ahead + first_index_flats[hand] + 1

            dists_range[hand] = [0, end_time]
            hand_points_range[hand] = [0, end_time]
            dists[hand] = dists[hand][start_time:end_time, :, :]
            hand_points[hand] = hand_points[hand][start_time:end_time, :, :]
            first_correct_vectors[hand] = first_correct_vectors[hand][start_time:end_time, :]

        if times_ahead == [0]:
            in_dict = {'dists': {key: value[:, :, :] for key, value in dists.items()},
                       'hand_points': {key: value[:, :, :] for key, value in hand_points.items()},
                       'dists_range': dists_range,
                       'hand_points_range': hand_points_range}
        else:
            in_dict = {'dists': {key: value[0:-max_times_ahead, :, :] for key, value in dists.items()},
                       'hand_points': {key: value[0:-max_times_ahead, :, :] for key, value in hand_points.items()},
                       'dists_range': {key: [value[0], value[1] - max_times_ahead] for key, value in dists_range.items()},
                       'hand_points_range': {key: [value[0], value[1] - max_times_ahead] for key, value in hand_points_range.items()}}

        contact_targets = {}
        for key, value in first_correct_vectors.items():
            cur_targets = np.zeros((value.shape[0]-max_times_ahead, len(times_ahead)))
            for i in range(cur_targets.shape[0]):
                cur_targets[i] = value[[i + t for t in times_ahead],0]
            contact_targets[key] = cur_targets


        target_dict = {'dists': {key: value[max_times_ahead:,:,:] for key, value in dists.items()},
                       'hand_points': {key: value[max_times_ahead:,:,:] for key, value in
                                      hand_points.items()},
                       'contacts': contact_targets,
                       'dists_range': {key: [value[0]+max_times_ahead, value[1]] for key, value in dists_range.items()},
                       'hand_points_range': {key: [value[0]+max_times_ahead, value[1]] for key, value in
                                             hand_points_range.items()}
                       }

        for hand in target_dict['hand_points'].keys():
            if not self.cfg['points_out']:
                points_shape = target_dict['hand_points'][hand].shape
                target_dict['hand_points'][hand] = np.zeros((points_shape[0], 0, 0))
            if not self.cfg['contact_out']:
                contacts_shape = target_dict['contacts'][hand].shape
                target_dict['contacts'][hand] = np.zeros((contacts_shape[0], contacts_shape[1], 0))
        out_meta = {'obj_name': meta_data['obj_name'], 'subj_id': meta_data['subj_id'], 'intent': meta_data['intent']}
        return in_dict, target_dict, out_meta


    @classmethod
    def generateGRABDataFunc(self, cfg, seq):
        print(seq)
        data_holder = GRABInstance(cfg, cfg['device_type'], sequence=seq)
        data_holder.generateAndSaveMeta()
        data_holder.generateSmplx()
        data_holder.forwardSmplx()

        data_holder.loadObjBaseMesh(reduced='full')

        data_holder.generateObjVerts(reduced='full')

        data_holder.generateHandJoints()
        data_holder.generateHandVerts()

        data_holder.generateContacts()

        #data_holder.generateHandFaces()
        data_holder.generatePointDistsJoints()
        data_holder.generatePointDists()

        data_holder.saveContacts()
        data_holder.saveHandVerts()
        data_holder.savePointDists()
        #data_holder.saveHandFaces()
        data_holder.saveHandJoints()
        data_holder.savePointDistsJoints()

    @classmethod
    def visGRABData(self, hand_vis, cfg, seq):
        with torch.no_grad():
            faces = {}
            verts = {}
            joints = {}
            point_dists = None
            hands = cfg['vis_hands']
            meta_file = GRABInstance.seqToMeta(seq, cfg)
            seq = GRABInstance.metaToSeq(meta_file, cfg)
            meta_data = np.load(meta_file, allow_pickle=True).item()
            data_holder = GRABInstance(cfg, cfg['device_type'], sequence=seq)
            contacts = data_holder.loadContacts(hands)
            verts = data_holder.loadHandVerts(hands)
            faces = data_holder.loadHandFaces(hands)
            data_holder.loadObjBaseMesh(reduced='full')
            data_holder.generateObjFaces(reduced='full')
            faces['obj'] = data_holder.obj_faces['full']
            data_holder.generateObjVerts(reduced='full')
            data_holder.obj_parms['global_orient'] = data_holder.obj_parms['global_orient'][2:, :]
            data_holder.obj_parms['transl'] = data_holder.obj_parms['transl'][2:, :]
            verts['obj'] = data_holder.obj_verts['full'][2:, :]
            joints = data_holder.loadHandJoints(hands)
            point_dists = data_holder.loadPointDists(hands)
            verts = {key: data_holder.forwardTransform(verts[key]) for key in verts.keys()}
            joints = {key: data_holder.forwardTransform(joints[key]) for key in joints.keys()}

            num_frames = verts['obj'].shape[0]
            train = {'verts': verts, 'joints': joints, 'contacts': contacts, 'faces': faces, 'sdfs': point_dists}
            test = {'verts': verts, 'joints': joints, 'contacts': contacts, 'faces': faces, 'sdfs': point_dists}
            target = {'verts': verts, 'joints': joints, 'contacts': contacts, 'faces': faces, 'sdfs': point_dists}
            image_temp_path = cfg['image_temp_path']
            hand_vis.saveSequence(f'{image_temp_path}/{seq.stem}', seq, train, test, target, history=[], num_frames = num_frames,
                                  show_vertices=True, show_joints=False, show_sdf=False)


    @classmethod
    def generateGRABData(self,cfg):
        num_cores = int(multiprocessing.cpu_count())
        print(f'num cores: {num_cores}')
        if cfg['visualize_generation']:
            from predictive_hands.utilities.visualize_data import VisualizeHands
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            hand_vis = VisualizeHands(cfg, offscreen=True, obj_view=False)
            shutil.rmtree(Path(cfg['image_temp_path']), ignore_errors=True)
        if cfg['parallel_generation']:
            if cfg['regenerate_data']:
                all_seqs = Path(cfg['grab_path']).glob('*/*npz')
                Parallel(n_jobs=num_cores)(delayed(DataContainer.generateGRABDataFunc)(cfg,seq=seq) for seq in all_seqs)

        else:
            all_seqs = Path(cfg['grab_path']).glob('*/*npz')
            for seq in all_seqs:
                print(seq)
                if cfg['regenerate_data']:
                    DataContainer.generateGRABDataFunc(cfg, seq)
                if cfg['visualize_generation']:
                    DataContainer.visGRABData(hand_vis, cfg, seq=seq)

    @classmethod
    def generateBaseLineFunc(self, cfg, seq):
        print(seq)
        data_holder = GRABInstance(cfg, cfg['device_type'], sequence=seq)
        if data_holder.obj_name not in cfg['obj_names']:
            return None, None
        data_holder.generateAndSaveMeta()
        data_holder.generateSmplx()
        data_holder.forwardSmplx(baseline_window=cfg['baseline_window'])
        data_holder.loadObjBaseMesh(reduced='full')
        data_holder.generateObjVerts(reduced='full')
        data_holder.obj_parms['global_orient'] = data_holder.obj_parms['global_orient'][2:, :]
        data_holder.obj_parms['transl'] = data_holder.obj_parms['transl'][2:, :]
        data_holder.generateHandVerts()
        data_holder.generateContacts()
        data_holder.generatePointDists()
        data_holder.generateHandFaces()
        if cfg['visualize_generation']:
            data_holder.saveHandVerts()
            data_holder.saveContacts()
            data_holder.saveHandFaces()

        truth_contacts = data_holder.contacts['right'][2:]
        diff_vector_truth = np.concatenate(([False], np.logical_xor(truth_contacts[1:], truth_contacts[:-1])), axis=0)
        first_index_truth = np.where(diff_vector_truth)
        if len(first_index_truth[0]) > 0:
            first_index_truth_flat = first_index_truth[0][0]
        else:
            return None, None

        projected_contact = data_holder.point_dists['right'] <= .0045
        projected_contact = projected_contact.squeeze().any(axis=1)
        diff_vector_baseline = np.concatenate(([False], np.logical_xor(projected_contact[1:], projected_contact[:-1])), axis=0)
        first_index_baseline = np.where(diff_vector_baseline)
        if len(first_index_baseline[0]) > 0:
            first_index_baseline_flat = first_index_baseline[0][0]
        else:
            return -1, data_holder.obj_name

        return np.abs(first_index_truth_flat - first_index_baseline_flat), data_holder.obj_name

    @classmethod
    def generateBaseline(self, cfg):
        if cfg['visualize_generation']:
            from predictive_hands.utilities.visualize_data import VisualizeHands
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            hand_vis = VisualizeHands(cfg, offscreen=True, obj_view=False)
            shutil.rmtree(Path(cfg['image_temp_path']), ignore_errors=True)
        num_cores = int(multiprocessing.cpu_count() / 2)
        all_seqs = Path(cfg['grab_path']).glob('*/*npz')
        if cfg['parallel_generation']:
            all_errors = Parallel(n_jobs=num_cores)(delayed(DataContainer.generateBaseLineFunc)(cfg, seq) for seq in all_seqs)
            obj_errors = {}
            for error in all_errors:
                if error[1] is not None:
                    if error[1] not in obj_errors.keys():
                        obj_errors[error[1]] = []
                    obj_errors[error[1]].append(error[0])

        else:
            errors = []
            for seq in list(all_seqs)[0:50]:
                errors.append(DataContainer.generateBaseLineFunc(cfg, seq))
                print(errors)
            obj_errors = {}
            for error in errors:
                if error[1] is not None:
                    if error[1] not in obj_errors.keys():
                        obj_errors[error[1]] = []
                    obj_errors[error[1]].append(error[0])






