import pickle
from pathlib import Path

import numpy as np
import smplx
import smplx.joint_names
import tensorflow_graphics.math.interpolation.slerp as slerp
import torch
from grab.tools.meshviewer import Mesh
from grab.tools.objectmodel import ObjectModel
from grab.tools.utils import params2torch
from grab.tools.utils import parse_npz
from scipy.spatial import KDTree
from smplx.lbs import batch_rodrigues

from predictive_hands.utilities import utils


#Basically just import from grab and put in dicts
class GRABInstance:

    def __init__(self, cfg, device_type, meta_data = None, sequence=None):

        self.device_type = device_type
        
        # Use these when processing
        self.hand_joints = {}
        self.contacts = {}
        self.hand_verts = {}
        self.hand_faces = {}
        self.obj_verts = {}
        self.obj_faces = {}
        self.point_dists = {}
        self.point_dists_joints = {}
        self.hand_edges = {}
        self.offset = None

        with open(cfg['mano_smplx_file'], 'rb') as f:
            self.idxs_data = pickle.load(f)

        if meta_data is not None:
            self.obj_name = meta_data['obj_name']
            self.subj_id = meta_data['subj_id']
            self.n_comps = meta_data['n_comps']
            self.gender = meta_data['gender']
            self.intent = meta_data['intent']
            self.frame_nums = meta_data['frame_nums']
            self.file_basename = meta_data['file_basename']
        elif sequence is not None:
            self.seq_data = parse_npz(sequence)
            T = self.seq_data.n_frames
            self.obj_name = self.seq_data.obj_name
            self.subj_id = self.seq_data.sbj_id
            self.n_comps = self.seq_data.n_comps
            self.gender = self.seq_data.gender
            self.intent = self.seq_data.motion_intent
            self.frame_nums = np.arange(T)
            self.sbj_m = None
            self.file_basename = Path(Path(sequence).name).with_suffix('')
        else:
            raise Exception('Need meta_file or sequence')

        self.cfg = cfg

        
        self.base_dir = Path(cfg['data_path']).joinpath(self.subj_id)
        self.verts_path = str(Path(self.base_dir).joinpath('vertices'))
        self.smplx_path = str(Path(self.base_dir).joinpath('smplxs'))
        self.hand_joints_path = str(Path(self.base_dir).joinpath('joints'))
        self.contacts_path = str(Path(self.base_dir).joinpath('contacts'))
        self.point_dists_path = str(Path(self.base_dir).joinpath('point_dists'))
        self.point_dists_joints_path = str(Path(self.base_dir).joinpath('point_dists_joints'))
        self.meta_path = str(Path(self.base_dir).joinpath('meta_data'))
        self.faces_path = str(Path(cfg['data_path']).joinpath('faces'))
        self.edges_path = str(Path(cfg['data_path']).joinpath('edges'))
        self.obj_cache_path = str(Path(cfg['data_path']).joinpath('reduced_obj_cache'))

        Path(self.verts_path).mkdir(parents=True, exist_ok=True)
        Path(self.hand_joints_path).mkdir(parents=True, exist_ok=True)
        Path(self.contacts_path).mkdir(parents=True, exist_ok=True)
        Path(self.faces_path).mkdir(parents=True, exist_ok=True)
        Path(self.meta_path).mkdir(parents=True, exist_ok=True)
        Path(self.smplx_path).mkdir(parents=True, exist_ok=True)
        Path(self.obj_cache_path).mkdir(parents=True, exist_ok=True)
        Path(self.point_dists_path).mkdir(parents=True, exist_ok=True)
        Path(self.point_dists_joints_path).mkdir(parents=True, exist_ok=True)
        Path(self.edges_path).mkdir(parents=True, exist_ok=True)

    # We want the object to be at origin, so apply reverse of object
    # transform to the hand
    def reverseTransform(self, vertices):
        orient = self.obj_parms['global_orient']
        transl = self.obj_parms['transl']
        rot_mats = batch_rodrigues(-orient.view(-1, 3)).view([vertices.shape[0], 3, 3])
        out_verts = torch.matmul(vertices - transl.unsqueeze(dim=1), rot_mats)
        return out_verts

    def forwardTransform(self, vertices, min_max=(None,None)):
        orient = self.obj_parms['global_orient'][min_max[0]:min_max[1]]
        transl = self.obj_parms['transl'][min_max[0]:min_max[1]]
        rot_mats = batch_rodrigues(orient.view(-1, 3)).view([vertices.shape[0], 3, 3])
        out_verts = torch.matmul(vertices, rot_mats) + transl.unsqueeze(dim=1)
        return out_verts

    def generateSmplx(self):
        sbj_mesh = Path(self.cfg['grab_path']).joinpath('..', self.seq_data.body.vtemp)
        sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)
        if self.cfg['baseline_window'] is not None:
            self.seq_data.n_frames -= 1

        self.sbj_m = smplx.create(model_path=self.cfg['model_path'],
                            model_type='smplx',
                            gender=self.seq_data.gender,
                            num_pca_comps=self.seq_data.n_comps,
                            v_template = sbj_vtemp,
                            batch_size=self.seq_data.n_frames)

    def forwardSmplx(self, seq=None, baseline_window=None):
        if seq is not None:
            self.seq_data = parse_npz(seq)
            self.obj_parms = params2torch(self.seq_data.object.params)
        sbj_params = params2torch(self.seq_data.body.params)
        sbj_params['left_hand_pose'] = torch.einsum(
            'bi,ij->bj', [sbj_params['left_hand_pose'], self.sbj_m.left_hand_components])
        sbj_params['right_hand_pose'] = torch.einsum(
            'bi,ij->bj', [sbj_params['right_hand_pose'], self.sbj_m.right_hand_components])
        self.sbj_m.use_pca = False
        if baseline_window is not None:
            left, right = self.projectBaseline(sbj_params, baseline_window)
            sbj_params['left_hand_pose'] = left
            sbj_params['right_hand_pose'] = right
            sbj_params['global_orient'] = sbj_params['global_orient'][1:]
            sbj_params['body_pose'] = sbj_params['body_pose'][1:]
            sbj_params['jaw_pose'] = sbj_params['jaw_pose'][1:]
            sbj_params['leye_pose'] = sbj_params['leye_pose'][1:]
            sbj_params['reye_pose'] = sbj_params['reye_pose'][1:]
            sbj_params['transl'] = sbj_params['transl'][1:]
            sbj_params['expression'] = sbj_params['expression'][1:]
        self.output_sbj = self.sbj_m(**sbj_params)
        if baseline_window is not None:
            self.translBaseline(sbj_params, baseline_window)
        if self.cfg['transl_noise'] is not None:
            self.addTransNoise(self.cfg['transl_noise'])
        if self.cfg['joint_noise'] is not None:
            self.addJointNoise(self.cfg['joint_noise'])

    def projectBaseline(self, sbj_params, baseline_window):
        duration = sbj_params['left_hand_pose'].shape[0]
        left_reshape = sbj_params['left_hand_pose'].reshape((duration, -1, 3))
        left_new_pos = slerp.interpolate(left_reshape[0:-1, :, :], left_reshape[1:, :, :], baseline_window+1, method=slerp.InterpolationType.VECTOR)

        duration = sbj_params['right_hand_pose'].shape[0]
        right_reshape = sbj_params['right_hand_pose'].reshape((duration, -1, 3))
        right_new_pos = slerp.interpolate(right_reshape[0:-1, :, :], right_reshape[1:, :, :], baseline_window+1, method=slerp.InterpolationType.VECTOR)

        return torch.Tensor(left_new_pos.numpy()).reshape((duration-1, -1)), torch.Tensor(right_new_pos.numpy()).reshape((duration-1, -1))

    def translBaseline(self, sbj_params, baseline_window):
        joint_names = smplx.joint_names.JOINT_NAMES
        lw_wrist = self.output_sbj.joints[:, [joint_names.index('left_wrist')], :]
        left_vel = lw_wrist[1:, :, :] - lw_wrist[0:-1, :, :]

        rw_wrist = self.output_sbj.joints[:, [joint_names.index('right_wrist')], :]
        right_vel = rw_wrist[1:, :, :] - rw_wrist[0:-1, :, :]

        self.output_sbj.joints[1:, [joint_names.index(name) for name in utils.RHAND_JOINT_NAMES], :] += right_vel[0:, :, :] * baseline_window

        self.output_sbj.joints[1:, [joint_names.index(name) for name in utils.LHAND_JOINT_NAMES], :] += left_vel[0:, :,
                                                :] * baseline_window

        self.output_sbj.vertices[1:,self.idxs_data['right_hand'],:] += right_vel[0:, :, :] * baseline_window
        self.output_sbj.vertices[1:,self.idxs_data['left_hand'],:] += left_vel[0:, :,
                                                :] * baseline_window
        self.output_sbj.joints = self.output_sbj.joints[1:]
        self.output_sbj.vertices = self.output_sbj.vertices[1:]


    def addTransNoise(self, transl_noise, random_angle = True):
        noise_vector = torch.normal(torch.zeros((1,1,3)))
        noise_magnitude = torch.normal(torch.zeros((1)), transl_noise)
        noise_vector = noise_magnitude*noise_vector/(torch.linalg.norm(noise_vector))
        self.output_sbj.joints += noise_vector
        self.output_sbj.vertices += noise_vector

    def addJointNoise(self, joint_noise):
        self.output_sbj.joints += torch.normal(torch.zeros(self.output_sbj.joints.shape), joint_noise)
        self.output_sbj.vertices += torch.normal(torch.zeros(self.output_sbj.vertices.shape), joint_noise)

    def saveSmplx(self):
        smplx_path = Path(self.smplx_path).joinpath(str(self.file_basename)).with_suffix('.pkl')
        pickle.dump(self.sbj_m, open(smplx_path, 'wb'))
    
    def loadSmplx(self):
        smplx_path = Path(self.smplx_path).joinpath(str(self.file_basename)).with_suffix('.pkl')
        self.sbj_m = pickle.load(open(smplx_path, 'rb'))


    def loadObjBaseMesh(self, reduced = 'reduced'):
        if reduced=='reduced':
            obj_cache = Path(self.obj_cache_path).joinpath(self.obj_name).with_suffix('.pkl')
            self.reduced_obj_mesh = pickle.load(open(obj_cache, 'rb'))
        else:
            obj_mesh_path = Path(self.cfg['grab_path']).joinpath('..', self.seq_data.object.object_mesh)
            self.obj_mesh = Mesh(filename=obj_mesh_path)

    ###########################
    ## saves ###
    #############################
    def generateAndSaveMeta(self):
        meta_dict = {}
        meta_dict['obj_name'] = self.obj_name
        meta_dict['subj_id'] = self.subj_id
        meta_dict['n_comps'] = self.n_comps
        meta_dict['gender'] = self.gender
        meta_dict['intent'] = self.intent
        meta_dict['frame_nums'] = self.frame_nums
        meta_dict['file_basename'] = self.file_basename
        meta_dict['offset'] = self.offset

        out_name = Path(self.meta_path).joinpath(str(self.file_basename)).with_suffix('.npy')
        np.save(out_name, meta_dict)
        return out_name

    def savePointDists(self):        
        out_name_left_point_dists = Path(self.point_dists_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
        out_name_right_point_dists = Path(self.point_dists_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
        np.save(out_name_left_point_dists, self.point_dists['left'])
        np.save(out_name_right_point_dists, self.point_dists['right'])

    def savePointDistsJoints(self):
        out_name_left_point_dists_joints = Path(self.point_dists_joints_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
        out_name_right_point_dists_joints = Path(self.point_dists_joints_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
        np.save(out_name_left_point_dists_joints, self.point_dists_joints['left'])
        np.save(out_name_right_point_dists_joints, self.point_dists_joints['right'])

    def saveHandVerts(self):        
        out_name_left_vert = Path(self.verts_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
        out_name_right_vert = Path(self.verts_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
        torch.save(self.hand_verts['left'], out_name_left_vert)
        torch.save(self.hand_verts['right'], out_name_right_vert)

    def saveHandFaces(self):
        out_name_left_face = Path(self.faces_path).joinpath(str(self.subj_id) + "_left").with_suffix('.npy')
        out_name_right_face = Path(self.faces_path).joinpath(str(self.subj_id) + "_right").with_suffix('.npy')
        if not Path(out_name_left_face).exists():
            np.save(out_name_left_face, self.hand_faces['left'])
        if not Path(out_name_right_face).exists():
            np.save(out_name_right_face, self.hand_faces['right'])

    def saveHandEdges(self):
        out_name_left_edge = Path(self.edges_path).joinpath(str(self.subj_id) + "_left").with_suffix('.npy')
        out_name_right_edge = Path(self.edges_path).joinpath(str(self.subj_id) + "_right").with_suffix('.npy')
        if not Path(out_name_left_edge).exists():
            np.save(out_name_left_edge, self.hand_edges['left'])
        if not Path(out_name_right_edge).exists():
            np.save(out_name_right_edge, self.hand_edges['right'])

    def saveObjMeshReduced(self):
        obj_faces = self.reduced_obj_mesh.faces
        out_name_vert = Path(self.verts_path).joinpath(str(self.file_basename) + "_obj_reduced").with_suffix('.npy')
        out_name_face = Path(self.faces_path).joinpath(str(self.obj_name) + "_obj_reduced").with_suffix('.npy')
        np.save(out_name_vert, self.obj_verts['reduced'])
        if not Path(out_name_face).exists():
            object_faces = obj_faces
            np.save(out_name_face, object_faces)
             

    def saveHandJoints(self):
        out_name_left = Path(self.hand_joints_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
        out_name_right = Path(self.hand_joints_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
        torch.save(self.hand_joints['left'], out_name_left)
        torch.save(self.hand_joints['right'], out_name_right)
    
    def saveContacts(self):
        out_name_left = Path(self.contacts_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
        out_name_right = Path(self.contacts_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
        np.save(out_name_left, self.contacts['left'])
        np.save(out_name_right, self.contacts['right'])

    ###########################
    ## Data loaders##
    ###################
    def loadPointDists(self, hands):
        if 'left' in hands:
            out_name_left = Path(self.point_dists_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
            self.point_dists['left'] = np.load(out_name_left)
        if 'right' in hands:
            out_name_right = Path(self.point_dists_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
            self.point_dists['right'] = np.load(out_name_right)
        return self.point_dists

    def loadPointDistsJoints(self, hands):
        if 'left' in hands:
            out_name_left = Path(self.point_dists_joints_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
            self.point_dists_joints['left'] = np.load(out_name_left)
        if 'right' in hands:
            out_name_right = Path(self.point_dists_joints_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
            self.point_dists_joints['right'] = np.load(out_name_right)
        return self.point_dists_joints
    
    def loadHandVerts(self, hands):
        if 'left' in hands:
            out_name_left = Path(self.verts_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
            self.hand_verts['left'] = torch.load(out_name_left)
        if 'right' in hands:
            out_name_right = Path(self.verts_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
            self.hand_verts['right'] = torch.load(out_name_right)
        return {key: self.hand_verts[key].detach() for key in hands}


    def loadHandJoints(self, hands):
        if 'left' in hands:
            out_name_left = Path(self.hand_joints_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
            self.hand_joints['left'] = torch.load(out_name_left)
        if 'right' in hands:
            out_name_right = Path(self.hand_joints_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
            self.hand_joints['right'] = torch.load(out_name_right)
        return {key: self.hand_joints[key].detach() for key in hands}

    def loadContacts(self, hands):
        if 'left' in hands:
            out_name_left = Path(self.contacts_path).joinpath(str(self.file_basename) + "_left").with_suffix('.npy')
            lhand_contacts = np.load(out_name_left)
            self.contacts['left'] = lhand_contacts
        if 'right' in hands:
            out_name_right = Path(self.contacts_path).joinpath(str(self.file_basename) + "_right").with_suffix('.npy')
            rhand_contacts = np.load(out_name_right)
            self.contacts['right'] = rhand_contacts
        return self.contacts

    def loadHandFaces(self, hands):
        if 'left' in hands:
            out_name_left = Path(self.faces_path).joinpath(str(self.subj_id) + "_left").with_suffix('.npy')
            lhand_faces = np.load(out_name_left)
            self.hand_faces['left'] = lhand_faces
        if 'right' in hands:
            out_name_right = Path(self.faces_path).joinpath(str(self.subj_id) + "_right").with_suffix('.npy')
            rhand_faces = np.load(out_name_right)
            self.hand_faces['right'] = rhand_faces
        return self.hand_faces

    def loadHandEdges(self, hands):
        if 'left' in hands:
            out_name_left = Path(self.edges_path).joinpath(str(self.subj_id) + "_left").with_suffix('.npy')
            lhand_edges = np.load(out_name_left)
            self.hand_edges['left'] = lhand_edges
        if 'right' in hands:
            out_name_right = Path(self.edges_path).joinpath(str(self.subj_id) + "_right").with_suffix('.npy')
            rhand_edges = np.load(out_name_right)
            self.hand_edges['right'] = rhand_edges
        return self.hand_edges
    
    def loadObjFaces(self, reduced='reduced'):
        reduced_str = ''
        if reduced=='reduced':
            reduced_str = '_reduced'
        faces_name = Path(self.faces_path).joinpath(str(self.obj_name) + f"_obj{reduced_str}").with_suffix('.npy')
        self.obj_faces = np.load(faces_name)
        return self.obj_faces

    def loadObjVerts(self, reduced='reduced'):
        reduced_str = ''
        if reduced=='reduced':
            reduced_str = '_reduced'
        out_name_vert = Path(self.verts_path).joinpath(str(self.file_basename) + f"_obj{reduced_str}").with_suffix('.npy')
        self.obj_verts = np.load(out_name_vert)
        return self.obj_verts


    ########################
    ## Generators##
    ########################

    def generatePointDists(self):
        r_shape = self.hand_verts['right'].shape
        self.point_dists['right'] = np.zeros((r_shape[0], r_shape[1]))
        l_shape = self.hand_verts['right'].shape
        self.point_dists['left'] = np.zeros((l_shape[0], l_shape[1]))
        h_verts_r = self.hand_verts['right'].detach().numpy()
        h_verts_l = self.hand_verts['left'].detach().numpy()
        o_verts = self.obj_verts['full'].detach().numpy()
        #Only works if mesh at origin!
        kd_tree = KDTree(o_verts[0, :, :], leafsize=500)
        for i in range(r_shape[0]):
            self.point_dists['right'][i,:] = kd_tree.query(h_verts_r[i, :, :])[0]
            self.point_dists['left'][i,:] = kd_tree.query(h_verts_l[i, :, :])[0]
        self.point_dists['right'] = np.expand_dims(self.point_dists['right'], axis=-1)
        self.point_dists['left'] = np.expand_dims(self.point_dists['left'], axis=-1)

    def generatePointDistsJoints(self):
        r_shape = self.hand_joints['right'].shape
        self.point_dists_joints['right'] = np.zeros((r_shape[0], r_shape[1]))
        l_shape = self.hand_joints['right'].shape
        self.point_dists_joints['left'] = np.zeros((l_shape[0], l_shape[1]))
        h_joints_r = self.hand_joints['right'].detach().numpy()
        h_joints_l = self.hand_joints['left'].detach().numpy()
        o_verts = self.obj_verts['full'].detach().numpy()
        #Only works if mesh at origin!
        kd_tree = KDTree(o_verts[0, :, :], leafsize=500)
        for i in range(r_shape[0]):
            self.point_dists_joints['right'][i,:] = kd_tree.query(h_joints_r[i, :, :])[0]
            self.point_dists_joints['left'][i,:] = kd_tree.query(h_joints_l[i, :, :])[0]
        self.point_dists_joints['right'] = np.expand_dims(self.point_dists_joints['right'], axis=-1)
        self.point_dists_joints['left'] = np.expand_dims(self.point_dists_joints['left'], axis=-1)


    def generateHandJoints(self):
        joint_names = smplx.joint_names.JOINT_NAMES
        rhand_joints = self.output_sbj.joints[:, [joint_names.index(name) for name in utils.RHAND_JOINT_NAMES], :]
        lhand_joints = self.output_sbj.joints[:, [joint_names.index(name) for name in utils.LHAND_JOINT_NAMES], :]
        smplx_vertex_ids = smplx.vertex_ids.vertex_ids['smplx']
        rhand_tips = self.output_sbj.vertices[:,[smplx_vertex_ids[name] for name in utils.RHAND_VERTEX_TIPS], :]
        lhand_tips = self.output_sbj.vertices[:,[smplx_vertex_ids[name] for name in utils.LHAND_VERTEX_TIPS], :]
        
        self.hand_joints['left'] = torch.cat((lhand_joints, lhand_tips), dim=1)
        self.hand_joints['right'] = torch.cat((rhand_joints, rhand_tips), dim=1)
        if self.cfg['obj_centered']:
            self.hand_joints['left'] = self.reverseTransform(self.hand_joints['left'])
            self.hand_joints['right'] = self.reverseTransform(self.hand_joints['right'])

    def generateContacts(self):
        self.contacts['right'] = (self.seq_data['contact']['body'][:,self.idxs_data['right_hand']]>0).any(axis=1)
        self.contacts['left'] = (self.seq_data['contact']['body'][:,self.idxs_data['left_hand']]>0).any(axis=1)
        
    def generateHandVerts(self):
        verts_sbj = self.output_sbj.vertices
        self.hand_verts['left'] = verts_sbj[:,self.idxs_data['left_hand'],:]
        self.hand_verts['right'] = verts_sbj[:,self.idxs_data['right_hand'],:]
        if self.cfg['obj_centered']:
            self.hand_verts['left'] = self.reverseTransform(self.hand_verts['left'])
            self.hand_verts['right'] = self.reverseTransform(self.hand_verts['right'])


    
    def generateObjVerts(self, reduced='reduced'):
        
        if reduced == 'reduced':
            obj_m = ObjectModel(v_template=self.reduced_obj_mesh.vertices,
                                batch_size=len(self.frame_nums))
            self.obj_parms = params2torch(self.seq_data.object.params)
            self.obj_verts['reduced'] = obj_m(**obj_parms).vertices
        else:
            obj_m = ObjectModel(v_template=self.obj_mesh.vertices,
                                batch_size=len(self.frame_nums))
            self.obj_parms = params2torch(self.seq_data.object.params)
            if self.cfg['obj_centered']:
                self.obj_verts['full'] = obj_m().vertices

            else:
                self.obj_verts['full'] = obj_m(**obj_parms).vertices

    def generateObjFaces(self, reduced='reduced'):
        if reduced=='reduced':
            self.obj_faces['reduced'] = self.reduced_obj_mesh.faces
        else:
            self.obj_faces['full'] = self.obj_mesh.faces
    
    def generateHandFaces(self):
        points_per_face = np.shape(self.sbj_m.faces)[1]
        lhand_faces = np.zeros((0,points_per_face))
        for i in range(np.shape(self.sbj_m.faces)[0]):
            valid_face = 1
            current_face = np.zeros((1,points_per_face))
            for j in range(points_per_face):
                new_index = np.where(self.idxs_data['left_hand'] == self.sbj_m.faces[i,j])[0]
                if new_index.size == 0:
                    valid_face = 0
                else:
                    current_face[0,j] = new_index[0]
            if valid_face:
                lhand_faces = np.concatenate((lhand_faces, current_face), axis=0)
            self.hand_faces['left'] = lhand_faces

        points_per_face = np.shape(self.sbj_m.faces)[1]
        rhand_faces = np.zeros((0,points_per_face))
        for i in range(np.shape(self.sbj_m.faces)[0]):
            valid_face = 1
            current_face = np.zeros((1,points_per_face))
            for j in range(points_per_face):
                new_index = np.where(self.idxs_data['right_hand'] == self.sbj_m.faces[i,j])[0]
                if new_index.size == 0:
                    valid_face = 0
                else:
                    current_face[0,j] = new_index[0]
            if valid_face:
                rhand_faces = np.concatenate((rhand_faces, current_face), axis=0)
        self.hand_faces['right'] = rhand_faces

    def generateHandEdges(self):
        for hand in ['left', 'right']:
            edges1 = self.hand_faces[hand][:, [0,1]]
            edges2 = self.hand_faces[hand][:, [1,2]]
            edges3 = self.hand_faces[hand][:, [0,2]]
            all_edges = np.concatenate((edges1, edges2, edges3), axis=0)
            #below line to guarantee digraph
            all_edges = np.concatenate((all_edges, np.flip(all_edges,1)), axis=0)
            all_edges = np.unique(all_edges, axis=0)
            self.hand_edges[hand] = all_edges

    @classmethod
    def seqToMeta(self, seq, cfg):
        return Path(cfg['data_path']).joinpath(seq.parts[-2], 'meta_data', seq.parts[-1]).with_suffix('.npy')
    
    @classmethod
    def metaToSeq(self, meta_file, cfg):
        return Path(cfg['grab_path']).joinpath(meta_file.parts[-3], meta_file.parts[-1]).with_suffix('.npz')