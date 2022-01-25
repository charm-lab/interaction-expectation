# Largely adapated directly from GRAB code
from pathlib import Path

import numpy as np
from grab.tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from grab.tools.objectmodel import ObjectModel
from grab.tools.utils import euler
from grab.tools.utils import params2torch
from grab.tools.utils import parse_npz
from grab.tools.utils import to_cpu


class VisualizeHands():

    def __init__(self, cfg, offscreen=False, obj_view = False):
        self.cfg = cfg

        self.mv = MeshViewer(offscreen=offscreen)
        self.obj_view = obj_view
        self.show_table = not self.obj_view

        if obj_view:
            # magic number that gives okay view
            new_pose = np.array([[-0.22886,-0.21549,0.94931,1.04691],
                        [0.96856,0.04729,0.24424,0.01016],
                        [-0.09752,0.97536,0.19789,0.33374],
                        [0.00000,0.00000,0.00000,1.00000]])

            self.mv.update_camera_pose(pose=new_pose)


    def getTableVerts(self,seq, num_frames):
        seq_data = parse_npz(seq)

        table_mesh_path = Path(self.cfg['grab_path']).joinpath('..', seq_data.table.table_mesh)
        table_mesh = Mesh(filename=table_mesh_path)
        table_vtemp = np.array(table_mesh.vertices)
        table_m = ObjectModel(v_template=table_vtemp,
                            batch_size=num_frames)
        table_parms = params2torch(seq_data.table.params)
        table_parms['global_orient'] = table_parms['global_orient'][0:num_frames]
        table_parms['transl'] = table_parms['transl'][0:num_frames]
        verts_table = to_cpu(table_m(**table_parms).vertices)
        return verts_table, table_mesh

    def liveSequence(self, seq, input_dict, predicted_dict, target_dict, num_frames, history = [0], show_vertices = True, show_joints = True, show_sdf = True):
        verts_table, table_mesh = self.getTableVerts(seq, num_frames)

        skip_frame = self.cfg['frame_render']
        if len(history) == 0:
            max_history = 0
        else:
            max_history = max(history)
        for frame in range(max_history,num_frames, skip_frame):
            self.setFrame(frame, verts_table, table_mesh, input_dict, predicted_dict, target_dict, history = history, show_vertices = show_vertices, show_joints = show_joints, show_sdf = show_sdf)

    def saveSequence(self, save_folder, seq, input_dict, predicted_dict, target_dict, num_frames, history = [0], show_vertices = True, show_joints = True, show_sdf = False):

        save_dir = Path(save_folder)
        save_dir.mkdir(parents=True, exist_ok=True)
        verts_table, table_mesh = self.getTableVerts(seq, num_frames)

        skip_frame = self.cfg['frame_render']
        save_names = []
        save_frames = []
        if len(history) == 0:
            max_history = 0
        else:
            max_history = max(history)
        for frame in range(max_history,num_frames, skip_frame):
            self.setFrame(frame, verts_table, table_mesh, input_dict, predicted_dict, target_dict, history = history, show_vertices = show_vertices, show_joints = show_joints, show_sdf = show_sdf)
            new_pose = np.eye(4)
            new_pose[:3,:3] = euler([90,-23.7,0], 'xzx')
            new_pose[:3, 3] = np.array([-.14, -1., 1.23])
            self.mv.update_camera_pose(pose=new_pose)
            save_name = save_dir.joinpath(f"imagine_{frame}").with_suffix('.png')
            self.mv.save_snapshot(save_name)
            save_names.append(save_name)
            save_frames.append(frame)
        return save_names, save_frames

    def setFrame(self, frame, verts_table, table_mesh, input_dict, predicted_dict, target_dict, history = [0], show_vertices = False, show_joints = False, show_sdf = False):
        out_meshes = []
        # view history
        alpha = [1]
        for hist_frame in history:
            alpha = [1/(hist_frame/max(history) + 1)]
            hist_frame = frame - hist_frame
            obj_color = colors['purple']
            o_mesh = Mesh(vertices=input_dict['verts']['obj'][hist_frame], faces=input_dict['faces']['obj'], vc=obj_color+alpha)
            out_meshes.extend([o_mesh])
            s_meshes = []
            for hand in self.cfg['vis_hands']:
                hand_color = colors['purple']
                if show_vertices and hand in input_dict['verts'].keys() and input_dict['verts'][hand] is not None:
                    s_meshes.append(Mesh(vertices=input_dict['verts'][hand][hist_frame], faces=input_dict['faces'][hand], wireframe = True,  vc=hand_color + alpha, smooth=False))
                if show_joints and hand in input_dict['joints'].keys() and input_dict['joints'][hand] is not None:
                    s_meshes.append(points2sphere(input_dict['joints'][hand][hist_frame], radius=.005, vc=hand_color + alpha))
            out_meshes.extend(s_meshes)

        # view target
        obj_color = colors['orange']
        for hand in self.cfg['vis_hands']:
            if predicted_dict['contacts'][hand][frame]:
                obj_color = colors['white'] #! if the predicted is touching, also turn object white! This isn't the target for this color!
        o_mesh = Mesh(vertices=target_dict['verts']['obj'][frame], faces=target_dict['faces']['obj'], vc=obj_color+alpha)
        out_meshes.extend([o_mesh])
        s_meshes = []
        for hand in self.cfg['vis_hands']:
            hand_color = colors['orange']
            if target_dict['contacts'][hand][frame]:
                    hand_color = colors['red']
            if show_vertices and hand in target_dict['verts'].keys() and target_dict['verts'][hand] is not None:
                s_meshes.append(Mesh(vertices=target_dict['verts'][hand][frame], faces=target_dict['faces'][hand], wireframe=True, vc=hand_color, smooth=False))
            if show_joints and hand in target_dict['joints'].keys() and target_dict['joints'][hand] is not None:
                s_meshes.append(points2sphere(target_dict['joints'][hand][frame], radius=.005, vc=hand_color))
            if show_sdf and hand in target_dict['sdfs'].keys() and target_dict['sdfs'][hand] is not None:
                sdf_hand = np.copy(target_dict['verts'][hand][frame])
                sdf_hand[:, 2] += target_dict['sdfs'][hand][frame][:,0]
                s_meshes.append(Mesh(vertices=sdf_hand, faces=target_dict['faces'][hand], vc=colors['pink'],
                         smooth=False))

        out_meshes.extend(s_meshes)


        # view predicted
        s_meshes = []
        for hand in self.cfg['vis_hands']:
            hand_color = colors['blue']
            if predicted_dict['contacts'][hand][frame]:
                    hand_color = colors['white']
            if show_vertices and hand in predicted_dict['verts'].keys() and predicted_dict['verts'][hand] is not None:
                s_meshes.append(Mesh(vertices=predicted_dict['verts'][hand][frame], faces=predicted_dict['faces'][hand], vc=hand_color, smooth=False))
            if show_joints and hand in predicted_dict['joints'].keys() and predicted_dict['joints'][hand] is not None:
                s_meshes.append(points2sphere(predicted_dict['joints'][hand][frame], radius=.005, vc=hand_color))
        out_meshes.extend(s_meshes)
        if self.show_table:
            t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])
            out_meshes.extend([t_mesh])
        self.mv.set_static_meshes(out_meshes)
