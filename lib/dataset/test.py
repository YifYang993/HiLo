
class TestDatasetv1():
    def __init__(self, cfg, device, args=None):

        # random.seed(1993)
        if args:
            self.args=args
        self.image_dir = cfg['image_dir']
        self.seg_dir = cfg['seg_dir']
        self.hps_type = cfg['hps_type']
        self.smpl_type = 'smpl' if cfg['hps_type'] != 'pixie' else 'smplx'
        self.smpl_gender = 'neutral'
        self.colab = cfg['colab']

        self.device = device

        keep_lst = sorted(glob.glob(f"{self.image_dir}/*"))
        img_fmts = ['jpg', 'png', 'jpeg', "JPG", 'bmp', 'exr']
        keep_lst = [item for item in keep_lst if item.split(".")[-1] in img_fmts]

        self.subject_list = sorted([item for item in keep_lst if item.split(".")[-1] in img_fmts])

        if self.colab:
            self.subject_list = [self.subject_list[0]]

        # smpl related
        self.smpl_data = SMPLX()

        # smpl-smplx correspondence
        self.smpl_joint_ids_24 = np.arange(22).tolist() + [68, 73]
        self.smpl_joint_ids_24_pixie = np.arange(22).tolist() + [68 + 61, 72 + 68]
        self.get_smpl_model = lambda smpl_type, smpl_gender: smplx.create(
            model_path=self.smpl_data.model_dir,
            gender=smpl_gender,
            model_type=smpl_type,
            ext='npz'
        )

        # Load SMPL model
        self.smpl_model = self.get_smpl_model(self.smpl_type, self.smpl_gender).to(self.device)
        self.faces = self.smpl_model.faces

        if self.hps_type == 'pymaf':
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
            self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)['model'], strict=True)
            self.hps.eval()

        elif self.hps_type == 'pare':
            self.hps = PARETester(path_config.CFG, path_config.CKPT).model
        elif self.hps_type == 'pixie':
            self.hps = PIXIE(config=pixie_cfg, device=self.device)
            self.smpl_model = self.hps.smplx
        elif self.hps_type == 'hybrik':
            smpl_path = osp.join(self.smpl_data.model_dir, "smpl/SMPL_NEUTRAL.pkl")
            self.hps = HybrIKBaseSMPLCam(
                cfg_file=path_config.HYBRIK_CFG,
                smpl_path=smpl_path,
                data_path=path_config.hybrik_data_dir
            )
            self.hps.load_state_dict(
                torch.load(path_config.HYBRIK_CKPT, map_location='cpu'), strict=False
            )
            self.hps.to(self.device)
        elif self.hps_type == 'bev':
            try:
                import bev
            except:
                print('Could not find bev, installing via pip install --upgrade simple-romp')
                os.system('pip install simple-romp==1.0.3')
                import bev
            settings = bev.main.default_settings
            # change the argparse settings of bev here if you prefer other settings.
            settings.mode = 'image'
            settings.GPU = int(str(self.device).split(':')[1])
            settings.show_largest = True
            # settings.show = True # uncommit this to show the original BEV predictions
            self.hps = bev.BEV(settings)

        print(colored(f"Using {self.hps_type} as HPS Estimator\n", "green"))

        self.render = Render(size=512, device=device)

    def __len__(self):
        return len(self.subject_list)

    def compute_vis_cmap(self, smpl_verts, smpl_faces):

        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, -z, torch.as_tensor(smpl_faces).long())
        smpl_cmap = self.smpl_data.cmap_smpl_vids(self.smpl_type)

        return {
            'smpl_vis': smpl_vis.unsqueeze(0).to(self.device),
            'smpl_cmap': smpl_cmap.unsqueeze(0).to(self.device),
            'smpl_verts': smpl_verts.unsqueeze(0)
        }

    def compute_voxel_verts(self, body_pose, global_orient, betas, trans, scale):

        smpl_path = osp.join(self.smpl_data.model_dir, "smpl/SMPL_NEUTRAL.pkl")
        tetra_path = osp.join(self.smpl_data.tedra_dir, 'tetra_neutral_adult_smpl.npz')
        smpl_model = TetraSMPLModel(smpl_path, tetra_path, 'adult')

        pose = torch.cat([global_orient[0], body_pose[0]], dim=0)
        smpl_model.set_params(rotation_matrix_to_angle_axis(rot6d_to_rotmat(pose)), beta=betas[0])

        verts = np.concatenate([smpl_model.verts, smpl_model.verts_added],
                               axis=0) * scale.item() + trans.detach().cpu().numpy()
        faces = np.loadtxt(
            osp.join(self.smpl_data.tedra_dir, 'tetrahedrons_neutral_adult.txt'), dtype=np.int32
        ) - 1

        pad_v_num = int(8000 - verts.shape[0])
        pad_f_num = int(25100 - faces.shape[0])

        verts = np.pad(verts,
                       ((0, pad_v_num),
                        (0, 0)), mode='constant', constant_values=0.0).astype(np.float32) * 0.5
        faces = np.pad(faces, ((0, pad_f_num), (0, 0)), mode='constant',
                       constant_values=0.0).astype(np.int32)

        verts[:, 2] *= -1.0

        voxel_dict = {
            'voxel_verts': torch.from_numpy(verts).to(self.device).unsqueeze(0).float(),
            'voxel_faces': torch.from_numpy(faces).to(self.device).unsqueeze(0).long(),
            'pad_v_num': torch.tensor(pad_v_num).to(self.device).unsqueeze(0).long(),
            'pad_f_num': torch.tensor(pad_f_num).to(self.device).unsqueeze(0).long()
        }

        return voxel_dict

    def __getitem__(self, index):

        img_path = self.subject_list[index]
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]

        if self.seg_dir is None:
            img_icon, img_hps, img_ori, img_mask, uncrop_param = process_image(
                img_path, self.hps_type, 512, self.device
            )

            data_dict = {
                'name': img_name,
                'image': img_icon.to(self.device).unsqueeze(0),
                'ori_image': img_ori,
                'mask': img_mask,
                'uncrop_param': uncrop_param
            }

        else:
            img_icon, img_hps, img_ori, img_mask, uncrop_param, segmentations = process_image(
                img_path,
                self.hps_type,
                512,
                self.device,
                seg_path=os.path.join(self.seg_dir, f'{img_name}.json')
            )
            data_dict = {
                'name': img_name,
                'image': img_icon.to(self.device).unsqueeze(0),
                'ori_image': img_ori,
                'mask': img_mask,
                'uncrop_param': uncrop_param,
                'segmentations': segmentations
            }

        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
            preds_dict = self.hps.forward(img_hps)

        data_dict['smpl_faces'] = torch.Tensor(self.faces.astype(np.int64)).long().unsqueeze(0).to(
            self.device
        )

        if self.hps_type == 'pymaf':
            output = preds_dict['smpl_out'][-1]
            scale, tranX, tranY = output['theta'][0, :3]
            data_dict['betas'] = output['pred_shape']
            data_dict['body_pose'] = output['rotmat'][:, 1:]
            data_dict['global_orient'] = output['rotmat'][:, 0:1]
            data_dict['smpl_verts'] = output['verts']
            data_dict["type"] = "smpl"

        elif self.hps_type == 'pare':
            data_dict['body_pose'] = preds_dict['pred_pose'][:, 1:]
            data_dict['global_orient'] = preds_dict['pred_pose'][:, 0:1]
            data_dict['betas'] = preds_dict['pred_shape']
            data_dict['smpl_verts'] = preds_dict['smpl_vertices']
            scale, tranX, tranY = preds_dict['pred_cam'][0, :3]
            data_dict["type"] = "smpl"

        elif self.hps_type == 'pixie':
            data_dict.update(preds_dict)
            data_dict['body_pose'] = preds_dict['body_pose']
            data_dict['global_orient'] = preds_dict['global_pose']
            data_dict['betas'] = preds_dict['shape']
            data_dict['smpl_verts'] = preds_dict['vertices']
            scale, tranX, tranY = preds_dict['cam'][0, :3]
            data_dict["type"] = "smplx"

        elif self.hps_type == 'hybrik':
            data_dict['body_pose'] = preds_dict['pred_theta_mats'][:, 1:]
            data_dict['global_orient'] = preds_dict['pred_theta_mats'][:, [0]]
            data_dict['betas'] = preds_dict['pred_shape']
            data_dict['smpl_verts'] = preds_dict['pred_vertices']
            scale, tranX, tranY = preds_dict['pred_camera'][0, :3]
            scale = scale * 2
            data_dict["type"] = "smpl"

        elif self.hps_type == 'bev':
            data_dict['betas'] = torch.from_numpy(preds_dict['smpl_betas'])[[0], :10].to(
                self.device
            ).float()
            pred_thetas = batch_rodrigues(
                torch.from_numpy(preds_dict['smpl_thetas'][0]).reshape(-1, 3)
            ).float()
            data_dict['body_pose'] = pred_thetas[1:][None].to(self.device)
            data_dict['global_orient'] = pred_thetas[[0]][None].to(self.device)
            data_dict['smpl_verts'] = torch.from_numpy(preds_dict['verts'][[0]]).to(self.device
                                                                                   ).float()
            tranX = preds_dict['cam_trans'][0, 0]
            tranY = preds_dict['cam'][0, 1] + 0.28
            scale = preds_dict['cam'][0, 0] * 1.1
            data_dict["type"] = "smpl"

        data_dict['scale'] = scale
        data_dict['trans'] = torch.tensor([tranX, tranY, 0.0]).unsqueeze(0).to(self.device).float()

        # data_dict info (key-shape):
        # scale, tranX, tranY - tensor.float
        # betas - [1,10] / [1, 200]
        # body_pose - [1, 23, 3, 3] / [1, 21, 3, 3]
        # global_orient - [1, 1, 3, 3]
        # smpl_verts - [1, 6890, 3] / [1, 10475, 3]

        # from rot_mat to rot_6d for better optimization
        N_body = data_dict["body_pose"].shape[1]
        data_dict["body_pose"] = data_dict["body_pose"][:, :, :, :2].reshape(1, N_body, -1)
        data_dict["global_orient"] = data_dict["global_orient"][:, :, :, :2].reshape(1, 1, -1)
        
        if self.args.calc_metric:
            self.smplx = SMPLX()
            image_root_path=os.path.split(img_path)[0]
            data_dict.update(self.load_calib(os.path.join(image_root_path, img_name+".txt")))
            data_dict.update(self.load_mesh(os.path.join(image_root_path, img_name+".glb"), 100))
            smplx_verts =self.rescale_smpl(os.path.join(image_root_path, img_name+".obj"), 100)
            data_dict.update({'smpl_verts': smplx_verts})
                    
        # data_pamir_icon=self.compute_voxel_verts(data_dict["body_pose"],data_dict["global_orient"],data_dict['betas'],data_dict['trans'],data_dict['scale'] ) #body_pose, global_orient, betas, trans, scale
        # data_dict.update(data_pamir_icon)
        return data_dict


    def rescale_smpl(self, fitted_path, scale=100, translate=(0, 0, 0)):


        fitted_body = trimesh.load(fitted_path, process=False, maintain_order=True, skip_materials=True)
        resize_matrix = trimesh.transformations.scale_and_translate(scale=(scale), translate=translate)

        fitted_body.apply_transform(resize_matrix)

        return np.array(fitted_body.vertices)

    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_rgb_image()

    def render_depth(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_depth_map(cam_ids=[0, 2])


    def load_mesh(self, mesh_path, scale=100.0):


        tscene: trimesh.Scene = trimesh.load_mesh(mesh_path, process=False)
        assert len(tscene.geometry) == 1
        tmesh: trimesh.Trimesh = next(iter(tscene.geometry.values()))
        tmesh.apply_scale(scale)

        mesh = HoppeMesh(tmesh)

        return {
            'mesh': mesh,
            'verts_gt': torch.as_tensor(mesh.verts).float(),
            'faces_gt': torch.as_tensor(mesh.faces).long()
        }
    def visualize_alignment(self, data):

        import vedo
        import trimesh

        if self.hps_type != 'pixie':
            smpl_out = self.smpl_model(
                betas=data['betas'],
                body_pose=data['body_pose'],
                global_orient=data['global_orient'],
                pose2rot=False
            )
            smpl_verts = ((smpl_out.vertices + data['trans']) *
                          data['scale']).detach().cpu().numpy()[0]
        else:
            smpl_verts, _, _ = self.smpl_model(
                shape_params=data['betas'],
                expression_params=data['exp'],
                body_pose=data['body_pose'],
                global_pose=data['global_orient'],
                jaw_pose=data['jaw_pose'],
                left_hand_pose=data['left_hand_pose'],
                right_hand_pose=data['right_hand_pose']
            )

            smpl_verts = ((smpl_verts + data['trans']) * data['scale']).detach().cpu().numpy()[0]

        smpl_verts *= np.array([1.0, -1.0, -1.0])
        faces = data['smpl_faces'][0].detach().cpu().numpy()

        image_P = data['image']
        image_F, image_B = self.render_normal(smpl_verts, faces)

        # create plot
        vp = vedo.Plotter(title="", size=(1500, 1500))
        vis_list = []

        image_F = (0.5 * (1.0 + image_F[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
        image_B = (0.5 * (1.0 + image_B[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
        image_P = (0.5 * (1.0 + image_P[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)

        vis_list.append(
            vedo.Picture(image_P * 0.5 + image_F * 0.5).scale(2.0 / image_P.shape[0]
                                                             ).pos(-1.0, -1.0, 1.0)
        )
        vis_list.append(vedo.Picture(image_F).scale(2.0 / image_F.shape[0]).pos(-1.0, -1.0, -0.5))
        vis_list.append(vedo.Picture(image_B).scale(2.0 / image_B.shape[0]).pos(-1.0, -1.0, -1.0))

        # create a mesh
        mesh = trimesh.Trimesh(smpl_verts, faces, process=False)
        mesh.visual.vertex_colors = [200, 200, 0]
        vis_list.append(mesh)

        vp.show(*vis_list, bg="white", axes=1, interactive=True)

    def load_calib(self, calib_path):
        calib_data = np.loadtxt(calib_path, dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib_mat = torch.from_numpy(calib_mat).float()
        return {'calib': calib_mat}