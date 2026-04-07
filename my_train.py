#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene import CameraDataset
import copy
import random
import math
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 设置默认值
# FEDERATED = False
FED_NUM_CLIENTS = 1
FED_ROUNDS = 1
FED_LOCAL_STEPS = 0
FED_IID = True

def client_local_update(cid, global_capture, dataset, cams, opt, pipe, background, local_steps, flame_param=None, flame_param_orig=None, iteration=0):
    if global_capture and isinstance(global_capture[0], int):
        sh_degree = dataset.sh_degree
        print(f"客户端 {cid} 收到全局模型参数，最大SH阶数为 {sh_degree}")
    else:
        print(f"错误，客户端 {cid} 没有收到全局模型参数")
        sys.exit(1)
    ########## 遵照3DGS的流程，先进行一个初始化（同时更新sh_degree），然后再恢复全局模型 ##########
    if dataset.bind_to_mesh:
        local_gaussians = FlameGaussianModel(sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
        print(f"客户端 {cid} 进入bind_to_mesh分支,使用FlameGaussianModel")

        if flame_param is not None:
            local_gaussians.flame_param = copy.deepcopy(flame_param)

            local_gaussians.flame_param_orig = copy.deepcopy(flame_param_orig) if flame_param_orig is not None else copy.deepcopy(flame_param)      
            try:
                local_gaussians.num_timesteps = local_gaussians.flame_param['expr'].shape[0]
            except Exception:
                print(f"客户端 {cid} 恢复全局模型参数失败，num_timesteps没有对齐")
                sys.exit(1)

    else:
        local_gaussians = GaussianModel(sh_degree)
        print(f"客户端 {cid} 进入非bind_to_mesh分支,使用GaussianModel")
        sys.exit(1)

    ########## 恢复全局模型参数 ##########
    # 注意这里默认只看iid情况下的恢复，如果是non-iid的话，optimizer应该重置
    # 另外：如果控制是否共享optimizer，可以作为消融实验的内容？
    try:
        local_gaussians.restore(copy.deepcopy(global_capture), opt)
        # 直接重置优化器状态，等于不共享优化器，这样可以避免非iid情况下的训练不稳定问题以及一些优化器同步的问题
        local_gaussians.training_setup(opt)
    except Exception as e:
        local_gaussians.training_setup(opt)
        print(f"客户端 {cid} 恢复全局模型参数失败: {e}")
        sys.exit(1)
    
    ########## 处理数据集 ##########
    local_dataset = CameraDataset(cams)
    # 这里要留意一下几个参数的设置，后面有一个保留进程的参数persistent_workers，不知道是否正确
    loader_camera_train = DataLoader(local_dataset, batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)

    progress_bar = tqdm(range(0, local_steps), desc="训练进度")
    # ema存在的理由是为了在进度条上显示一个平滑的损失值，避免每一步的损失波动过大导致难以观察训练趋势
    ema_loss_for_log = 0.0

    ########## 本地训练循环 ##########
    for step in tqdm(range(1, local_steps + 1), desc=f"客户端 {cid} 本地训练", leave=False):
        # 更新学习率
        local_gaussians.update_learning_rate(iteration + step)  
        # 增加阶数
        if (iteration + step) % 1000 == 0:
            # print("进入增加阶数的分支")
            local_gaussians.oneupSHdegree()
            # print(f"最大阶数为： {local_gaussians.max_sh_degree}，当前阶数为： {local_gaussians.active_sh_degree}")
        # 这里分支是必要的，因为数据集可能不够1000步训练
        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)
        # 有绑定 mesh，则选择对应时间步的 mesh
        if local_gaussians.binding is not None:
            local_gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        ##### 渲染 #####
        # 这里去掉了debug_from的逻辑，因为用不到
        # if (iteration - 1) == debug_from:
        #     pipe.debug = True    

        render_pkg = render(viewpoint_cam, local_gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        losses = {}
        losses["l1"] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim
        if local_gaussians.binding != None:
            if opt.metric_xyz:    # 公式(8)
                losses['xyz'] = F.relu((local_gaussians._xyz*local_gaussians.face_scaling[local_gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:
                # losses['xyz'] = local_gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                losses['xyz'] = F.relu(local_gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                if opt.metric_scale:     # 公式(9)
                    losses['scale'] = F.relu(local_gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    # losses['scale'] = F.relu(local_gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    losses['scale'] = F.relu(torch.exp(local_gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
            # dynamic offset 是作用于每个高斯点的时间步偏移量，用于表示脸部的细微变化
            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = local_gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices = [ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < local_gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = local_gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
            # 拉普拉斯：平滑约束，避免高频噪声或孤立点
            if opt.lambda_laplacian != 0:
                losses['lap'] = local_gaussians.compute_laplacian_loss() * opt.lambda_laplacian
        losses['total'] = sum([v for k, v in losses.items()])

        losses['total'].backward()
        # 输出日志
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if (iteration + step) % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if step == local_steps:
                progress_bar.close()
        
        # 稠密化前置步骤：积累梯度信息，更新计数器以及最大半径统计
        local_gaussians.max_radii2D[visibility_filter] = torch.max(local_gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        local_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
        if step < local_steps:
            local_gaussians.optimizer.step()
            local_gaussians.optimizer.zero_grad(set_to_none = True)
        

    capture = local_gaussians.capture()
    del local_gaussians
    torch.cuda.empty_cache()
    return capture, len(cams)

        

def fedavg_aggregate(captures, weights):
    """
    加权平均
    self._xyz
    self._features_dc
    self._features_rest
    self._scaling
    self._rotation
    self._opacity   
    不变
    self.binding
    self.binding_counter
    self.spatial_lr_scale
    取最大
    self.max_radii2D
    累加
    self.xyz_gradient_accum
    self.denom
    不维护
    self.optimizer.state_dict()
    取一个
    self.active_sh_degree

    captures: List[tuple]  # 每个 client 的 capture()
    weights:  List[float]  # 每个 client 的权重（如数据量占比）

    return: aggregated capture (tuple)
    """
    print("====聚合====")
    assert len(captures) > 0
    K = len(captures)

    # 归一化权重 
    weights = torch.tensor(weights, dtype=torch.float32, device="cuda")
    weights = weights / weights.sum()

    # 每个 capture 的结构：
    # (
    #   active_sh_degree,
    #   _xyz,
    #   _features_dc,
    #   _features_rest,
    #   _scaling,
    #   _rotation,
    #   _opacity,
    #   binding,
    #   binding_counter,
    #   max_radii2D,
    #   xyz_gradient_accum,
    #   denom,
    #   optimizer_state_dict,
    #   spatial_lr_scale
    # )

    active_sh_degree = captures[0][0]  # 直接取一个，因为按照设计，每个客户端的阶数在聚合阶段都是一样的

    # 加权平均参数
    def weighted_sum(idx):
        return sum(w * cap[idx] for w, cap in zip(weights, captures))

    xyz = weighted_sum(1).detach().clone()  
    f_dc = weighted_sum(2).detach().clone()
    f_rest = weighted_sum(3).detach().clone()
    scaling = weighted_sum(4).detach().clone()
    # rotation 聚合
    rotations = [cap[5] for cap in captures]

    # 选第一个 client 作为参考方向
    ref_rot = rotations[0]

    aligned_rots = []
    for r in rotations:
        # 计算点积（逐点）
        dot = (r * ref_rot).sum(dim=-1, keepdim=True)
        
        # 如果方向相反（dot < 0），就翻转
        r = torch.where(dot < 0, -r, r)
        
        aligned_rots.append(r)

    # 加权平均
    rotation = sum(w * r for w, r in zip(weights, aligned_rots))

    # 归一化（避免数值问题）
    rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True).clamp(min=1e-8)

    # 防止 non-leaf tensor
    rotation = rotation.detach().clone()

    opacity = weighted_sum(6).detach().clone()

    # 不变项
    binding = captures[0][7]
    binding_counter = captures[0][8]
    spatial_lr_scale = captures[0][13]

    # max_radii2D（取最大）
    max_radii2D = torch.stack([cap[9] for cap in captures], dim=0).max(dim=0).values.detach().clone()

    # densification 统计（累加）
    xyz_gradient_accum = sum(cap[10] for cap in captures).detach().clone()
    denom = sum(cap[11] for cap in captures).detach().clone()

    # optimizer（不维护）
    opt_dict = None

    return (
        active_sh_degree,
        xyz,
        f_dc,
        f_rest,
        scaling,
        rotation,
        opacity,
        binding,
        binding_counter,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale,
    )


def training(dataset, opt, pipe, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
        print("进入bind_to_mesh分支,使用FlameGaussianModel")
    else:
        gaussians = GaussianModel(dataset.sh_degree)
        print("进入非bind_to_mesh分支,使用GaussianModel")
        sys.exit(1)  
        
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # 我把checkpoint机制暂时去除 ##############################
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    # 置背景色张量到 CUDA   白色背景可以使得模型快速收敛
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("====初始化完成，进入联邦学习训练流程====")
    
    # 创建两个 CUDA 计时事件 iter_start/iter_end ########################3
    # iter_start = torch.cuda.Event(enable_timing = True)
    # iter_end = torch.cuda.Event(enable_timing = True)

    # 分割训练相机列表
    # 这里scale_key 默认为1，因为论文实现中利用的数据集都是有原分辨率的
    scale_key = 1.0 
    all_train_cams = scene.train_cameras[scale_key]
    # clients 是一个元素为列表的列表，每个元素对应一个客户端的相机子集
    clients = split_cameras(all_train_cams, FED_NUM_CLIENTS, iid=FED_IID)

    # 全局迭代次数
    iteration = 0

    ##### 主训练循环 #####
    for r in tqdm(range(1, FED_ROUNDS + 1), desc="训练轮次"):
        print(f"=== 联邦第 {r}/{FED_ROUNDS} 轮 ===") 
        # 记录全局训练参数
        global_capture = gaussians.capture()

        client_captures = []
        client_weights = []    # 这里的权重是指每个客户端的训练数据量占总训练数据量的比例

        for cid in tqdm(range(FED_NUM_CLIENTS), desc=f"客户端 (r={r})", leave=False):
            # cams:每个客户端的相机子集，代表该客户端的训练数据
            cams = clients[cid]

            if len(cams) == 0:
                print(f"错误，客户端 {cid} 没有数据")
                sys.exit(1)
                # continue

            cap, n_samples = client_local_update(
                cid,
                copy.deepcopy(global_capture),
                dataset,
                cams,
                opt,
                pipe,
                background,
                FED_LOCAL_STEPS,
                # provide flame parameters so clients can initialize mesh state
                copy.deepcopy(gaussians.flame_param) if hasattr(gaussians, 'flame_param') else None,
                copy.deepcopy(gaussians.flame_param_orig) if hasattr(gaussians, 'flame_param_orig') else None,
                iteration
            )
            client_captures.append(cap)
            client_weights.append(n_samples)
            # 这里是把每个client的训练看作是并行的部分，也就是说n个clients进行FED_LOCAL_STEPS次训练对于全局来说
            if cid == (FED_NUM_CLIENTS - 1):  
                iteration += FED_LOCAL_STEPS
            print(f"直到现在，所有客户端前 {iteration} 次迭代已经完成")

        if len(client_captures) == 0:
            print("错误，没有任何客户端模型参数")
            sys.exit(1)
            # break

        ##### 聚合 #####
        aggregated = fedavg_aggregate(client_captures, client_weights)
        # 恢复聚合后的全局模型参数
        gaussians.restore(aggregated, opt)  

        ##### 评估 #####
        with torch.no_grad():
            training_report(iteration, args.test_iterations, scene, render, (pipe, background))

        ##### 保存 #####
        if (iteration in saving_iterations):
            print("[第 {} 次迭代] 正在保存".format(iteration))
            scene.save(iteration)

        ##### 稠密化和剪枝 #####
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            print(f"进入稠密化和剪枝的分支，全局第 {iteration} 次迭代")
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        
        # 周期性地“重置高斯点的不透明度（opacity/alpha）”，用来促进稠密化和防止死点（可以理解为透明点，dead Gaussians）长期存在
        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            print(f"进入全局重置透明度的分支，全局第 {iteration} 次迭代")
            gaussians.reset_opacity()




def split_cameras(all_train_cameras, num_clients, iid=True):
    """
    将训练相机列表切分到多个客户端（用于联邦模拟）。

    输入:
    - all_train_cameras: 所有训练相机列表。
    - num_clients: 客户端数量。
    - iid: 是否按交替分配（True）以实现近似 IID 分布；否则按连续区间切分。

    返回:
    - clients: 长度为 num_clients 的列表，每项是该客户端的相机子集。
    """
    cams = list(all_train_cameras)
    n = len(cams)
    if num_clients <= 1:
        return [cams]
    clients = [[] for _ in range(num_clients)]
    if iid:
        random.shuffle(cams)
        for i, cam in enumerate(cams):
            clients[i % num_clients].append(cam)
    else:
        # cams = sorted(cams, key=lambda x: x.timestep)  # 如果相机对象有时间戳属性，可以按时间排序以实现非 IID 分布 这里默认原顺序就是按时间排序的 注意scene在初始化的时候默认shuffle=True
        per = math.ceil(n / num_clients)
        for i in range(num_clients):
            start = i * per
            end = min((i + 1) * per, n)
            clients[i] = cams[start:end]
    return clients

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(iteration, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if iteration in testing_iterations:
        print("\n[第 {} 次迭代] 正在测试".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                image_cache = []
                gt_image_cache = []
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                with open(os.path.join(scene.model_path, "log.txt"), 'a') as file:
                    file.write("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}\n".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
            else:
                print(f"第 {iteration} 次迭代测试出错，相机集合的长度为{len(config['cameras']) if config['cameras'] else 'None'}")

        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser   设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")    # (这就是解析器对象)参数说明=训练脚本参数
    lp = ModelParams(parser)          # 模型参数
    op = OptimizationParams(parser)   # 优化参数
    pp = PipelineParams(parser)       # 管道参数

    # 联邦参数
    # parser.add_argument('--federated', action='store_true', help='Enable federated simulation')
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--local_steps', type=int, default=2000)
    parser.add_argument('--iid', action='store_true')

    parser.add_argument('--ip', type=str, default="127.0.0.1")   # 设置GUI服务器的ip，用于可视化训练
    parser.add_argument('--port', type=int, default=6009)        # 设置GUI服务器的端口，用于可视化训练
    parser.add_argument('--debug_from', type=int, default=-1)    # 从哪个迭代开始调试，默认-1表示不调试
    parser.add_argument('--detect_anomaly', action='store_true', default=False)   #异常检测开关
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")  # 迭代间隔
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])     # 测试迭代点
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])     # 保存迭代点
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[]) # 保存断点的迭代点
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    # 联邦参数设置
    FED_NUM_CLIENTS = args.num_clients
    FED_ROUNDS = args.rounds
    FED_LOCAL_STEPS = args.local_steps
    FED_IID = args.iid

    # 健壮性设置
    if args.interval > op.iterations:    # 如果间隔大于总迭代次数，则将间隔设置为总迭代次数的五分之一
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:   # 如果没有指定测试迭代点，则自动设置测试迭代点
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:   # 如果没有指定保存迭代点，则自动设置保存迭代点
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0: # 如果没有指定检查点迭代点，则自动设置检查点迭代点
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)  初始化系统状态，并且使用固定种子
    safe_state(args.quiet)   

    # Start GUI server, configure and run training   
    # network_gui.init(args.ip, args.port)   # 初始化GUI服务器
    torch.autograd.set_detect_anomaly(args.detect_anomaly)   # 设置异常检测

    globals()['FED_NUM_CLIENTS'] = FED_NUM_CLIENTS
    globals()['FED_ROUNDS'] = FED_ROUNDS
    globals()['FED_LOCAL_STEPS'] = FED_LOCAL_STEPS
    globals()['FED_IID'] = FED_IID

    # 开始训练
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations)

    # All done
    print("\nTraining complete.")
