from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np

from model import RL_Policy, Semantic_Mapping
from utils.storage import GlobalRolloutStorage
from envs import make_vec_envs
from arguments import get_args
import algo

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = get_args() #解析命令行参数，也就是args变成了一个包含各种参数的对象，比如args.seed, args.device等等

    np.random.seed(args.seed) #设置NumPy的随机种子，确保所有使用NumPy生成的随机数都可以复现
    torch.manual_seed(args.seed) #设置CPU端的随机种子

    if args.cuda: #如果启动了CUDA（即使用GPU进行计算）
        torch.cuda.manual_seed(args.seed) #设置GPU的随机种子

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name) #生成模型保存路径，这个目录就是用来保存训练过程中生成的模型文件（.pth）或 checkpoints 的。
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name) #生成运行日志或中间结果保存路径

    if not os.path.exists(log_dir):  #检查上面的log_dir 和dump_dir路径是否存在，如果不存在就自动创建 
        os.makedirs(log_dir)  
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir) 

    logging.basicConfig(  #配置了 Python 的内置 logging 模块，用来将日志信息写入文件
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir)) #把保存路径打印出来，方便你知道模型/log 会存在哪里
    print(args) #输出参数配置，方便知道种子、学习率等等
    logging.info(args) #同时把参数写进日志文件里

    # Logging and loss variables
    num_scenes = args.num_processes  #并发环境数量等于使用等进程数，每个进程负责跑一部分 episode（比如第 0 个进程跑第 0~49 个 episode）
    num_episodes = int(args.num_eval_episodes) #评估时要执行的 episode 数量
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu") #使用 GPU 还是 CPU 进行训练，赋值为 cuda:0 或 cpu

    g_masks = torch.ones(num_scenes).float().to(device) # 表示创建一个长度为num_scenes，值都为1的张量，数据类型为float32,并把这个张量移动到指定的设备上，比如 CPU 或 GPU。

    best_g_reward = -np.inf #赋值为负无穷，用于保存最好的全局reward

    if args.eval: #如果是评估模式，这里关心具体的进程，因为不同的进程可能是在不同的场景下跑的（如厨房，卧室、浴室）
        episode_success = []
        episode_spl = []
        episode_dist = []
        for _ in range(args.num_processes): #这里下划线跟i是一样的，不过这里你不关心它的值
            episode_success.append(deque(maxlen=num_episodes)) #在episode_success这个列表里添加一个最大长度为num_episodes的双端队列（deque是语法），这里一个列表最终添加了args.num_processes（进程数）个双端队列
            episode_spl.append(deque(maxlen=num_episodes))  #也就是一个列表里最后有进程数个双端队列
            episode_dist.append(deque(maxlen=num_episodes)) #记录 agent 与目标的最终距离

    else: #如果不是评估模式（即训练模式下），就使用全局deque, 只关心总体趋势（如成功率是否上升）
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)

    finished = np.zeros((args.num_processes)) #记录每个进程的 episode 是否完成
    wait_env = np.zeros((args.num_processes)) #可能用于统计环境是否在等待某些操作（比如数据加载）

    g_episode_rewards = deque(maxlen=1000) #每个episode总的reward

    g_value_losses = deque(maxlen=1000) #值函数的loss
    g_action_losses = deque(maxlen=1000) #策略/动作函数的loss
    g_dist_entropies = deque(maxlen=1000) #策略分布的熵（用于鼓励探索）

    per_step_g_rewards = deque(maxlen=1000) #每一步的reward

    g_process_rewards = np.zeros((num_scenes)) #每个并发环境的rewards，用于更新

    # Starting environments 
    torch.set_num_threads(1) #只用一个线程（不是进程）来执行CPU运算
    envs = make_vec_envs(args) #创建一个并行环境容器，即多个环境实例组成的向量   ######并行环境容器：可以同时运行多个仿真环境，并像一个大环境一样控制它们 #####如 envs.reset()就是一次性重制所有环境  envs.step(action)就是给每个环境执行一个动作，所有环境都执行一步
    obs, infos = envs.reset() #obs是智能体看到的所有并行环境的初始观测，通常是图像（RGB/RGB-D张量等等）或者状态向量（坐标、角度等等） #### infos是一些环境返回的额外信息

    torch.set_grad_enabled(False) #关闭梯度计算，用于评估模式或无训练阶段

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels ####这里好像有一个问题，论文里是n+2个channel(obstacle map and explored area)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution  #地图边长整除每一个格子的边长，这个地图有map_size * map_size 格
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)  #args.global_downscaling 是缩放比例，得到局部语义地图的边长数  #####局部语义地图是为了让智能体专注于当前可行走的局部区域
    local_h = int(full_h / args.global_downscaling) 

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device) #创建一个全局地图张量,形状是[num_scenes, nc, full_w, full_h] ### num_scenes：并发的环境/进程数量（每个环境一个地图）nc:channel数
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)  #同理，创建一个局部地图张量

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)   #创建全局位姿张量, 形状是(num_scenes, 3) ###num_scenes： 并发环境/进程数量/agent数量（每个并发环境中有一个agent）, 3表示每个agent的位姿由x,y,θ组成
    local_pose = torch.zeros(num_scenes, 3).float().to(device) # 创建局部位姿张量，用于局部规划，避障

    # Origin of local map
    origins = np.zeros((num_scenes, 3)) #创建一个局部地图原点的数组，形状是[num_scenes, 3] 每一行记录一个agent的局部地图原点在全局地图中的坐标和朝向

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int) #一个数组，每一行记录局部地图的上下左右边界索引在全局地图中的位置，astype(int)是将边界从float类型转化为int类型

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))。#其实就是合并了上面的origins和lmp数组，这个是专门为路径规划模块准备的

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):  #根据机器人当前位置，从全局语义地图中裁剪出局部地图 （上下左右边界）
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes     ######   这里的三行为拆解数据
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:   #如果需要缩放
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2   #这里的局部窗口是以agent的坐标为中心，计算出左上角点的坐标
            gx2, gy2 = gx1 + local_w, gy1 + local_h                #根据左上角的坐标计算出右下角的坐标
            if gx1 < 0:                                   #### 以下这四个if是为了保证局部边界不超出地图范围
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:    #正常实验肯定是args,global_downscaling大于1的，这里加了不缩放的情况是要跑一个baseline或ablation实验，
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2] #返回左上角点和右下角点的坐标，注意顺序

    def init_map_and_pose():
        full_map.fill_(0.)   #将地图中所有值设置为0，
        full_pose.fill_(0.)  #将位姿都设为0
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0  #将Agent的初始位置设置在地图的中心（单位m） ##full_pose的形状为[num_scenes, 3], 这里的full_pose[:, :2]表示所有scene只取前面两个维度

        locs = full_pose.cpu().numpy()   #把full_pose从张量转化numpy ###这里.cpu()是把这个张量转化到CPU上，因为PyTorch不允许GPU张量直接变成NumPy数组
        planner_pose_inputs[:, :3] = locs 
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),  #把米单位的坐标转化为地图像素单位（格）: 为的是方便把数据存储在数组里（下标调用）
                            int(c * 100.0 / args.map_resolution)]  #args.map_resolution表示地图中的一格（一个像素）代表现实中的多少cm

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0   #在全局地图的通道2（explored area）和通道3 (current agent location)上,以当前位置为中心，画一个3*3的小标记块，表示当前agent的位置

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),      #lmb记录的就是gx1, gx2, gy1, gy2
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,           #把局部地图的左上角设置为原点 （这张局部地图是以agent为中心裁的）
                          lmb[e][0] * args.map_resolution / 100.0, 0.]      #即 origins[e] = gx1, gy1

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,                   #从 full_map 中裁剪一块 agent 附近的区域（大小为 local_w × local_h），赋值给 local_map。
                                    lmb[e, 0]:lmb[e, 1],    #local_map的大小就是 (lmb[e, 0]:lmb[e, 1]) * (lmb[e, 2]:lmb[e, 3]), 也就是（gx2-gx1）* (gy2-gy1)
                                    lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \                
                torch.from_numpy(origins[e]).to(device).float()。#计算agent局部地图的坐标 （局部坐标 = 全局坐标 - 局部地图左上角在全局中的坐标）

    def init_map_and_pose_for_env(e):  #这个函数跟上面的函数相同，只不过是针对单环境（好处：可以被单独调用，比如在某个环境重置或失败后，只对这个环境重新初始化，而不影响其他环境。）
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()

    def update_intrinsic_rew(e): #强化学习中探索驱动策略的一种形式：探索了多少新区域，就给多少奖励（有助于agent主动探索环境）
        prev_explored_area = full_map[e, 1].sum(1).sum(0)   #计算之前区域已经探索的面积 ### full_map[e, 1]：第 e 个环境的 full_map 的第 1 个通道，代表“已探索区域”  ### sum(1).sum(0)：先按行再按列求和，相当于对整个通道求总和（像素数量）
        full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \   #把局部地图更新回全局地图中
            local_map[e]
        curr_explored_area = full_map[e, 1].sum(1).sum(0)        # 再次计算更新之后的已探索区域面积
        intrinsic_rews[e] = curr_explored_area - prev_explored_area       # 这就是探索新区域的像素数，作为intrinsic reward
        intrinsic_rews[e] *= (args.map_resolution / 100.)**2  # to m^2   # 把 reward 从单位“像素数”转换为“平方米”，跟实际物理世界相挂钩 ### args.map_resolution表示地图中的一格（一个像素）代表现实中的多少cm

    init_map_and_pose()

    # Global policy observation space。 #“观测张量”（observation tensor）是智能体每一步从环境中接收到的信息的集合 —— 通常是一个张量（多维数组）   比如图像类型的观测（channels, height, width）
    ngc = 8 + args.num_sem_categories     
    es = 2
    g_observation_space = gym.spaces.Box(0, 1,     #gym.space.Box 是 OpenAI Gym 中的一个类，用来定义一个 连续数值空间 ### 用法上可以理解为Box(low, high, shape)：所有数值都在low到high之间，张量形状是shape, 数据类型是dtype
                                         (ngc,      #这个张量是一个多通道的局部语义地图图像
                                          local_w,
                                          local_h), dtype='uint8') #dtype='uint8' 表示张量中的每个元素的数据类型是 无符号8位整数

    # Global policy action space 
    g_action_space = gym.spaces.Box(low=0.0, high=0.99,  #这里是动作空间，这个动作代表“下一步想去的位置”的地图坐标 [0, 0.99]会映射成实际坐标
                                    shape=(2,), dtype=np.float32)

    # Global policy recurrent layer size
    g_hidden_size = args.global_hidden_size    #把全局策略中RNN的隐藏层大小/维度（隐藏层中神经元的个数）赋值给变量 g_hidden_size  ##

    # Semantic Mapping   ### 根据智能体的感知结果（图像等），构建出具有语义标签的地图
    sem_map_module = Semantic_Mapping(args).to(device) #创建一个语义地图构建器的实例，args 传入了参数（如类别数、分辨率等）
    sem_map_module.eval()  #进入评估模式


    关于强化学习
        强化学习（RL）是一个智能体（agent）在环境（environment）中“试错学习”的过程，它大致分为两个阶段：
            1. 交互阶段（作用：收集经验）
                action = g_agent.act(observation, ...)
                g_agent.act(...) 就是让 agent 根据当前观测 observation 选出一个动作 action。
		        然后这个动作会被送到环境中执行，环境会返回：
		            新的观测 observation’
		            奖励 reward
		            是否结束 done
		        这个过程会持续多步，直到收集够了一批“经验”。
                这批经验的形式可以是这样一组数据：(observation, action, reward, next_observation, done)
                我们称为 transition（转移）。多步的 transition 加起来就是一批经验。
            2. 学习阶段（作用：更新策略）
                g_agent.update(batch)
                这一步使用上面收集的那一批经验（batch）来更新策略网络中的参数。
                它本质上就是 用“表现好”的动作增加概率，差的减少概率。
                PPO 算法用的是“剪切损失函数 + 值函数 + 熵正则”，这一切都封装在 g_agent.update() 里了。
            也就是说：
                act() 是“行动”，update() 是“学习”。
    
    # Global policy
    #g_poicy这一行会构建一个用于全局导航策略的神经网络模型，它的输入是局部语义地图，输出是一个动作
    g_policy = RL_Policy(g_observation_space.shape, # 输入空间形状，例如 (channel, height, width)
                         g_action_space,    # 动作空间，前面定义为 Box(2,)
                         model_type=1,    # 模型类型：通常控制网络结构的选择 这里暂时不用去管
                         base_kwargs={'recurrent': args.use_recurrent_global, # 是否使用RNN（LSTM/GRU）
                                      'hidden_size': g_hidden_size,       # 隐藏层神经元数
                                      'num_sem_categories': ngc - 8    # 语义类别数（减去基础通道）
                                      }).to(device)
    g_agent = algo.PPO(g_policy,    #用PPO算法初始化一个“智能体”对象g_agent，它包含一个策略网络 (g_policy)和value网络 (值函数)，包含优化器（用于调参） ### 可以执行g_agent.act(), g_agent.update()等动作 
                       args.clip_param, 
                       args.ppo_epoch,
                       args.num_mini_batch, 
                       args.value_loss_coef,
                       args.entropy_coef, 
                       lr=args.lr, 
                       eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h) #创建一个张量，这是全局策略网络（g_policy / global policy）的观测输入，语义地图格式的图像张量. 它不直接观测RGB图像，而是使用的是语义地图这种空间结构的数据表示
    global_orientation = torch.zeros(num_scenes, 1).long() #表示每个环境中agent的朝向，用一个整数（通常代表角度的理想值来存储）
    intrinsic_rews = torch.zeros(num_scenes).to(device) #每个环境当前时间步的内在奖励（intrinsic reward），比如鼓励探索新区域就是内在奖励，而外在奖励是完成某种任务给予的（比如说到达目标、吃掉豆子、打败敌人）
    extras = torch.zeros(num_scenes, 2) #这个张量一般用于存储一些额外信息

    # Storage 
    # 这个是初始化全局策略的经验存储器。 ###经验存储器（rollout buffer）是在强化学习中用来 临时保存智能体与环境交互过程中的数据，以便后续执行策略更新
    # 在PPO中，智能体会与环境交互一段时间（比如128个时间步），每一步都存以下内容：当前观测（state）, 当前动作（action）, 得到的奖励（reward）, 策略的概率输出（log prob）, 值函数预测的 value, RNN 的隐藏状态（如果用了 RNN）
    g_rollouts = GlobalRolloutStorage(args.num_global_steps, #每个rollout的时间步数，rollout时间步数是指你希望智能体在更新策略之前，与环境交互多少步
                                      num_scenes, # 并行环境数
                                      g_observation_space.shape, #观测张量形状 (ngc, local_w, local_h)
                                      g_action_space, g_policy.rec_state_size, #动作空间
                                      es).to(device) #extras 的维度（一般是2维）
    #如果设置了加载预训练模型，则载入参数 
    ###  一个“模型”通常指的是一个神经网络，它有很多参数（也就是权重和偏置）
    ###  “预训练模型”就是：一个已经在某个任务或数据上训练过，保存下来的模型参数文件，你可以拿来“加载”继续使用或微调，而不是从头开始训练。
    if args.load != "0": #如果传入的参数 args.load 不是字符串 "0"，说明用户提供了一个模型文件的路径
        print("Loading model {}".format(args.load)) #torch.load(...) 读取这个文件，获得里面保存的参数（通常是字典形式）。
        state_dict = torch.load(args.load, 
                                map_location=lambda storage, loc: storage)  #torch.load(...) 读取这个文件，获得里面保存的参数（通常是字典形式）。
        g_policy.load_state_dict(state_dict) #load_state_dict(...) 把这些参数“加载”到你的策略网络 g_policy 中。

    if args.eval:  #如果是评估模式（禁用dropout 和 batchnorm 的训练行为）
        g_policy.eval()

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray( #从每个并行环境中提取当前的 agent 位姿信息  ### np.asarray(...)是将Python 列表转为 NumPy 数组，得到一个 (num_scenes, 3) 的数组  # .from_numpy把Numpy数组转化为tensor
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)]) #infos 是一个列表或字典，记录了每个环境（env_idx）当前 timestep 的一些信息（比如位置、目标类别等）
    ).float().to(device) #

    _, local_map, _, local_pose = \    #构建/更新了 agent 的局部语义地图和局部位姿估计。 比如说初始位置是看到了什么，把东西放到地图里
        sem_map_module(obs, poses, local_map, local_pose) #obs: 当前时间步中 agent 的 RGB-D 图像观测 #poses: 每个环境中 agent 的当前位姿 #local_map: 当前的局部语义地图（上一帧） #当前的局部位姿（上一帧）

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)  #上面已经定义过了，这里又定义了一次？而且中间也没有对global_input和global_orientation的更改？
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution), 
                        int(c * 100.0 / args.map_resolution)] 

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1. #前面是在full_map上标记，现在是在local_map上标记
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.) #将角度（-180° 到 180°）变成 0 到 71 的整数（360° / 5 = 72 个离散方向）

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()       #局部语义地图 full_map 的前 4 个通道  ### .detach() 是避免梯度传播，因为地图不是网络要训练的参数
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(     #把局部语义地图 full_map 的前 4 个通道下采样
        full_map[:, 0:4, :, :])
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()     #剩下别的语义通道
    goal_cat_id = torch.from_numpy(np.asarray(        #提取每个环境当前的目标类别（如“椅子”、“冰箱”）用整数表示，goal_cat_id的shape为(num_scenes,)
        [infos[env_idx]['goal_cat_id'] for env_idx
         in range(num_scenes)]))

    extras = torch.zeros(num_scenes, 2)    #拼成一个 extras 张量，作为额外信息传给策略网络：[朝向，目标类别]
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = goal_cat_id

    g_rollouts.obs[0].copy_(global_input)  #将初始化好的 global_input 和 extras 存入 GlobalRolloutStorage 的第 0 步。  这样策略网络在第 1 步就能用这些输入做动作选择。
    g_rollouts.extras[0].copy_(extras)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = \。#g_value当前状态的值函数估计 g_action全局策略网络的输出的动作，形状为（num_scenes,2）但数值在[0,1)区间 #g_action_log_prob 输出动作的log概率，用于PPO算法计算loss
        g_policy.act(                    #g_rec_states RNN的下一个隐藏状态，存到下一timestep用
            g_rollouts.obs[0],  #当前timestep的输入语义图（多通道）
            g_rollouts.rec_states[0], #RNN的历史隐藏状态（用来保存记忆）
            g_rollouts.masks[0], #
            extras=g_rollouts.extras[0], #额外输入，包括朝向和物理类别ID
            deterministic=False #
        )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy() #全局策略网络输出的 g_action 可能是未经归一化的值，所以通过 Sigmoid() 将其压缩到 (0, 1) 区间
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]   #把归一化地图坐标转化为地图坐标
                    for action in cpu_actions]
    global_goals = [[min(x, int(local_w - 1)), min(y, int(local_h - 1))]   #确保目标坐标不会越界（坐标最大为地图边界）
                    for x, y in global_goals]        

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]  #创建一张大小为（local_w, local_h）的空白地图

    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1 #把目标点标记到地图上

    planner_inputs = [{} for e in range(num_scenes)] #为每个并行环境初始化一个空字典，后面要往里面填导航器需要的信息 ### planner_inputs 是一个列表，里面每一项是一个字典
    for e, p_input in enumerate(planner_inputs):  #
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy() #存入第0个通道（Obstacle Map）
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy() #存入第1个通道（explored area）
        p_input['pose_pred'] = planner_pose_inputs[e] #planner_pose_input的上面shape为（num_scenes, 7）的张量
        p_input['goal'] = goal_maps[e]  # global_goals[e]
        p_input['new_goal'] = 1  #1 表示刚刚设置的新目标点。
        p_input['found_goal'] = 0 #0 表示当前还未到达目标物体
        p_input['wait'] = wait_env[e] or finished[e] #如果当前环境被标记为 wait 或已经结束，则在这一时刻让 agent 停止规划行为
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()

    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs) #调用了环境的 plan_act_and_preprocess 方法，让 agent 根据前面设置的目标，执行一次决策动作并返回反馈。
                                                       #planner_inputs是我们上一轮准备的目标和地图，现在 agent 根据它来采取实际动作，然后环境给出反馈
						      #done是一个布尔列表，表示哪些环境已经完成（即 episode 结束）。
    start = time.time() #获取当前时间，用于之后计算某段代码执行时间（profiling 用）。
    g_reward = 0 #初始化全局奖励计数器，后面可以记录该时间步下所有 agent 的总奖励

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    if len(episode_success[e]) == num_episodes:
                        finished[e] = 1
                else:
                    episode_success.append(success)
                    episode_spl.append(spl)
                    episode_dist.append(dist)
                wait_env[e] = 1.
                update_intrinsic_rew(e)
                init_map_and_pose_for_env(e)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        _, local_map, _, local_pose = \
            sem_map_module(obs, poses, local_map, local_pose)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.
                else:
                    update_intrinsic_rew(e)

                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = \
                nn.MaxPool2d(args.global_downscaling)(
                    full_map[:, 0:4, :, :])
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
            goal_cat_id = torch.from_numpy(np.asarray(
                [infos[env_idx]['goal_cat_id'] for env_idx
                 in range(num_scenes)]))
            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id

            # Get exploration reward and metrics
            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx]['g_reward'] for env_idx in range(num_scenes)])
            ).float().to(device)
            g_reward += args.intrinsic_rew_coeff * intrinsic_rews.detach()

            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * \
                (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            # Add samples to global policy storage
            if step == 0:
                g_rollouts.obs[0].copy_(global_input)
                g_rollouts.extras[0].copy_(extras)
            else:
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, extras
                )

            # Sample long-term goal from global policy
            g_value, g_action, g_action_log_prob, g_rec_states = \
                g_policy.act(
                    g_rollouts.obs[g_step + 1],
                    g_rollouts.rec_states[g_step + 1],
                    g_rollouts.masks[g_step + 1],
                    extras=g_rollouts.extras[g_step + 1],
                    deterministic=False
                )
            cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
            global_goals = [[int(action[0] * local_w),
                             int(action[1] * local_h)]
                            for action in cpu_actions]
            global_goals = [[min(x, int(local_w - 1)),
                             min(y, int(local_h - 1))]
                            for x, y in global_goals]

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        for e in range(num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

        for e in range(num_scenes):
            cn = infos[e]['goal_cat_id'] + 4
            if local_map[e, cn, :, :].sum() != 0.:
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]  # global_goals[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                    :].argmax(0).cpu().numpy()

        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if g_step % args.num_global_steps == args.num_global_steps - 1 \
                and l_step == args.num_local_steps - 1:
            if not args.eval:
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.masks[-1],
                    extras=g_rollouts.extras[-1]
                ).detach()

                g_rollouts.compute_returns(g_next_value, args.use_gae,
                                           args.gamma, args.tau)
                g_value_loss, g_action_loss, g_dist_entropy = \
                    g_agent.update(g_rollouts)
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            g_rollouts.after_update()

        torch.set_grad_enabled(False)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
            ])

            log += "\n\tRewards:"

            if len(g_episode_rewards) > 0:
                log += " ".join([
                    " Global step mean/med rew:",
                    "{:.4f}/{:.4f},".format(
                        np.mean(per_step_g_rewards),
                        np.median(per_step_g_rewards)),
                    " Global eps mean/med/min/max eps rew:",
                    "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_episode_rewards),
                        np.median(g_episode_rewards),
                        np.min(g_episode_rewards),
                        np.max(g_episode_rewards))
                ])

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                for e in range(args.num_processes):
                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)

                if len(total_spl) > 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_dist),
                        len(total_spl))
            else:
                if len(episode_success) > 100:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(episode_success),
                        np.mean(episode_spl),
                        np.mean(episode_dist),
                        len(episode_spl))

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_value_losses),
                        np.mean(g_action_losses),
                        np.mean(g_dist_entropies))
                ])

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(g_episode_rewards) >= 1000 and \
                    (np.mean(g_episode_rewards) >= best_g_reward) \
                    and not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_g_reward = np.mean(g_episode_rewards)

        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_{}.pth".format(total_steps)))
        # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logging.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)


if __name__ == "__main__":
    main()
