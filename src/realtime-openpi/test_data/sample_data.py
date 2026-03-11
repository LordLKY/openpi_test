import os
import pathlib
import numpy as np
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def save_libero_observation(  
    task_suite_name: str = "libero_spatial",
    task_id: int = 0,
    resize_size: int = 224,
    output_dir: str = "data/libero/observations"
):
    """
    读取LIBERO数据集，获取一组观察图像和prompt，并保存到文件中
    
    参数:
    - task_suite_name: 任务套件名称，可选值: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    - task_id: 任务ID (在任务套件中的索引)
    - resize_size: 图像调整后的大小
    - output_dir: 输出文件目录
    """
    # 创建输出目录
    # output_path = pathlib.Path(output_dir)
    # output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir
    print(f"输出目录: {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    try:
        # 初始化LIBERO任务套件
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        
        if task_id >= task_suite.n_tasks:
            raise ValueError(f"Task ID {task_id} is out of range for suite {task_suite_name} (0-{task_suite.n_tasks-1})")
        
        logger.info(f"任务套件: {task_suite_name}")
        logger.info(f"任务总数: {task_suite.n_tasks}")
        
        # 获取指定任务
        task = task_suite.get_task(task_id)
        task_description = task.language
        logger.info(f"当前任务: {task_description}")
        
        # 初始化LIBERO环境
        # env_args = {
        #     "bddl_file_name": task.bddl_file,
        #     "camera_heights": 256,  # 匹配训练数据的分辨率
        #     "camera_widths": 256
        # }
        # env = OffScreenRenderEnv(**env_args)
        env, task_description = _get_libero_env(task, 256, 7)
        
        # 重置环境获取初始观察
        logger.info("重置环境...")
        obs = env.reset()
        
        # 获取并处理图像
        logger.info("处理图像...")
        
        # 主视图图像 (agentview_image)
        main_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # 旋转180度
        main_img_resized = image_tools.resize_with_pad(main_img, resize_size, resize_size)
        main_img_uint8 = image_tools.convert_to_uint8(main_img_resized)
        
        # 手腕视角图像 (eye-in-hand)
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])  # 旋转180度
        wrist_img_resized = image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
        wrist_img_uint8 = image_tools.convert_to_uint8(wrist_img_resized)
        
        # 保存原始图像
        raw_main_path = os.path.join(output_path, f"task_{task_id:02d}_raw_main.png")
        raw_wrist_path = os.path.join(output_path, f"task_{task_id:02d}_raw_wrist.png")
        imageio.imwrite(raw_main_path, np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        imageio.imwrite(raw_wrist_path, np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        
        # 保存调整大小后的图像
        resized_main_path = os.path.join(output_path, f"task_{task_id:02d}_resized_main.png")
        resized_wrist_path = os.path.join(output_path, f"task_{task_id:02d}_resized_wrist.png")
        imageio.imwrite(resized_main_path, main_img_uint8)
        imageio.imwrite(resized_wrist_path, wrist_img_uint8)
        
        # 保存prompt
        prompt_path = os.path.join(output_path, f"task_{task_id:02d}_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(task_description)
        
        # 保存观察状态信息
        state_info = {
            "task_description": task_description,
            "task_id": task_id,
            "task_suite": task_suite_name,
            "eef_pos": obs["robot0_eef_pos"].tolist(),
            "eef_quat": obs["robot0_eef_quat"].tolist(),
            "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
            "main_image_shape": main_img.shape,
            "wrist_image_shape": wrist_img.shape,
            "resized_image_shape": main_img_uint8.shape
        }
        
        state_path = os.path.join(output_path, f"task_{task_id:02d}_state.npy")
        np.save(state_path, state_info, allow_pickle=True)
        
        logger.info(f"保存完成!")
        logger.info(f"原始主视图图像: {raw_main_path}")
        logger.info(f"原始手腕图像: {raw_wrist_path}")
        logger.info(f"调整大小后主视图图像: {resized_main_path}")
        logger.info(f"调整大小后手腕图像: {resized_wrist_path}")
        logger.info(f"Prompt: {prompt_path}")
        logger.info(f"状态信息: {state_path}")
        
        return {
            "main_image": main_img_uint8,
            "wrist_image": wrist_img_uint8,
            "prompt": task_description,
            "state": {
                "eef_pos": obs["robot0_eef_pos"],
                "eef_quat": obs["robot0_eef_quat"],
                "gripper_qpos": obs["robot0_gripper_qpos"]
            },
            "files_saved": {
                "raw_main": str(raw_main_path),
                "raw_wrist": str(raw_wrist_path),
                "resized_main": str(resized_main_path),
                "resized_wrist": str(resized_wrist_path),
                "prompt": str(prompt_path),
                "state": str(state_path)
            }
        }
        
    except Exception as e:
        logger.error(f"发生错误: {e}")
        raise
    finally:
        # 关闭环境
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    # 示例用法
    result = save_libero_observation(
        task_suite_name="libero_spatial",
        task_id=0,
        resize_size=224,
        output_dir=os.path.join(BASE_DIR, "sampled_obs") #  "data/libero/observations"
    )
    print("\n获取的观察信息:")
    print(f"Prompt: {result['prompt']}")
    print(f"主视图图像形状: {result['main_image'].shape}")
    print(f"手腕图像形状: {result['wrist_image'].shape}")
    print(f"保存的文件:")
    for key, path in result['files_saved'].items():
        print(f"- {key}: {path}")
