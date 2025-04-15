import os
import yaml
from typing import Dict, Any, Optional
from core.utils.util import get_project_dir, read_config

def load_module_config(module_name: str) -> Optional[Dict[str, Any]]:
    """
    加载模块特定的配置文件
    
    Args:
        module_name: 模块名称，例如 'story_mode'
        
    Returns:
        Dict 或 None: 如果配置文件存在，返回配置字典；否则返回 None
    """
    config_path = os.path.join(get_project_dir(), f'{module_name}')
    if os.path.exists(config_path):
        return read_config(config_path)
    return None

def get_module_config(main_config: Dict[str, Any], module_name: str) -> Dict[str, Any]:
    """
    获取模块配置，优先从单独的配置文件加载，如果不存在则从主配置中获取
    
    Args:
        main_config: 主配置对象
        module_name: 模块名称
        
    Returns:
        Dict: 模块配置字典，如果不存在则返回空字典
    """
    # 尝试从单独的配置文件加载
    module_config = load_module_config(module_name)
    
    # 如果单独配置文件不存在，则从主配置中获取
    if module_config is None:
        module_config = main_config.get('modules', {}).get(module_name, {})
    
    return module_config

def merge_configs(main_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并主配置和所有模块配置
    
    Args:
        main_config: 主配置对象
        
    Returns:
        Dict: 合并后的配置字典
    """
    # 创建配置副本，避免修改原始配置
    config = main_config.copy()
    
    # 初始化modules字典（如果不存在）
    if 'modules' not in config:
        config['modules'] = {}
    
    # 扫描根目录下的模块配置文件
    root_dir = get_project_dir()
    module_files = []
    for file in os.listdir(root_dir):
        if file.endswith(('-module.yaml', '-module.yml')):
            module_files.append(file)
    
    # 处理各模块的配置
    for module_file in module_files:
        module_config = load_module_config(module_file)
        if module_config:
            module_name = module_file.split('.')[0]
            config['modules'][module_name] = module_config
    
    return config