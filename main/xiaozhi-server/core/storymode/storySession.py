from config.logger import setup_logging
from config.module_config import get_module_config
from core.prompts import PromptManager
import asyncio
import threading  # 添加threading模块导入

TAG = __name__
logger = setup_logging()

class StorySession:
    def __init__(self, conn, session_id):
        self.conn = conn
        self.session_id = session_id
        self.config = get_module_config(conn.config, "story-module")
        self.stage = "init"  # 初始阶段
        self.prompt_manager = PromptManager()
        self.dialogue_history = {}  # 为不同阶段存储对话历史
        self.outline_cache = {}
        self.next_sort = 0

        # 初始化不同阶段所需的 LLM 实例
        self._init_llm_instances()

        self.tts_stage_index = -1
        self.tts_stage_index_lock = asyncio.Lock()

        self.tts_text_index = -1
        self.tts_text_index_lock = threading.Lock()  # 修改为threading.Lock

        self.tts_stage_dict_lock = threading.Lock()  # 新增：用于保护tts_stage_dict的锁
        self.tts_stage_seg_count = {}  # 新增：记录每个stage的seg总数
        self.next_play_index = 0
        self.next_play_stage_index = 0  # 新增，stage顺序指针
        
        
    def _create_llm_instance(self, template_name, template_data):
        """根据模板创建LLM实例的公共方法"""
        llm_name = template_data.get("llm")
        
        if llm_name and llm_name in self.conn.config["LLM"]:
            # 创建 LLM 实例
            llm_config = self.conn.config["LLM"][llm_name].copy()
            
            # 添加模板中的LLM特定配置（仅添加模板中存在的配置）
            for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                if key in template_data:
                    llm_config[key] = template_data[key]
            
            # 专门处理token限制参数，仅在模板中明确指定时才添加
            if "max_tokens" in template_data:
                llm_config["max_tokens"] = template_data["max_tokens"]
            if "min_tokens" in template_data:
                llm_config["min_tokens"] = template_data["min_tokens"]
            
            llm_type = llm_config.get("type", llm_name)
            from core.utils import llm as llm_utils
            return llm_utils.create_instance(llm_type, llm_config)
        else:
            # 使用默认 LLM
            return self.conn.llm
        
    def _init_llm_instances(self):
        """初始化各阶段所需的 LLM 实例"""
        self.llm_instances = {}
        
        # 从 story-prompts.yaml 读取各阶段所需的 LLM 配置
        for template_name, template_data in self.prompt_manager.templates_data.items():
            # 检查模板是否属于故事模式
            if template_data.get("type") == "story":
                self.llm_instances[template_name] = self._create_llm_instance(
                    template_name, template_data
                )
                    
        # 记录初始化的LLM实例
        logger.bind(tag=TAG).info(f"故事模式初始化了 {len(self.llm_instances)} 个LLM实例: {list(self.llm_instances.keys())}")
        
    def get_llm(self, stage):
        """获取指定阶段的 LLM 实例"""
        # 首先尝试直接用阶段名称获取
        if stage in self.llm_instances:
            return self.llm_instances[stage]
        
        # 如果没有找到，尝试查找带有阶段名称前缀的模板
        stage_prefix = f"{stage}_"
        for template_name, instance in self.llm_instances.items():
            if template_name.startswith(stage_prefix):
                return instance
        
        # 最后尝试在模板数据中查找对应阶段
        for template_name, template_data in self.prompt_manager.templates_data.items():
            if template_data.get("type") == "story" and template_name.startswith(stage):
                # 如果找到匹配的模板但没有对应的LLM实例，现在创建一个
                self.llm_instances[template_name] = self._create_llm_instance(
                    template_name, template_data
                )
                return self.llm_instances[template_name]
        
        # 如果所有尝试都失败，返回默认LLM
        logger.bind(tag=TAG).warning(f"未找到阶段 '{stage}' 对应的LLM实例，使用默认LLM")
        return self.conn.llm
    
    def get_dialogue(self, stage):
        """获取指定阶段的对话历史"""
        if stage not in self.dialogue_history:
            # 初始化对话历史，只包含系统提示
            from core.utils.dialogue import Dialogue, Message
            dialogue = Dialogue()
            
            # 直接从对应的stage中加载初始化的提示词
            prompt_template = self.prompt_manager.get_template(stage)
            dialogue.put(Message(role="system", content=prompt_template.formatted_prompt))
            self.dialogue_history[stage] = dialogue
            
        return self.dialogue_history[stage]
    
    def update_stage(self, new_stage):
        """更新当前阶段"""
        self.stage = new_stage
        return self.stage

    async def incr_stage_index(self):
        """更新 TTS 索引"""
        async with self.tts_stage_index_lock:
            self.tts_stage_index += 1
            return self.tts_stage_index

    def incr_text_index(self):
        """更新 TTS 文本索引"""
        with self.tts_text_index_lock:
            self.tts_text_index += 1
            return self.tts_text_index


    def set_stage_seg_count(self, stage, seg_count):
        """设置每个stage的seg总数"""
        stage_key = str(stage)
        with self.tts_stage_dict_lock:
            self.tts_stage_seg_count[stage_key] = seg_count

    def get_stage_seg_count(self, stage):
        stage_key = str(stage)
        with self.tts_stage_dict_lock:
            return self.tts_stage_seg_count.get(stage_key, None)
