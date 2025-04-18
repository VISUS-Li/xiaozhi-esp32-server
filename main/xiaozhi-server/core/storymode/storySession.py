from core.prompts import PromptManager
from core.utils import asr, vad, llm, tts, memory, intent
from config.module_config import get_module_config
from config.logger import setup_logging

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

        # 初始化不同阶段所需的 LLM 实例
        self._init_llm_instances()
        
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
        