import asyncio
import json
import os
import datetime  # 添加datetime模块用于生成时间戳
from concurrent.futures import Future

from loguru import logger

from config.logger import setup_logging
from core.handle.sendAudioHandle import send_stt_message
from core.storymode.storyRespProcessor import StoryRespProcessor
from core.storymode.storySession import StorySession
from core.utils.dialogue import Message

TAG = __name__
logger = setup_logging()

async def handle_story_mode(conn, text):
    """处理故事模式请求"""
    # 进入故事模式
    return await enter_story_mode(conn, text)


async def enter_story_mode(conn, text):
    """进入故事模式"""
    logger.bind(tag=TAG).info("用户进入故事模式")

    # 创建故事会话
    if not hasattr(conn, 'story_session'):
        conn.story_session = StorySession(conn, conn.session_id)
    
    # 确保conn具有tts_index_lock属性
    if not hasattr(conn.story_session, 'tts_index_lock'):
        conn.story_session.tts_stage_index_lock = asyncio.Lock()
        
    
    # 发送用户输入的文本
    await send_stt_message(conn, text)
    
    # 记录对话
    conn.dialogue.put(Message(role="user", content=text))
    
    # 初始化提示词管理器
    prompt_manager = conn.story_session.prompt_manager
    
    # 获取故事模式的初始提示词
    initial_prompt = prompt_manager.get_template("story_mode_intro")
    
    # 记录对话内容，但使用预先生成的语音文件进行回复
    if initial_prompt.formatted_prompt is not None or initial_prompt.formatted_prompt != "":
        conn.dialogue.put(Message(role="assistant", content=initial_prompt.formatted_prompt))
    
    # 创建欢迎语段落
    conn.recode_first_last_text(initial_prompt.template, 0) # 这是第一条语音
    
    # 使用预先生成的语音文件代替TTS生成
    story_start_file = "tmp/nailong-start.mp3"
    if os.path.exists(story_start_file) and os.path.isfile(story_start_file):
        # 创建future
        completed_future = Future()
        completed_future.set_result((story_start_file, initial_prompt.template, 0))
        # 将音频放入播放队列
        conn.tts_queue.put(completed_future)
        logger.bind(tag=TAG).info("使用预生成语音文件进行故事模式初始反馈")
    else:
        # 如果预生成文件不存在，回退到实时TTS生成
        logger.bind(tag=TAG).warning(f"预生成语音文件 {story_start_file} 不存在，使用实时TTS生成")
        future = conn.executor.submit(conn.speak_and_play, initial_prompt.template, 0)
        conn.tts_queue.put(future)

    # 更新故事阶段
    conn.story_session.update_stage("outline_generation")

    # 直接用协程调度后续任务
    # 提取故事主题 -> 生成大纲 -> 续写
    async def story_mode_main():
        try:
            theme_data = await extract_story_theme(conn, text)
            conn.story_session.theme_data = theme_data
            await generate_story_outline(conn, theme_data)
        except Exception as e:
            logger.bind(tag=TAG).error(f"故事模式任务执行出错: {str(e)}")

    # 启动后台任务
    asyncio.create_task(story_mode_main())

    return True


async def save_llm_response_to_file(conn, stage, prompt, user_prompt, response):
    """将LLM响应保存到文件
    
    Args:
        conn: 连接对象，包含session_id
        stage: 当前阶段名称
        prompt: 提示词模板的格式化内容
        user_prompt: 用户提供的提示内容
        response: LLM返回的完整响应
        
    Returns:
        filename: 保存的文件名
    """
    try:
        # 创建logs目录和llm_responses子目录
        log_dir = os.path.join("logs", "llm_responses")
        os.makedirs(log_dir, exist_ok=True)
        
        # 获取session_id
        session_id = conn.session_id
        
        # 使用session_id和stage命名文件
        filename = f"{session_id}_{stage}.json"
        filepath = os.path.join(log_dir, filename)

        # 处理response对象，确保能够正确序列化为JSON
        if hasattr(response, 'model_dump'):
            # 如果是Pydantic模型，使用model_dump()方法转为字典
            resp_dict = response.model_dump()
        elif isinstance(response, dict):
            # 如果是AttrDict或普通字典，转换为标准字典
            resp_dict = dict(response)
        elif isinstance(response, str):
            # 如果是字符串，尝试解析为JSON对象
            try:
                resp_dict = json.loads(response)
            except json.JSONDecodeError:
                # 如果不是有效的JSON字符串，则保留原始字符串
                resp_dict = response
        else:
            # 其他类型尝试直接转换
            resp_dict = str(response)
        
        # 处理user_prompt对象，确保正确序列化为JSON
        if user_prompt and isinstance(user_prompt, str):
            # 如果是字符串，尝试解析为JSON对象
            try:
                user_prompt = json.loads(user_prompt)
            except json.JSONDecodeError:
                # 如果不是有效的JSON字符串，保留原始值
                pass

        # 新增：递归解析 user_prompt 中 before 列表的 content 字段和 outline 字段
        if isinstance(user_prompt, dict):
            # 解析 before 列表中的 content
            if "before" in user_prompt and isinstance(user_prompt["before"], list):
                for item in user_prompt["before"]:
                    if isinstance(item, dict) and "content" in item and isinstance(item["content"], str):
                        try:
                            item["content"] = json.loads(item["content"])
                        except Exception:
                            pass
            # 解析 outline 字段
            if "outline" in user_prompt and isinstance(user_prompt["outline"], str):
                try:
                    user_prompt["outline"] = json.loads(user_prompt["outline"])
                except Exception:
                    pass
        # 生成当前记录的时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建保存内容
        save_data = {
            "timestamp": timestamp,
            "stage": stage,
            "system_prompt": prompt,
            "user_prompt": user_prompt,
            "response": resp_dict
        }
        
        # 检查文件是否已存在
        file_data = []
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                # 读取现有文件内容
                with open(filepath, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    # 确保file_data是列表
                    if not isinstance(file_data, list):
                        file_data = [file_data]
            except json.JSONDecodeError:
                # 如果文件格式不正确，创建新文件
                logger.bind(tag=TAG).warning(f"文件 {filepath} 格式不正确，将创建新文件")
                file_data = []
        
        # 将新数据添加到列表中
        file_data.append(save_data)
        
        # 写入文件（覆盖原有内容）
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)
            
        logger.bind(tag=TAG).debug(f"已将LLM响应追加到文件: {filepath}")
        return filename
    
    except Exception as e:
        logger.bind(tag=TAG).error(f"保存LLM响应到文件时出错: {str(e)}")
        return None

async def call_llm_with_template(conn, stage, user_prompt):
    """封装LLM调用的通用函数
    
    Args:
        conn: 连接对象
        stage: 当前阶段名称
        user_prompt: 用户提示内容
        
    Returns:
        complete_response: LLM返回的完整回复
    """
    try:
        # 获取当前阶段的LLM
        _llm = conn.story_session.get_llm(stage)
        
        # 获取提示词模板
        prompt_manager = conn.story_session.prompt_manager
        prompt_template = prompt_manager.get_template(stage)

        if user_prompt:
            if isinstance(user_prompt, str):
                pass
            elif hasattr(user_prompt, 'model_dump_json'):
                # Pydantic模型
                user_prompt = user_prompt.model_dump_json()
            elif isinstance(user_prompt, dict):
                # 字典类型
                user_prompt = json.dumps(user_prompt, ensure_ascii=False)
            else:
                # 其他类型转字符串
                user_prompt = str(user_prompt)
        else:
            user_prompt = ""

        # 异步调用LLM
        loop = asyncio.get_running_loop()
        complete_response = await loop.run_in_executor(
            None,
            _llm.response_no_stream,
            prompt_template.formatted_prompt,
            user_prompt
        )
        
        story_data = prompt_template.parse(complete_response, True)
        if isinstance(story_data, str):
            pass
        elif hasattr(story_data, 'model_dump_json'):
            # Pydantic模型
            story_data = story_data.model_dump_json()
        elif isinstance(story_data, dict):
            # 字典类型
            story_data = json.dumps(story_data, ensure_ascii=False)
        else:
            # 其他类型转字符串
            story_data = str(story_data)
        # 保存LLM响应到文件
        await save_llm_response_to_file(
            conn,
            stage, 
            prompt_template.formatted_prompt, 
            user_prompt, 
            story_data
        )
        
        return story_data, prompt_template
    
    except Exception as e:
        logger.bind(tag=TAG).error(f"调用LLM时出错 (阶段:{stage}): {str(e)}")
        raise

async def process_text_in_background(conn, stage, complete_response, stage_index=None, next_stage_func=None):
    """后台处理文本的通用函数
    
    Args:
        conn: 连接对象
        complete_response: LLM返回的完整响应
        stage_index: 开始的文本索引
        next_stage_func: 下一阶段要执行的函数
    """
    try:
        # 创建处理器实例
        processor = StoryRespProcessor(conn)
        
        # 确定开始索引
        if stage_index is None:
            # 使用锁安全获取下一个索引
            stage_index = await conn.story_session.incr_stage_index()
        
        # 如果有下一阶段处理函数，立即启动它而不等待文本处理完成
        if next_stage_func:
            # 创建一个新任务来执行下一阶段，让它与文本处理并行执行
            asyncio.create_task(next_stage_func(conn))
        
        # 处理文本（这会在后台继续执行，不会阻塞next_stage_func）
        await processor.process_complete_text(
            stage,
            complete_response,
            is_json_mode=True,
            stage_index=stage_index
        )
            
    except Exception as e:
        logger.bind(tag=TAG).error(f"后台处理文本时出错: {e}")


async def generate_story_outline(conn, theme_data=None):
    """生成故事大纲并缓存 - 使用非流式方式"""
    stage = "outline_generation"
    try:
        # 获取大纲生成阶段的对话历史
        dialogue = conn.story_session.get_dialogue(stage)
        # 如果提供了额外数据，构建输入提示
        # 调用LLM生成内容
        complete_response, prompt_template = await call_llm_with_template(conn, stage, theme_data)
        
        # 更新对话历史
        dialogue.update_system_message(prompt_template.formatted_prompt)
        if theme_data:
            dialogue.put(Message(role="user", content=theme_data))
        dialogue.put(Message(role="assistant", content=complete_response))
        conn.story_session.outline_cache = complete_response
        # 更新故事阶段为交互阶段
        conn.story_session.update_stage("story_continuation")

        # 获取下一个安全的文本索引
        next_text_index = await conn.story_session.incr_stage_index()
        
        # 创建后台任务处理文本，并准备故事续写
        asyncio.create_task(process_text_in_background(
            conn, 
            stage,
            complete_response, 
            next_text_index,
            prepare_story_continuation  # 处理完文本后，启动故事续写任务
        ))
        
        return True
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"生成故事大纲时出错: {str(e)}")
        
        # 通知用户出错
        error_message = "很抱歉，故事创作过程中出现了问题。让我们下次再试吧。"
        text_index = await conn.story_session.incr_stage_index()
        conn.recode_first_last_text(error_message, text_index)
        future = conn.executor.submit(conn.speak_and_play, error_message, text_index)
        conn.tts_queue.put(future)
        return False


async def check_story_mode_keywords(text):
    """检查文本中是否包含故事模式的触发关键词"""
    # 故事模式的触发关键词
    story_triggers = ["讲个故事", "开始故事", "故事模式", "讲故事", "讲一个故事", 
                     "说个故事", "故事时间", "开启故事模式"]
    
    # 检查文本是否包含触发词
    for trigger in story_triggers:
        if trigger in text:
            return True
    
    return False 


async def get_next_sort(complete_response, prompt_template):
    """检查故事是否结束，并获取下一个sort值"""
    try:
        story_data = prompt_template.parse(complete_response, True)
        
        # 默认故事未结束和下一个sort为0
        story_ended = False
        next_sort = 0
        
        # 从story_data获取ended属性
        if hasattr(story_data, "ended"):
            story_ended = story_data.ended
        
        # 从story_seg中查找最大的sort值
        if hasattr(story_data, "story_seg") and story_data.story_seg:
            # 初始化最大sort值
            max_sort = -1
            
            # 遍历story_seg数组查找最大sort值
            for seg in story_data.story_seg:
                if hasattr(seg, "sort"):
                    max_sort = max(max_sort, seg.sort)
                elif isinstance(seg, dict) and "sort" in seg:
                    max_sort = max(max_sort, seg["sort"])
            
            # 如果找到了有效的sort值，则next_sort为max_sort+1
            if max_sort >= 0:
                next_sort = max_sort + 1
        
        return story_ended, next_sort
    except Exception as e:
        logger.bind(tag=TAG).error(f"获取next_sort时出错: {str(e)}")
        return getattr(complete_response, "ended", False), getattr(complete_response, "next_sort", 0)

async def prepare_story_continuation(conn):
    """预先准备故事续写内容 - 使用非流式方式"""
    try:
        stage = "story_continuation"
        # 更新故事阶段
        conn.story_session.update_stage(stage)
        
        # 构建输入数据
        dialogue = conn.story_session.get_dialogue(stage)
        before = dialogue.get_last_n_message_without_system(5)
        user_prompt = {
            "outline": conn.story_session.outline_cache,
            "before": before,
            "next_sort": conn.story_session.next_sort
        }
        
        # 调用LLM生成内容
        complete_response, prompt_template = await call_llm_with_template(conn, stage, user_prompt)
        
        # 更新对话历史
        dialogue.update_system_message(prompt_template.formatted_prompt)
        dialogue.put(Message(role="assistant", content=complete_response))
        
        # 检查故事是否结束
        story_ended, next_sort = await get_next_sort(complete_response, prompt_template)
        conn.story_session.next_sort = next_sort
        
        # 获取下一个安全的文本索引
        next_text_index = await conn.story_session.incr_stage_index()
        
        # 在后台处理文本，如果故事未结束则继续准备下一段
        next_stage = None if story_ended else prepare_story_continuation
        asyncio.create_task(process_text_in_background(
            conn, 
            stage,
            complete_response, 
            next_text_index,
            next_stage
        ))
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"预准备故事续写内容时出错: {str(e)}")


async def extract_story_theme(conn, text):
    """从用户输入中提取故事主题，并结合用户信息"""
    default_theme = {
        "theme": "一般故事",
        "age_group": "通用",
        "style": "温馨",
        "has_explicit_theme": False
    }
    try:
        stage = "story_theme_extraction"
        
        # 构建用户信息
        user_info = {}
        
        # 如果有用户记忆，可以添加用户信息
        if hasattr(conn, 'memory') and conn.memory:
            memory_str = await conn.memory.query_memory(text)
            if memory_str:
                user_info["memory"] = memory_str
        

        if check_story_mode_keywords(text):
            return None
        
        # 添加对话历史中的关键信息
        user_info["dialogue_history"] = conn.dialogue.get_last_n_message_without_system(5)
        
        # 构建输入数据
        extra_data = {
            "user_info": user_info,
            "text": text
        }
        
        # 调用LLM提取主题
        theme_result, template_data = await call_llm_with_template(conn, stage, extra_data)
        
        # 解析结果
        try:
            # 尝试使用解析器解析结果
            theme_data = template_data.parse(theme_result)
            return theme_data
        except Exception as parse_error:
            logger.bind(tag=TAG).error(f"解析主题数据时出错: {str(parse_error)}")
            return theme_result
            
    except Exception as e:
        logger.bind(tag=TAG).error(f"提取故事主题时出错: {str(e)}")
        # 返回默认主题
        return json.dumps(default_theme, ensure_ascii=False)
