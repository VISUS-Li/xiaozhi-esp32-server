from config.logger import setup_logging
import json
from core.handle.sendAudioHandle import send_stt_message
from core.storymode.storySession import StorySession
from core.utils.dialogue import Message
from loguru import logger
import asyncio
from core.prompts import PromptManager
import re
from core.storymode.storyRespProcessor import StoryRespProcessor
import os
import threading
import concurrent.futures

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
    
    # 发送用户输入的文本
    await send_stt_message(conn, text)
    
    # 记录对话
    conn.dialogue.put(Message(role="user", content=text))
    
    # 初始化提示词管理器
    prompt_manager = PromptManager()
    
    # 获取故事模式的初始提示词
    initial_prompt = prompt_manager.get_template("story_mode_intro")
    
    # 记录对话内容，但使用预先生成的语音文件进行回复
    if initial_prompt.formatted_prompt is not None or initial_prompt.formatted_prompt != "":
        conn.dialogue.put(Message(role="assistant", content=initial_prompt.formatted_prompt))
    
    # 获取当前文本索引并记录文本内容
    text_index = conn.tts_last_text_index + 1 if hasattr(conn, 'tts_last_text_index') else 0
    conn.recode_first_last_text(initial_prompt.template, text_index)
    
    # 使用预先生成的语音文件代替TTS生成
    story_start_file = "tmp/nailong-start.mp3"
    if os.path.exists(story_start_file) and os.path.isfile(story_start_file):
        # 将音频文件转换为opus格式
        opus_packets, duration = conn.tts.audio_to_opus_data(story_start_file)
        # 将音频数据放入播放队列
        conn.audio_play_queue.put((opus_packets, initial_prompt.template, text_index))
        logger.bind(tag=TAG).info("使用预生成语音文件进行故事模式初始反馈")
    else:
        # 如果预生成文件不存在，回退到实时TTS生成
        logger.bind(tag=TAG).warning(f"预生成语音文件 {story_start_file} 不存在，使用实时TTS生成")
        future = conn.executor.submit(conn.speak_and_play, initial_prompt.template, text_index)
        conn.tts_queue.put(future)
    
    # 更新故事阶段
    conn.story_session.update_stage("outline_generation")

    # 在单独的线程中提取故事主题并启动后续任务
    def run_story_tasks():
        # 使用asyncio事件循环运行协程任务
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)

        try:
            # 提取故事主题
            theme_data = asyncio_loop.run_until_complete(extract_story_theme(conn, text))
            conn.story_session.theme_data = theme_data

            # 启动故事大纲生成任务
            asyncio_loop.run_until_complete(generate_story_outline(conn, theme_data))
        except Exception as e:
            logger.bind(tag=TAG).error(f"故事模式任务执行出错: {str(e)}")
        finally:
            asyncio_loop.close()

    # 启动单独的线程运行故事任务
    story_thread = threading.Thread(target=run_story_tasks, daemon=True)
    story_thread.start()

    return True


async def generate_story_outline(conn, theme_data=None):
    """生成故事大纲并缓存 - 使用非流式方式"""
    stage = "outline_generation"
    try:
        # 获取大纲生成阶段的 LLM
        _llm = conn.story_session.get_llm(stage)
        
        # 获取大纲生成阶段的对话历史
        dialogue = conn.story_session.get_dialogue(stage)

        # 使用提示词管理器获取提示词模板
        prompt_manager = conn.story_session.prompt_manager
        prompt_template = prompt_manager.get_template(stage)

        # 更新对话历史：系统提示放在system消息中，主题数据放在user消息中
        dialogue.update_system_message(prompt_template.formatted_prompt)
        # 如果有主题数据，添加到提示词中
        if theme_data:
            dialogue.put(Message(role="user", content=theme_data))
        
        # 创建缓存结构
        conn.story_session.outline_cache = {
            "completed": False,
            "content": "",
            "tts_chunks": []
        }
        
        complete_response = _llm.response_no_stream(
            system_prompt=prompt_template.formatted_prompt,
            user_prompt=theme_data if theme_data else ""
        )
        
        logger.bind(tag=TAG).info("故事大纲生成完成，开始处理")
        
        # 创建处理器实例
        processor = StoryRespProcessor(conn)
        
        # 更新缓存内容
        conn.story_session.outline_cache["content"] = complete_response
        conn.story_session.outline_cache["completed"] = True
        
        # 处理生成的内容
        await processor.process_complete_text(
            complete_response,
            is_json_mode=True,
            start_text_index=conn.tts_last_text_index + 1
        )
        
        # 缓存TTS索引
        conn.story_session.outline_cache["tts_chunks"] = list(range(
            conn.tts_first_text_index,
            conn.tts_last_text_index + 1
        ))
        
        # 记录助手回复
        dialogue.put(Message(role="assistant", content=complete_response))
        conn.dialogue.put(Message(role="assistant", content=complete_response))
        
        # 更新故事阶段为交互阶段
        conn.story_session.update_stage("story_continuation")
        
        # 启动预缓存故事续写内容的任务
        await prepare_story_continuation(conn)
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"生成故事大纲时出错: {str(e)}")
        
        # 通知用户出错
        error_message = "很抱歉，故事创作过程中出现了问题。让我们下次再试吧。"
        text_index = conn.tts_last_text_index + 1 if hasattr(conn, 'tts_last_text_index') else 0
        conn.recode_first_last_text(error_message, text_index)
        future = conn.executor.submit(conn.speak_and_play, error_message, text_index)
        conn.tts_queue.put(future)


async def check_story_mode_keywords(conn, text):
    """检查文本中是否包含故事模式的触发关键词"""
    # 故事模式的触发关键词
    story_triggers = ["讲个故事", "开始故事", "故事模式", "讲故事", "讲一个故事", 
                     "说个故事", "故事时间", "开启故事模式"]
    
    # 检查文本是否包含触发词
    for trigger in story_triggers:
        if trigger in text:
            return True
    
    return False 


async def prepare_story_continuation(conn):
    """预先准备故事续写内容 - 使用非流式方式"""
    try:
        # 获取故事续写阶段的 LLM
        _llm = conn.story_session.get_llm("story_continuation")
        
        # 获取故事续写阶段的对话历史
        dialogue = conn.story_session.get_dialogue("story_continuation")
        
        # 获取大纲内容
        outline = conn.story_session.outline_cache.get("content", "")
        
        # 创建续写缓存结构
        conn.story_session.continuation_cache = {
            "generating": True,
            "completed": False,
            "content": "",
            "tts_chunks": [],
            "current_position": 0,  # 当前播放位置
            "is_paused": False,     # 是否暂停
            "is_interrupted": False # 是否被用户中断
        }
        
        # 使用提示词管理器获取续写提示模板
        prompt_manager = conn.story_session.prompt_manager
        prompt_template = prompt_manager.get_template("story_continuation")
        
        # 更新系统提示
        dialogue.update_system_message(prompt_template.formatted_prompt)
        input_prompt = prompt_template.get_input_prompt({
            "outline": outline,
            "before": conn.story_session.outline_cache.get("content", "")
        })

        # 更新用户消息
        dialogue.put(Message(role="user", content=input_prompt))
        
        # 非流式方式调用大模型
        logger.bind(tag=TAG).info("开始非流式生成故事续写内容")
        complete_response = _llm.response_no_stream(
            system_prompt=prompt_template.formatted_prompt,
            user_prompt=input_prompt
        )
        
        logger.bind(tag=TAG).info("故事续写内容生成完成，开始处理")

        # 创建处理器实例
        processor = StoryRespProcessor(conn)
        
        # 更新缓存内容
        conn.story_session.continuation_cache["content"] = complete_response
        conn.story_session.continuation_cache["generating"] = False
        conn.story_session.continuation_cache["completed"] = True
        
        # 处理生成的内容
        await processor.process_complete_text(
            complete_response,
            is_json_mode=True,
            start_text_index=conn.tts_last_text_index + 1
        )
        
        # 记录助手回复
        dialogue.put(Message(role="assistant", content=complete_response))
        conn.dialogue.put(Message(role="assistant", content=complete_response))
        
        # 更新故事阶段为交互阶段
        conn.story_session.update_stage("story_continuation")
        
        # 检查故事是否已经结束
        try:
            # 尝试解析JSON以检查ended标志
            story_data = prompt_template.parse(complete_response)
            ended = getattr(story_data, "ended", False) if hasattr(story_data, "ended") else story_data.get("ended", False)
            
            if ended:
                logger.bind(tag=TAG).info("故事已经结束，不再准备后续内容")
                return
        except Exception as e:
            logger.bind(tag=TAG).error(f"检查故事结束状态时出错: {e}")
        
        # 只有故事未结束时，才准备下一段内容
        # 在单独的协程中启动下一个续写
        asyncio.create_task(prepare_story_continuation(conn))
        logger.bind(tag=TAG).info("故事续写内容已处理完毕")
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"预准备故事续写内容时出错: {str(e)}")
        # 标记缓存生成失败
        if hasattr(conn.story_session, 'continuation_cache'):
            conn.story_session.continuation_cache["generating"] = False
            conn.story_session.continuation_cache["completed"] = False

async def extract_story_theme(conn, text):
    """从用户输入中提取故事主题，并结合用户信息"""
    try:
        # 获取内容判断LLM
        _llm = conn.story_session.get_llm("story_theme_extraction")
        
        # 构建提示词
        user_info = {}
        
        # 如果有用户记忆，可以添加用户信息
        if hasattr(conn, 'memory') and conn.memory:
            memory_str = await conn.memory.query_memory(text)
            if memory_str:
                user_info["memory"] = memory_str
        
        # 添加对话历史中的关键信息
        user_info["dialogue_history"] = conn.dialogue.get_last_n_message_without_system(5)
        
        # 获取提示词模板和JSON Schema
        prompt_manager = conn.story_session.prompt_manager
        template_data = prompt_manager.get_template("story_theme_extraction")
        input_prompt = template_data.get_input_prompt({
            "user_info": user_info,
            "text": text
        })
        
        # 调用LLM分析
        theme_result = _llm.response_no_stream(
            system_prompt=template_data.formatted_prompt,
            user_prompt=input_prompt
        )

        default_theme = {
            "theme": "一般故事",
            "age_group": "通用", 
            "style": "温馨",
            "has_explicit_theme": False
        }
        
        # 解析结果
        try:
            # 尝试使用解析器解析结果
            theme_data = template_data.parse(theme_result)
            # 如果是Pydantic模型，转换为字符串
            if hasattr(theme_data, "model_dump"):
                return json.dumps(theme_data.model_dump(), ensure_ascii=False)
            return json.dumps(theme_data, ensure_ascii=False)
        except Exception as parse_error:
            logger.bind(tag=TAG).error(f"解析主题数据时出错: {str(parse_error)}")
            # 如果解析失败，尝试直接返回原始结果
            if isinstance(theme_result, str):
                return theme_result
            
            # 默认返回一般主题
            return json.dumps(default_theme, ensure_ascii=False)
            
    except Exception as e:
        logger.bind(tag=TAG).error(f"提取故事主题时出错: {str(e)}")
        # 返回默认主题
        return json.dumps(default_theme, ensure_ascii=False)
