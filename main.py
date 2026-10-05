"""TedaBot 入口：加载配置 → 初始化组件 → 主循环（watchdog 自动重启）。"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

from bot.ai_backend import ContextManager
from bot.handler import MessageHandler, scenario_key
from bot.logger import setup_logging
from bot.send_queue import SendQueue, PRIORITY_AT, PRIORITY_NORMAL, PRIORITY_PROACTIVE
from bot.wechat_client import WeChatClient

POLL_INTERVAL = 0.5        # 消息轮询基础间隔（秒，实际=间隔-本轮耗时）
RESTART_DELAY = 5          # 崩溃后自动重启的等待时间（秒）
MAX_RESTARTS = 10          # 连续崩溃的最大重启次数（防死循环）
MAX_WORKERS = 8            # AI 回复线程池大小（防慢请求占满排队）


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def run_once(cfg: dict, log):
    """单次运行：初始化组件并进入消息轮询循环。"""
    ai_cfg = cfg.get("ai") or {}
    if "替换" in ai_cfg.get("api_key", ""):
        raise SystemExit("请先在 config.yaml 中填写 ai.api_key")

    from bot.ai_backend import AIBackend
    from bot.memory_store import MemoryStore
    from bot.rate_limit import ScenarioThrottle
    ai = AIBackend(ai_cfg)

    # 场景记忆持久化：每个群聊/私聊独立记忆文件，重启自动恢复
    mem_cfg = cfg.get("memory") or {}
    store = MemoryStore(mem_cfg.get("dir", "memory"), mem_cfg.get("enabled", True))
    contexts = ContextManager(ai_cfg.get("max_context_rounds", 8), store=store)

    # 场景级回复限流：所有回复路径（@回复/私聊回复/主动发言）共享同一冷却计时
    # 私聊默认不设回复间隔（0 = 不限制），群聊保留防刷屏冷却
    ri = cfg.get("reply_interval") or {}
    throttle = ScenarioThrottle(ri.get("private", 0), ri.get("group", 60))

    handler = MessageHandler(cfg, ai, contexts, throttle=throttle)
    client = WeChatClient(cfg)

    # 串行发送队列：所有回复统一入队，专职线程按序发送（防多回复乱序/穿插）
    send_queue = SendQueue(
        client, (cfg.get("wechat") or {}).get("send_queue_size", 200)
    )

    from bot.proactive import ProactiveEngine
    engine = ProactiveEngine(cfg, handler, client.bot_name, throttle=throttle)

    # Dashboard Web 管理后台（可选，默认关闭）：在守护线程中运行 Flask，
    # 不影响消息轮询主循环；所有 API 只读访问运行时组件
    dash_cfg = cfg.get("dashboard") or {}
    if dash_cfg.get("enabled", False):
        from dashboard.server import DashboardServer
        dash = DashboardServer(
            cfg,
            send_queue=send_queue,
            contexts=contexts,
            handler=handler,
            store=store,
            engine=engine,
        )
        dash.start()

    targets = client.group_whitelist + client.private_whitelist
    log.info("机器人启动完成，正在监听: %s（角色: %s，主动互动: %s，记忆持久化: %s）",
             targets, handler.roles.default,
             (cfg.get("proactive") or {}).get("mode", "keyword"),
             store.dir if store.enabled else "关闭")

    poll_interval = float((cfg.get("wechat") or {}).get("poll_interval", 1.0))
    status_interval = float((cfg.get("memory") or {}).get("status_interval", 60))
    last_refresh = 0.0
    last_status = 0.0
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            while True:
                loop_start = time.monotonic()
                for msg in client.poll():
                    # 同步执行（必须在提交线程池前）：
                    # 1) observe 立即重置群静默计时，防止主循环下一轮误判"群静默"
                    #    而对刚有成员发言的群触发主动发言（与@回复同内容重复发送）
                    # 2) 立即标记场景忙碌，主动发言引擎据此跳过正在处理的场景
                    engine.observe(msg)
                    key = scenario_key(msg)
                    handler.mark_busy(key)
                    pool.submit(_process, handler, client, engine, send_queue, msg, key)
                # 检查各群是否满足主动发言条件
                for chat in client.group_whitelist:
                    reason = engine.should_speak(chat)
                    if reason:
                        pool.submit(_proactive_speak, engine, send_queue, handler, chat, reason)
                # 每 30 秒检查一次是否有新打开的独立聊天窗口
                now = time.monotonic()
                if now - last_refresh > 30:
                    client.refresh_subwindows()
                    last_refresh = now
                # 周期性输出场景状态监控（活跃场景 = 10 分钟内有对话的会话）
                if now - last_status > status_interval:
                    _log_status(handler, contexts, log, send_queue)
                    last_status = now
                # 自适应休眠：本轮 poll 耗时（如主窗口切换 1s+）不额外叠加固定间隔
                time.sleep(max(0.1, poll_interval - (time.monotonic() - loop_start)))
    finally:
        # 退出（含 Ctrl+C）前冲刷发送队列：轮询已停止、微信窗口空闲，
        # 正好让排队中的回复（含此前卡住未发出的）完成发送，不再丢消息
        send_queue.wait_done(timeout=15)
        log.info("发送队列已冲刷完成（%s）", send_queue.stats())


def _log_status(handler: MessageHandler, contexts, log, send_queue: SendQueue = None):
    """状态监控：输出各活跃场景的记忆条数、回复次数与最近活跃时间。"""
    stats = contexts.stats()
    if not stats:
        log.info("[状态监控] 暂无活跃场景")
        return
    now = time.time()
    parts = []
    for key, info in sorted(stats.items()):
        if now - info["last_ts"] > 600:  # 10 分钟无动静的场景视为不活跃
            continue
        replies = handler.reply_counts.get(key, 0)
        age = int(now - info["last_ts"])
        parts.append(f"{key}(记忆{info['messages']}条/回复{replies}次/{age}s前活跃)")
    # 发送队列状态监控：待发送/已发送/失败/丢弃
    if send_queue is not None:
        q = send_queue.stats()
        parts.append(f"发送队列(待发{q['pending']}/已发{q['sent']}/失败{q['failed']}/丢弃{q['dropped']})")
    if parts:
        log.info("[状态监控] 活跃场景: %s", " | ".join(parts))
    else:
        log.info("[状态监控] 暂无活跃场景（已记录 %d 个场景）", len(stats))


def _process(handler: MessageHandler, client: WeChatClient, engine, send_queue: SendQueue,
             msg, key: str = ""):
    """在线程池中执行：AI 生成回复并入发送队列（不阻塞消息轮询）。

    observe 已在主循环同步完成；处理完毕（回复入队或无需回复）后清除忙碌标记。
    """
    try:
        reply = handler.handle(msg)
        if reply:
            # 被@的回复最高优先级，普通回复次之；队列保证按序串行发送
            priority = PRIORITY_AT if msg.is_at else PRIORITY_NORMAL
            send_queue.submit(msg.chat_name, reply, priority)
    except Exception:
        import logging
        logging.getLogger("teda_bot").exception("处理消息时发生未预期异常")
    finally:
        if key:
            handler.clear_busy(key)


def _proactive_speak(engine, send_queue: SendQueue, handler, chat: str, reason: str):
    """主动发言任务：生成内容并加入发送队列。"""
    try:
        reply = engine.speak(chat, reason)
        if reply:
            send_queue.submit(chat, reply, PRIORITY_PROACTIVE)
            handler.record_sent(chat, reply)  # 发送成功后登记，供重复检测使用
    except Exception:
        import logging
        logging.getLogger("teda_bot").exception("主动发言时发生未预期异常")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)  # 保证相对路径（config/logs）稳定
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)
    log = setup_logging(cfg.get("logging") or {})

    restarts = 0
    while restarts < MAX_RESTARTS:
        try:
            run_once(cfg, log)
        except KeyboardInterrupt:
            log.info("收到退出信号，机器人已停止")
            break
        except SystemExit as e:
            log.error("%s", e)
            break
        except Exception as e:
            restarts += 1
            log.exception("主循环异常（第%d次自动重启）: %s", restarts, e)
            time.sleep(RESTART_DELAY)
    else:
        log.error("连续崩溃 %d 次，停止自动重启，请检查日志", MAX_RESTARTS)


if __name__ == "__main__":
    main()
