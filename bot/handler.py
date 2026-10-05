"""消息处理管线：白名单 → 敏感过滤 → 触发判定 → 频率限制 → AI 调用 → 重复检测 → 生成回复。

场景模型：
- 每个群聊/私聊是一个独立场景，键为 group:<名> / private:<名>（记忆互相隔离）
- 私聊：每条新消息即时回复
- 群聊：@/关键词即时回复 + 主动引擎按消息阈值触发（差异化频率策略）
- 每个场景有独立的最小回复间隔（防刷屏）与回复锁（并发安全）
"""

import logging
import threading
from collections import deque
from difflib import SequenceMatcher

from bot.matcher import match_keywords, strip_mentions
from bot.rate_limit import RateLimiter, ScenarioThrottle
from bot.role_cards import RoleCards
from bot.safety import SafetyFilter

log = logging.getLogger("teda_bot.handler")


def scenario_key(msg) -> str:
    """消息所属场景键：群聊与私聊前缀不同，同名群/好友记忆完全隔离。"""
    return f"{'group' if msg.is_group else 'private'}:{msg.chat_name}"


class MessageHandler:
    def __init__(self, cfg: dict, ai, contexts, throttle: ScenarioThrottle = None):
        self.cfg = cfg
        self.ai = ai
        self.contexts = contexts
        self.roles = RoleCards(cfg.get("roles"), cfg.get("bot", {}).get("name", ""))
        self.safety = SafetyFilter(
            (cfg.get("safety") or {}).get("block_words", []),
            (cfg.get("safety") or {}).get("enabled", True),
        )
        trig = cfg.get("trigger") or {}
        self.keywords = trig.get("keywords", [])
        self.fuzzy = trig.get("fuzzy", True)
        self.fuzzy_threshold = trig.get("fuzzy_threshold", 0.75)
        self.reply_on_at = trig.get("reply_on_at", True)
        self.reply_private = trig.get("reply_private", True)
        # 群聊回复是否在开头 @发送者（默认开启：@提问人）
        self.reply_at_sender = trig.get("reply_at_sender", True)
        # 与最近回复相似度 >= 该值时拦截（防重复发送）
        self.duplicate_threshold = float(trig.get("duplicate_threshold", 0.8))
        # 上下文时间边界：超过该秒数的旧消息不再带入 AI（防误关联历史）
        self.context_max_age = float((cfg.get("ai") or {}).get("context_max_age", 1800))
        # 主动互动模式：context 模式下关键词不再触发直接回复（@机器人仍回复）
        self.proactive_mode = str((cfg.get("proactive") or {}).get("mode", "keyword")).lower()
        self.limiter = RateLimiter((cfg.get("ai") or {}).get("request_interval", 2.0))
        # 每场景最小回复间隔（防刷屏）：私聊默认 0 不限制，群聊可配置
        ri = cfg.get("reply_interval") or {}
        self.throttle = throttle or ScenarioThrottle(
            ri.get("private", 0), ri.get("group", 60)
        )
        # 每个场景最近发出的回复（用于重复发送检测）
        self._recent_replies: dict = {}
        # 每个场景忙碌标记：消息到达即标记、回复入队后清除，
        # 供主动发言引擎判断（场景正在生成回复时不再主动冒泡，防止同内容重复发送）
        self._busy: set = set()
        self._busy_lock = threading.Lock()
        # 每个场景一把锁：同场景消息串行处理（保证回复与记忆一致），跨场景并发不受影响
        self._scene_locks: dict = {}
        self._locks_guard = threading.Lock()
        # 状态统计：各场景回复次数
        self.reply_counts: dict = {}

    def _scene_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            if key not in self._scene_locks:
                self._scene_locks[key] = threading.Lock()
            return self._scene_locks[key]

    # ---------- 场景忙碌标记 ----------

    def mark_busy(self, key: str):
        """标记场景忙碌（消息到达时调用，在主循环同步执行避免竞态）。"""
        with self._busy_lock:
            self._busy.add(key)

    def clear_busy(self, key: str):
        """清除场景忙碌标记（回复已入队或处理完毕时调用）。"""
        with self._busy_lock:
            self._busy.discard(key)

    def is_busy(self, key: str) -> bool:
        """场景是否有消息正在处理/回复正在生成。"""
        with self._busy_lock:
            return key in self._busy

    def should_reply(self, msg) -> bool:
        """触发判定：关键词命中 / 被@ / 私聊，满足任一即回复。"""
        if not msg.is_group:
            return self.reply_private
        if msg.is_at and self.reply_on_at:
            return True
        if self.proactive_mode == "context":  # 纯上下文模式：忽略关键词
            return False
        return match_keywords(
            msg.content, self.keywords, self.fuzzy, self.fuzzy_threshold
        ) is not None

    def handle(self, msg) -> str | None:
        """处理一条消息，返回要发送的回复文本；不需要回复返回 None。

        同一场景的消息串行处理；跨场景完全并发。
        """
        if not self.should_reply(msg):
            return None
        if self.safety.is_blocked(msg.content):
            log.warning("消息命中敏感词，已忽略 [%s/%s]", msg.chat_name, msg.sender)
            return None

        key = scenario_key(msg)
        with self._scene_lock(key):
            # 场景级防刷屏：距上次回复不足最小间隔则跳过。
            # 被 @机器人 的消息享有最高优先级，绕过冷却立即回复
            if not msg.is_at and not self.throttle.allow(key):
                log.info("场景 [%s] 处于回复间隔冷却中，跳过消息: %s", key, msg.content[:40])
                return None

            log.info("收到消息 [%s/%s]: %s", msg.chat_name, msg.sender, msg.content[:80])
            # 清除当前消息中的 @提及（如 @机器人昵称 "@xiang "），
            # 防止 AI 把 "@xiang" 解析为对话对象/人名导致答非所问；
            # 清洗后为空（整条只有 @提及）时回退原文，保证仍可回复
            clean_msg = strip_mentions(msg.content) or msg.content
            ctx = self.contexts.get(key)
            self.limiter.wait()
            try:
                reply = self.ai.chat(
                    self.roles.system_prompt(user=msg.sender),
                    ctx.snapshot(self.context_max_age),  # 时间边界：旧消息不带入
                    clean_msg,
                )
            except Exception as e:
                log.error("AI 调用失败，跳过该消息: %s", e)
                return None
            if not reply:
                return None

            # 清洗 AI 输出中的 @提及（防止 AI 模仿历史格式产生 @self 或乱@人）
            clean = strip_mentions(reply)
            if not clean:
                return None

            # 重复发送检测：与该场景最近回复高度相似则自动拦截
            if self.is_duplicate(msg.chat_name, clean):
                log.warning("回复与最近发送内容高度相似，已拦截 [%s]: %s",
                            msg.chat_name, clean[:60])
                return None

            # 上下文存发送者标注的 user 消息 + 不带 @ 的纯文本回复，避免 AI 模仿 @ 格式。
            # 群消息清除 @提及后存入：AI 不会把 "@机器人" 当成别人话术，也不会误归属发言人
            user_text = clean_msg
            if msg.is_group and msg.sender:
                user_text = f"{msg.sender}: {clean_msg}"
            self.contexts.record(key, "user", user_text)      # 写入并自动持久化
            self.contexts.record(key, "assistant", clean)     # assistant = 机器人自己说的话
            self.record_sent(msg.chat_name, clean)
            self.throttle.mark(key)                           # 登记本次回复时间（冷却开始）
            self.reply_counts[key] = self.reply_counts.get(key, 0) + 1

            # 按配置在开头 @提问人
            if self.reply_at_sender and msg.is_group and msg.sender:
                clean = f"@{msg.sender} {clean}"
            return clean

    # ---------- 发送状态跟踪 ----------

    def is_duplicate(self, chat_name: str, text: str) -> bool:
        """与该会话最近几条已发送回复相似度过高即视为重复。"""
        for prev in self._recent_replies.get(chat_name, ()):
            if SequenceMatcher(None, text, prev).ratio() >= self.duplicate_threshold:
                return True
        return False

    def record_sent(self, chat_name: str, text: str):
        """记录已发送回复（发送成功后调用）。"""
        recent = self._recent_replies.setdefault(chat_name, deque(maxlen=5))
        recent.append(text)
