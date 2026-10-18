"""中文提示词 → 英文（面向游戏素材领域的词典翻译）。

背景：SD1.5 / SDXL / SD-Turbo 的文本编码器是 CLIP，训练语料以英文为主，
直接喂中文会丢掉语义（模型只看到一堆未知 token）。所以本地模型必须把中文
指令翻成英文。

这里走「领域术语词典 + 最长匹配」而不是通用机翻，原因：
- 游戏素材的描述高度模式化（颜色 + 角色 + 装备 + 动作 + 画风），词表能覆盖绝大多数说法
- 零依赖、零网络、毫秒级，符合本项目「本地优先」的定位
- 通用机翻对这种短语常常译得比词典更啰嗦，反而稀释 CLIP 权重

未覆盖的片段会原样返回在 unmatched 里，前端据此提示用户补充英文。
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# 词表按「长度优先」匹配，长词必须放前面由算法自行保证（按长度排序）
TERMS: Dict[str, str] = {
    # ---- 颜色 ----
    "灰白色": "grey and white", "灰白": "grey and white", "黑白": "black and white",
    "深蓝色": "dark blue", "浅蓝色": "light blue", "深绿色": "dark green", "浅绿色": "light green",
    "深红色": "dark red", "紫红色": "magenta", "金黄色": "golden", "银白色": "silver white",
    "红色": "red", "橙色": "orange", "黄色": "yellow", "绿色": "green", "青色": "cyan",
    "蓝色": "blue", "紫色": "purple", "粉色": "pink", "棕色": "brown", "褐色": "brown",
    "灰色": "grey", "白色": "white", "黑色": "black", "金色": "gold", "银色": "silver",
    "红": "red", "橙": "orange", "黄": "yellow", "绿": "green", "青": "cyan", "蓝": "blue",
    "紫": "purple", "粉": "pink", "棕": "brown", "灰": "grey", "白": "white", "黑": "black",
    "金": "gold", "银": "silver",

    # ---- 生物 / 角色 ----
    "英短": "british shorthair", "橘猫": "orange tabby cat", "狸花猫": "tabby cat",
    "猫": "cat", "狗": "dog", "狼": "wolf", "狐狸": "fox", "兔子": "rabbit", "老鼠": "rat",
    "熊": "bear", "老虎": "tiger", "狮子": "lion", "马": "horse", "猪": "pig", "牛": "cow",
    "羊": "sheep", "鸡": "chicken", "鱼": "fish", "青蛙": "frog", "蛇": "snake",
    "龙": "dragon", "恐龙": "dinosaur", "史莱姆": "slime", "骷髅": "skeleton", "僵尸": "zombie",
    "幽灵": "ghost", "恶魔": "demon", "天使": "angel", "哥布林": "goblin", "兽人": "orc",
    "骑士": "knight", "法师": "mage", "巫师": "wizard", "战士": "warrior", "剑士": "swordsman",
    "弓箭手": "archer", "盗贼": "rogue", "牧师": "priest", "武士": "samurai", "忍者": "ninja",
    "公主": "princess", "王子": "prince", "国王": "king", "女王": "queen", "商人": "merchant",
    "村民": "villager", "铁匠": "blacksmith", "农夫": "farmer", "士兵": "soldier",
    "机器人": "robot", "机甲": "mecha", "精灵": "elf", "矮人": "dwarf", "怪物": "monster",
    "敌人": "enemy", "boss": "boss", "主角": "protagonist", "宠物": "pet",

    # ---- 装备 / 道具 ----
    "武士刀": "katana", "太刀": "katana", "长剑": "long sword", "短剑": "dagger",
    "剑": "sword", "刀": "blade", "盾牌": "shield", "盾": "shield", "弓箭": "bow and arrow",
    "弓": "bow", "箭": "arrow", "法杖": "staff", "长枪": "spear", "斧头": "axe", "斧": "axe",
    "锤子": "hammer", "披风": "cape", "斗篷": "cloak", "头盔": "helmet", "铠甲": "armor",
    "盔甲": "armor", "靴子": "boots", "手套": "gloves", "腰带": "belt", "面具": "mask",
    "帽子": "hat", "背包": "backpack", "药水": "potion", "宝箱": "treasure chest",
    "钥匙": "key", "金币": "gold coin", "宝石": "gem", "卷轴": "scroll", "书本": "book",
    "火把": "torch", "灯笼": "lantern", "旗帜": "banner", "帐篷": "tent", "木箱": "crate",
    "桶": "barrel", "门": "door", "栏杆": "fence",

    # ---- 动作 ----
    "挥砍": "slashing with a sword", "劈砍": "slashing", "攻击": "attacking",
    "奔跑": "running", "跑步": "running", "行走": "walking", "走路": "walking",
    "待机": "idle", "站立": "standing", "跳跃": "jumping", "施法": "casting a spell",
    "防御": "blocking", "受伤": "taking damage", "死亡": "dying", "胜利": "victory pose",
    "坐下": "sitting", "睡觉": "sleeping", "举手": "raising hand", "挥手": "waving",
    "射箭": "shooting an arrow", "格挡": "parrying", "翻滚": "rolling", "冲刺": "dashing",

    # ---- 画风 / 规格 ----
    "像素风": "pixel art style", "像素": "pixel art", "Q版": "chibi", "q版": "chibi",
    "卡通": "cartoon", "日系": "japanese", "二次元": "anime", "写实": "realistic",
    "平涂": "flat colors", "厚涂": "painterly", "描边": "outline", "黑边": "black outline",
    "粗描边": "thick black outline", "等距": "isometric", "俯视": "top-down view",
    "侧视": "side view", "正视图": "front view", "背视图": "back view", "全身": "full body",
    "半身": "half body", "头像": "portrait", "单个": "single", "居中": "centered",
    "三视图": "character turnaround sheet", "精灵表": "sprite sheet",
    "透明背景": "transparent background", "纯色背景": "plain solid background",
    "背景": "background", "画面": "scene", "尺寸": "size",

    # ---- 场景 ----
    "森林": "forest", "沙漠": "desert", "城堡": "castle", "城镇": "town", "村庄": "village",
    "洞穴": "cave", "地牢": "dungeon", "海边": "seaside", "雪山": "snowy mountain",
    "草地": "grassland", "道路": "road", "房子": "house", "商店": "shop", "塔": "tower",
    "桥": "bridge", "河流": "river", "天空": "sky", "云": "cloud", "山": "mountain",
    "树": "tree", "花": "flower", "石头": "rock", "墙壁": "wall", "地板": "floor",
    "城市": "city", "街道": "street", "室内": "indoor", "室外": "outdoor",

    # ---- 连接词（译成英文时多数要丢掉）----
    "的": "", "和": "and", "与": "and", "以及": "and", "一个": "a", "一只": "a",
    "一位": "a", "拿着": "holding", "手持": "holding", "穿着": "wearing", "戴着": "wearing",
    "站在": "standing in", "在": "in", "中": "", "里": "", "上": "",
    "动作": "", "场景": "", "画面": "", "版": "", "风格": "style", "样式": "style",
    "纯": "", "纯色": "solid color", "纯绿色": "solid green", "纯绿": "solid green",
    "纯白色": "solid white", "纯白": "solid white", "纯蓝色": "solid blue",
    "身体": "body", "身体轮廓": "body silhouette",

    # ---- 补充：部位 / 数量 / 方位 ----
    "眼睛": "eyes", "两只": "two", "一只眼": "one eye", "耳朵": "ears", "尾巴": "tail",
    "翅膀": "wings", "角": "horns", "手臂": "arms", "腿": "legs", "头发": "hair",
    "远处": "distant", "近处": "foreground", "旁边": "beside", "上面": "above",
    "下面": "below", "左边": "left side", "右边": "right side", "中间": "center",
    "两个": "two", "三个": "three", "多个": "multiple", "几个": "several",
}

# 这些词要和后面的名词合并成一个短语（"red" + "cape" → "red cape"），
# 否则 CLIP 看到的是零散的 "red, cape"，修饰关系会被削弱。
_ADJ_EN = {
    "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown",
    "grey", "white", "black", "gold", "silver", "golden", "magenta",
    "grey and white", "black and white", "dark blue", "light blue", "dark green",
    "light green", "dark red", "silver white", "a", "two", "three", "several",
    "multiple", "distant", "foreground",
}

_ORDERED = sorted(TERMS.items(), key=lambda kv: -len(kv[0]))
_PUNCT = {"，": ", ", "。": ", ", "、": ", ", "；": ", ", ";": ", ", "/": ", ", "：": ", ",
          "（": "", "）": "", "(": "", ")": ""}
CJK = re.compile(r"[\u4e00-\u9fa5]")
DEFAULT_STYLE = ("2D chibi cartoon style, thick black outline, flat colors, "
                 "plain solid background, game asset, no text")


def _merge(words: List[str]) -> str:
    """把形容词与后续名词合并：["red","cape"] → "red cape"。

    CLIP 对「修饰语 + 名词」紧邻出现的权重明显高于逗号分隔的词袋，
    所以颜色 / 数量 / 方位词都要和后一个词黏在一起。
    """
    out: List[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if not w or w == "and":
            i += 1
            continue
        if w in _ADJ_EN:
            # 连续形容词 + 名词黏成一个短语：a + green + slime → a green slime
            phrase = [w]
            j = i + 1
            while j < len(words) and words[j] in _ADJ_EN:
                phrase.append(words[j])
                j += 1
            if j < len(words) and words[j] and words[j] != "and":
                phrase.append(words[j])
                j += 1
            out.append(" ".join(phrase))
            i = j
            continue
        out.append(w)
        i += 1
    # 合并后可能出现重复（同色同物），去重但保持顺序
    seen = set()
    uniq: List[str] = []
    for w in out:
        if w in seen:
            continue
        seen.add(w)
        uniq.append(w)
    return ", ".join(uniq).strip(" ,")


def has_chinese(text: str) -> bool:
    return bool(CJK.search(text or ""))


def translate(text: str) -> Tuple[str, dict]:
    """把中文素材描述译成英文短语。返回 (英文, 统计)。"""
    src = (text or "").strip()
    if not src:
        return "", {"coverage": 0.0, "unmatched": [], "auto": True}

    for zh, en in _PUNCT.items():
        src = src.replace(zh, ",")

    words: List[str] = []
    unmatched: List[str] = []
    matched_chars = 0
    total_chars = 0
    i = 0
    n = len(src)

    while i < n:
        ch = src[i]
        if not CJK.match(ch):
            i += 1
            continue
        hit = None
        for zh, en in _ORDERED:
            if src.startswith(zh, i):
                hit = (zh, en)
                break
        if hit:
            zh, en = hit
            total_chars += len(zh)
            matched_chars += len(zh)
            if en and (not words or words[-1] != en):
                words.append(en)
            i += len(zh)
        else:
            # 未收录：收集连续未匹配的中文片段
            j = i
            while j < n and CJK.match(src[j]):
                if any(src.startswith(zh, j) for zh, _ in _ORDERED):
                    break
                j += 1
            chunk = src[i:j] if j > i else src[i]
            total_chars += len(chunk)
            unmatched.append(chunk)
            i += len(chunk)

    english = _merge(words)
    coverage = round(matched_chars / total_chars, 3) if total_chars else 1.0
    meta = {
        "auto": True,
        "coverage": coverage,
        "unmatched": unmatched,
        "engine": "dict-zh2en",
    }
    return english, meta


def translate_prompt(text: str, add_style: bool = True) -> Tuple[str, dict]:
    """翻译并补上游戏素材通用风格后缀（缺风格词时才补）。"""
    english, meta = translate(text)
    if add_style and english:
        low = english.lower()
        if "outline" not in low and "pixel art" not in low:
            english = f"{english}, {DEFAULT_STYLE}"
            meta["styleAppended"] = True
    meta["source"] = text
    meta["english"] = english
    return english, meta
