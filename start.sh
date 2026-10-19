#!/usr/bin/env bash
# PixelForge 一键启动（macOS / Linux / Windows Git Bash）
#
# 首次运行会自动做三件事：建 .venv 装依赖 → 构建前端 → 启动服务。
# 之后再运行会跳过已完成的步骤，几秒内起来。
#
# 用法：
#   ./start.sh                启动（自动打开浏览器）
#   ./start.sh --no-browser   启动但不开浏览器
#   ./start.sh --port 9000    指定端口（被占用时自动顺延）
#   ./start.sh --reload       开发模式：改后端代码自动重载
#   ./start.sh --setup        只准备环境，不启动
#   ./start.sh --check        只做自检，打印还缺什么
#
# 环境变量：
#   PF_PYTHON     指定 Python 解释器（默认自动探测 3.10–3.13，优先用项目 .venv）
#   PF_VENV       虚拟环境目录（默认 <项目>/.venv）
#   PF_PIP_INDEX  自定义 pip 源（默认先用本机 pip 配置，失败后自动切阿里云）
#   HOST / PORT   监听地址（默认 127.0.0.1:8787）
set -euo pipefail

# 有人会用 sh start.sh 跑；本脚本用到数组与 BASH_SOURCE，自动切回 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="${PF_VENV:-$ROOT/.venv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8787}"
OPEN_BROWSER=1
RELOAD=0
SETUP_ONLY=0
CHECK_ONLY=0
PY=""

# ---------- 输出 ----------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_DIM=$'\033[2m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_B=$'\033[1m'; C_0=$'\033[0m'
else
  C_DIM=''; C_OK=''; C_WARN=''; C_ERR=''; C_B=''; C_0=''
fi
step() { printf '%s==>%s %s\n' "$C_B" "$C_0" "$1"; }
ok()   { printf '  %s✓%s %s\n' "$C_OK" "$C_0" "$1"; }
warn() { printf '  %s!%s %s\n' "$C_WARN" "$C_0" "$1"; }
die()  { printf '\n%s[错误]%s %s\n' "$C_ERR" "$C_0" "$1" >&2; exit 1; }
dim()  { printf '%s%s%s\n' "$C_DIM" "$1" "$C_0"; }

usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"; }

# ---------- 参数 ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --no-browser|-n) OPEN_BROWSER=0; shift ;;
    --reload)        RELOAD=1; shift ;;
    --setup)         SETUP_ONLY=1; shift ;;
    --check)         CHECK_ONLY=1; OPEN_BROWSER=0; shift ;;
    --port)          [ $# -ge 2 ] || die "--port 需要一个端口号"; PORT="$2"; shift 2 ;;
    --port=*)        PORT="${1#*=}"; shift ;;
    --host)          [ $# -ge 2 ] || die "--host 需要一个地址"; HOST="$2"; shift 2 ;;
    --host=*)        HOST="${1#*=}"; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "未知参数：$1（用 --help 查看用法）" ;;
  esac
done

# ---------- 1. 找 Python ----------
MIN_VER="3.10"
MAX_VER="3.13"
# 上限比下限更要紧：3.14/3.15 上 PyYAML（经 uvicorn[standard] 引入）等依赖
# 还没有预编译 wheel，pip 会退回源码编译，然后卡在「需要 MSVC」而整包失败。
IN_RANGE=(3.13 3.12 3.11 3.10)

py_ver() { "$1" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null; }
py_ge_min() {
  "$1" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >/dev/null 2>&1
}
py_supported() {
  "$1" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >/dev/null 2>&1
}

# 把 "py -3.13" 这类带参数的命令解析成真实路径，后续统一按路径引用
resolve_exe() { $1 -c "import sys; print(sys.executable)" 2>/dev/null; }

find_base_python() {
  # 两轮：优先落在 3.10–3.13；实在没有才退回「过新的版本」并给出警告。
  local want="$1" cmd exe
  for v in "${IN_RANGE[@]}"; do
    for cmd in "python$v" "py -$v"; do
      exe="$(resolve_exe "$cmd")" || continue
      [ -x "$exe" ] || continue
      if [ "$want" = "supported" ]; then
        py_supported "$exe" || continue
      else
        py_ge_min "$exe" || continue
      fi
      printf '%s\n' "$exe"; return 0
    done
  done
  for cmd in python3 python "py -3"; do
    exe="$(resolve_exe "$cmd")" || continue
    [ -x "$exe" ] || continue
    if [ "$want" = "supported" ]; then py_supported "$exe" || continue; else py_ge_min "$exe" || continue; fi
    printf '%s\n' "$exe"; return 0
  done
  return 1
}

venv_python() {
  for c in "$VENV/bin/python" "$VENV/Scripts/python.exe"; do
    [ -x "$c" ] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}

create_venv() {
  step "创建虚拟环境 $VENV"
  "$1" -m venv "$VENV" || die "创建虚拟环境失败。Debian/Ubuntu 需先执行：sudo apt install python3-venv"
  PY="$(venv_python)" || die "虚拟环境创建后仍找不到 Python，请检查 $VENV 是否完整"
  ok "$("$PY" -V 2>&1)"
}

step "检查 Python 环境"
if [ -n "${PF_PYTHON:-}" ]; then
  [ -x "$PF_PYTHON" ] || die "PF_PYTHON 指向的解释器不存在：$PF_PYTHON"
  PY="$PF_PYTHON"
  ok "使用指定解释器：$PY（$(py_ver "$PY")）"
  py_ge_min "$PY" || die "指定解释器版本过低，需要 Python ${MIN_VER}+"
  py_supported "$PY" || warn "该版本高于 ${MAX_VER}，部分依赖可能没有预编译包，安装或将失败"
elif PY="$(venv_python)"; then
  if py_supported "$PY"; then
    ok "复用虚拟环境：${VENV#$ROOT/}（$(py_ver "$PY")）"
  else
    # 之前用 3.14/3.15 建过环境：依赖装不上，且会一直复用这个坏环境
    warn "现有虚拟环境是 Python $(py_ver "$PY")，超出支持范围 ${MIN_VER}–${MAX_VER}"
    dim "  该版本多数依赖还没有预编译 wheel，编译需要 MSVC，安装会失败"
    OLD="${VENV%/}.bak-$(date +%Y%m%d%H%M%S)"
    mv "$VENV" "$OLD" || die "无法移动旧虚拟环境：$VENV"
    dim "  旧环境已保留在 ${OLD#$ROOT/}"
    BASE="$(find_base_python supported)" || die "未找到 Python ${MIN_VER}–${MAX_VER}。请安装 https://www.python.org/downloads/"
    ok "使用系统 Python：$BASE（$(py_ver "$BASE")）"
    create_venv "$BASE"
  fi
else
  if BASE="$(find_base_python supported)"; then
    ok "使用系统 Python：$BASE（$(py_ver "$BASE")）"
  elif BASE="$(find_base_python any)"; then
    warn "只找到 Python $(py_ver "$BASE")（高于 ${MAX_VER}），依赖可能没有预编译包"
    dim "  建议安装 Python 3.12 或 3.13：https://www.python.org/downloads/"
  else
    die "未找到 Python ${MIN_VER}+。请先安装 Python（https://www.python.org/downloads/），安装时勾选「Add to PATH」。"
  fi
  create_venv "$BASE"
fi

# ---------- 2. 装依赖 ----------
REQ="$ROOT/backend/requirements.txt"
MISSING=""
"$PY" - <<'EOF' >/dev/null 2>&1 || MISSING=1
import fastapi, uvicorn, PIL, numpy, cv2  # noqa: F401
EOF

pip_install() {
  local args=(-m pip install --disable-pip-version-check)
  if [ -n "${PF_PIP_INDEX:-}" ]; then
    "$PY" "${args[@]}" -i "$PF_PIP_INDEX" "$@"
  else
    "$PY" "${args[@]}" "$@"
  fi
}

if [ -n "$MISSING" ]; then
  step "安装 Python 依赖（首次约 1-3 分钟）"
  "$PY" -m pip install --disable-pip-version-check -U pip >/dev/null 2>&1 || warn "pip 升级失败，继续用当前版本"

  # 兜底依赖集：不含 uvicorn[standard]，因此不会拉 PyYAML/uvloop 这些需要编译的包。
  # 少了 extras 只影响 --reload 的 watchfiles 热重载与 httptools 提速，功能不受影响。
  CORE_FALLBACK=(fastapi uvicorn pillow numpy python-multipart aiofiles httpx opencv-python-headless)
  MIRROR="https://mirrors.aliyun.com/pypi/simple"

  if ! pip_install -r "$REQ"; then
    if [ -z "${PF_PIP_INDEX:-}" ]; then
      warn "默认源安装失败，改用阿里云镜像重试"
      if ! "$PY" -m pip install --disable-pip-version-check -r "$REQ" -i "$MIRROR"; then
        warn "镜像仍失败，去掉 uvicorn 的 extras 兜底安装核心依赖"
        dim "  （只影响热重载与 HTTP 解析加速，功能完整）"
        "$PY" -m pip install --disable-pip-version-check "${CORE_FALLBACK[@]}" -i "$MIRROR" || \
          die "依赖安装失败。可指定镜像重试：PF_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple ./start.sh"
      fi
    else
      die "依赖安装失败，请检查网络或 PF_PIP_INDEX：$PF_PIP_INDEX"
    fi
  fi
  "$PY" -c "import fastapi, uvicorn, PIL, numpy, cv2" >/dev/null 2>&1 || \
    die "核心依赖仍不完整。若使用 Python ${MAX_VER} 以上，请改用 3.12/3.13 后删除 .venv 重试"
  ok "核心依赖就绪"
else
  ok "Python 依赖已就绪"
fi

# 可选：rembg + u2net 深度抠图。装不上不影响使用，抠图会自动退到 GrabCut。
# 失败一次就写标记，避免每次启动都重试拖慢速度。
SKIP_STAMP="$VENV/.pf-optional-skip"
if ! "$PY" -c "import rembg" >/dev/null 2>&1; then
  if [ -f "$SKIP_STAMP" ]; then
    dim "  （可选）rembg 之前装失败过，已跳过；删掉 .venv/.pf-optional-skip 可重试"
  else
    dim "  （可选）安装深度抠图依赖 rembg…"
    if "$PY" -m pip install --disable-pip-version-check -q -r "$ROOT/backend/requirements-optional.txt" >/dev/null 2>&1; then
      ok "深度抠图依赖已安装"
    else
      touch "$SKIP_STAMP" 2>/dev/null || true
      warn "rembg 未安装，抠图会退到 GrabCut（效果略差，功能正常）"
    fi
  fi
fi

# ---------- 3. 前端 ----------
step "检查前端"
DIST="$ROOT/frontend/dist/index.html"
if [ -f "$DIST" ]; then
  ok "前端已构建"
else
  if command -v npm >/dev/null 2>&1; then
    ok "未找到构建产物，开始构建"
    pushd "$ROOT/frontend" >/dev/null
    if [ -d node_modules ]; then
      ok "复用已有 node_modules"
    elif [ -f package-lock.json ]; then
      npm ci --no-audit --no-fund || die "npm ci 失败，请检查网络或手动执行：cd frontend && npm install"
    else
      npm install --no-audit --no-fund || die "npm install 失败，请检查网络"
    fi
    npm run build || die "前端构建失败，请手动执行：cd frontend && npm run build"
    popd >/dev/null
    ok "前端构建完成"
  else
    warn "未安装 Node.js/npm，无法构建前端"
    warn "界面将不可用，但 API 与 /docs 仍可访问"
    dim "  装好 Node.js 18+ 后重新运行本脚本即可：https://nodejs.org/"
  fi
fi

# ---------- 4. 模型提示 ----------
step "检查本地模型"
if [ -d "$ROOT/models" ] && [ -n "$(ls -A "$ROOT/models" 2>/dev/null)" ]; then
  ok "已检测到：$(ls -1 "$ROOT/models" | tr '\n' ' ')"
else
  warn "尚未下载模型，AI 生成不可用（抠图/像素化/延展等确定性管线不受影响）"
  dim "  下载 Z-Image（约 20GB，需 16GB+ 显存）：bash backend/fetch_zimage.sh"
fi

# ---------- 5. 选端口 ----------
# 用「能否连上」判断占用，而不是 bind：Windows 的 SO_REUSEADDR 允许重复绑定，
# bind 探测会把已被监听的端口误判为空闲。
free_port() {
  "$PY" - "$HOST" "$1" <<'EOF'
import socket, sys
host, start = sys.argv[1], int(sys.argv[2])
# 监听 0.0.0.0/:: 时不能直接连这个地址，探测改用回环
probe = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
for port in range(start, start + 30):
    with socket.socket() as s:
        s.settimeout(0.25)
        if s.connect_ex((probe, port)) != 0:
            print(port)
            break
else:
    sys.exit(1)
EOF
}

step "选择端口"
if ! CHOSEN="$(free_port "$PORT")"; then
  die "$PORT 起连续 30 个端口都被占用，请用 --port 指定其它端口"
fi
if [ "$CHOSEN" != "$PORT" ]; then
  warn "$PORT 已被占用，改用 $CHOSEN"
fi
PORT="$CHOSEN"
URL="http://$HOST:$PORT"
ok "$URL"

# ---------- 6. 自检 / 仅准备环境 ----------
if [ "$CHECK_ONLY" = "1" ]; then
  step "自检结果"
  ok "Python：$("$PY" -V 2>&1)  ($PY)"
  "$PY" -c "import fastapi, uvicorn, PIL, numpy; print('  ✓ Web 依赖齐全')" || warn "Web 依赖缺失"
  "$PY" -c "import cv2; print('  ✓ OpenCV 可用（GrabCut 抠图）')" 2>/dev/null || warn "OpenCV 缺失，抠图退到颜色键"
  "$PY" -c "import rembg; print('  ✓ rembg 可用（u2net 深度抠图）')" 2>/dev/null || warn "rembg 缺失（可选）"
  if [ -f "$DIST" ]; then ok "前端构建产物存在"; else warn "前端未构建"; fi
  if [ -d "$ROOT/models" ] && [ -n "$(ls -A "$ROOT/models" 2>/dev/null)" ]; then ok "本地模型存在"; else warn "本地模型未下载（AI 生成不可用）"; fi
  ok "端口 $PORT 可用"
  exit 0
fi

if [ "$SETUP_ONLY" = "1" ]; then
  step "环境已就绪"
  dim "  启动服务：./start.sh"
  exit 0
fi

# ---------- 7. 启动 ----------
step "启动服务"
dim "  按 Ctrl+C 停止"

open_browser() {
  case "${OSTYPE:-}" in
    darwin*) open "$URL" >/dev/null 2>&1 || true ;;
    msys*|cygwin*) cmd //c start "" "$URL" >/dev/null 2>&1 || true ;;
    *) xdg-open "$URL" >/dev/null 2>&1 || true ;;
  esac
}

if [ "$OPEN_BROWSER" = "1" ]; then
  ( sleep 2; open_browser ) &
fi

ARGS=(--host "$HOST" --port "$PORT")
if [ "$RELOAD" = "1" ]; then
  ARGS+=(--reload)
fi
exec "$PY" -m uvicorn backend.app.main:app "${ARGS[@]}"
