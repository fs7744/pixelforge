import type { PartDef } from './components/PartPainter'

/**
 * 常用角色部位预设。
 *
 * 放在这里而不是各页面自己写：后处理页与动画编辑器都要用同一份，
 * 两边不一致会导致「在 A 页涂的部位名，到 B 页认不出来」。
 *
 * 命名尽量用「部位」而不是「朝向+序号」——装骨架时靠名字判断
 * pivot 位置（头绕脖子转、手臂绕肩转）与父子关系。
 */
/**
 * 「自动识别」模式用的色块：只有颜色，没有部位名。
 *
 * 用户涂色时看到的是「一个绿色的块」，不是「左上臂」。让他先命名再涂，
 * 等于把工作推给用户。所以默认给几块纯色，涂完由服务端按几何识别
 * （头/躯干/四肢/尾巴），前端再把识别结果显示出来、允许改名。
 */
export const AUTO_PARTS: PartDef[] = [
  { id: 'a1', name: '色 1', color: '#ff3b30' },
  { id: 'a2', name: '色 2', color: '#34c759' },
  { id: 'a3', name: '色 3', color: '#0a84ff' },
  { id: 'a4', name: '色 4', color: '#ff9f0a' },
  { id: 'a5', name: '色 5', color: '#bf5af2' },
  { id: 'a6', name: '色 6', color: '#5ac8fa' },
  { id: 'a7', name: '色 7', color: '#ff2d55' },
  { id: 'a8', name: '色 8', color: '#ffd60a' },
]

export const DEFAULT_PARTS: PartDef[] = [
  { id: 'head', name: '头', color: '#ff3b30' },
  { id: 'body', name: '身体', color: '#34c759' },
  { id: 'arm_l', name: '左上臂', color: '#0a84ff' },
  { id: 'arm_r', name: '右上臂', color: '#ff9f0a' },
  { id: 'leg_l', name: '左腿', color: '#bf5af2' },
  { id: 'leg_r', name: '右腿', color: '#5ac8fa' },
  { id: 'tail', name: '尾巴/附属', color: '#ff2d55' },
]

/**
 * 分节部位：名字里的数字是「这几节属于同一条链」的显式标记。
 *
 * 装骨架时，尾巴1→尾巴2→尾巴3 会串成父子链——转第一节时后面几节跟着摆，
 * 才有波浪感。不用数字的话（比如叫「尾巴A/尾巴B」），服务端无从判断
 * 哪两节该接在一起。
 *
 * 为什么要靠名字而不是自动猜：几何启发式会把「左上臂/右上臂」误串成一条链
 * （核心词都是「臂」），而它们本该是两条独立肢体。数字是用户明确表达的意图。
 */
export const EXTRA_PARTS: PartDef[] = [
  { id: 'tail_1', name: '尾巴1', color: '#ff375f' },
  { id: 'tail_2', name: '尾巴2', color: '#c86bfa' },
  { id: 'tail_3', name: '尾巴3', color: '#a2845e' },
]
