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
export const DEFAULT_PARTS: PartDef[] = [
  { id: 'head', name: '头', color: '#ff3b30' },
  { id: 'body', name: '身体', color: '#34c759' },
  { id: 'arm_l', name: '左上臂', color: '#0a84ff' },
  { id: 'arm_r', name: '右上臂', color: '#ff9f0a' },
  { id: 'leg_l', name: '左腿', color: '#bf5af2' },
  { id: 'leg_r', name: '右腿', color: '#5ac8fa' },
  { id: 'tail', name: '尾巴/附属', color: '#ff2d55' },
]

/** 尾巴等长条部位可以再分节，涂的时候用不同颜色区分 */
export const EXTRA_PARTS: PartDef[] = [
  { id: 'tail_1', name: '尾巴1', color: '#ff375f' },
  { id: 'tail_2', name: '尾巴2', color: '#c86bfa' },
]
