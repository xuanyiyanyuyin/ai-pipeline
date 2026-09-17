# -*- coding: utf-8 -*-
"""★ 正式修复 (2026-09-13): RoPE 非持久 buffer 带着未初始化内存上线。
- to_empty(device='cpu') 会把 persistent=False 的 buffer 重建为未初始化内存
- load_state_dict 的 sd 里没有这些 key → *_inv_freq / *_original_inv_freq 从不被写入
- 云端这批内存含 NaN 位模式 → 整层 hidden 全 NaN (空输出); 本地是干净页 → 只是静默频率失效
修法: 按官方 _init_weights 的 RoPE 分支, 用 config 重算 inv_freq 覆盖。
覆盖两条构建路径: 43 层 (_build_template/build_and_inject) + MTP 模块。
幂等。"""
import io

P = '/workspace/engine/v4flash_chunked_sched.py'
s = io.open(P, encoding='utf-8').read()
orig = s

HELPER = '''

# ═══ RoPE buffer 修复 (2026-09-13) ═══
# 根因: _build_template/V4MTPModule 里的 to_empty(device='cpu') 把 persistent=False 的
# buffer 重建为「未初始化内存」, 而 load_state_dict 的 state_dict 不含这些 key
# → RoPE 的 *_inv_freq / *_original_inv_freq 会带着上一手内存数据上线。
# 云端该内存常含 NaN 位模式(FP8 权重读盘留下的 0xFF) → 该层 hidden 全 NaN;
# 本地拿到的是干净页 → 只表现为频率静默失效(值≈0), 不报错。
_ROPE_FIX = {'n_mod': 0, 'n_buf': 0, 'n_bad_before': 0, 'n_bad_after': 0}


def _reinit_rope_buffers(mod, tag=''):
    """照官方 transformers `_init_weights` 的 DeepseekV4RotaryEmbedding 分支重算 inv_freq。"""
    try:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        except Exception:
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import ROPE_INIT_FUNCTIONS
    except Exception as _e:
        print('[ropefix] 导入失败, 跳过:', _e, flush=True)
        return
    _ROPE_FIX['n_mod'] += 1
    for _m in mod.modules():
        if not isinstance(_m, DeepseekV4RotaryEmbedding):
            continue
        for _lt in getattr(_m, 'layer_types', []):
            _b = getattr(_m, f'{_lt}_inv_freq', None)
            if _b is None:
                continue
            _bo = getattr(_m, f'{_lt}_original_inv_freq', None)
            try:
                if int((~torch.isfinite(_b)).sum()) or (_bo is not None and int((~torch.isfinite(_bo)).sum())):
                    _ROPE_FIX['n_bad_before'] += 1
            except Exception:
                pass
            _rt = (getattr(_m, 'rope_type', None) or {}).get(_lt, 'default')
            _fn = _m.compute_default_rope_parameters if _rt == 'default' else ROPE_INIT_FUNCTIONS[_rt]
            _inv, _ = _fn(_m.config, layer_type=_lt)
            with torch.no_grad():
                _b.copy_(_inv.to(device=_b.device, dtype=_b.dtype))
                if _bo is not None:
                    _bo.copy_(_inv.to(device=_bo.device, dtype=_bo.dtype))
            _ROPE_FIX['n_buf'] += 1
            try:
                if int((~torch.isfinite(_b)).sum()):
                    _ROPE_FIX['n_bad_after'] += 1
            except Exception:
                pass
    # 体检: 小张量里还有没有非有限值 (其它未初始化 buffer 的话能立刻发现)
    _bad = []
    for _n, _t in list(mod.named_parameters()) + list(mod.named_buffers()):
        try:
            if _t.numel() <= 65536 and int((~torch.isfinite(_t)).sum()):
                _bad.append(_n)
        except Exception:
            pass
    if _bad:
        print('[ropefix] %s 仍有非有限小张量: %s' % (tag, _bad[:8]), flush=True)


'''
if 'def _reinit_rope_buffers' not in s:
    a0 = 'def convert_layer_sd(raw_sd: dict) -> dict:'
    assert a0 in s, 'convert_layer_sd 锚点未找到'
    s = s.replace(a0, HELPER.lstrip('\n') + '\n' + a0, 1)
    print('  [1] 修复函数已注入')

# 层构建路径
a1 = '        layer.load_state_dict(sd, strict=True, assign=True)  # assign 保持 sd dtype (norm bf16, FP8 fp8)'
if '_reinit_rope_buffers(layer' not in s:
    assert a1 in s, '层 load_state_dict 锚点未找到'
    s = s.replace(a1, a1 + "\n        _reinit_rope_buffers(layer, 'lid=%d' % lid)   # ★ 修复: 非持久 buffer 未被 sd 覆盖", 1)
    print('  [2] 层路径已插入')

# MTP 路径
a2 = '        self.load_state_dict(conv, strict=True, assign=True)  # 保持 sd dtype'
if '_reinit_rope_buffers(self' not in s:
    assert a2 in s, 'MTP load_state_dict 锚点未找到'
    s = s.replace(a2, a2 + "\n        _reinit_rope_buffers(self, 'mtp')   # ★ 修复: 同上", 1)
    print('  [3] MTP 路径已插入')

# 启动汇总
a3 = '        self.layers = [self.build_and_inject(lid) for lid in range(self.cfg.num_hidden_layers)]'
if '[ropefix] 汇总' not in s:
    assert a3 in s, '构建层锚点未找到'
    s = s.replace(a3, a3 + """
        if _ROPE_FIX['n_buf']:
            print('[ropefix] 汇总 模块=%d 重算缓冲=%d 修复前带NaN=%d 修复后带NaN=%d' % (
                _ROPE_FIX['n_mod'], _ROPE_FIX['n_buf'], _ROPE_FIX['n_bad_before'],
                _ROPE_FIX['n_bad_after']), flush=True)""", 1)
    print('  [4] 启动汇总已插入')

if s != orig:
    io.open(P, 'w', encoding='utf-8').write(s)
    print('修复已写入:', P)
else:
    print('无变化')
