import sys; sys.path.insert(0, r'D:\ai')
from flux2_3tier import text2img_f2

fn = text2img_f2(
    prompt='a cute orange cat',
    steps=4,
    width=512,
    height=512,
    resident_d=0,
    seed=1,
    images=[r'D:\ai\output\images\flux2_full_test.png'],   # 参考猫图
    embeds_file=r'D:\ai\_cached_flux2_embeds_test.pt',      # 现成 embeds (prompt 匹配)
)
print('I2V_ENGINE_RESULT:' + fn, flush=True)
