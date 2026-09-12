import sys; sys.path.insert(0, r'D:\ai')
from flux2_3tier import text2img_f2
fn = text2img_f2(
    prompt='a cute orange cat, same cat, keep the cat identity',
    steps=4, width=512, height=512, resident_d=0,
    images=[r'D:\ai\output\images\flux2_full_test.png'],
    seed=1,
)
print('I2V_RESULT:' + fn, flush=True)
