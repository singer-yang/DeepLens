import json
import string
import random
import os
from datetime import datetime

from deeplens import GeoLens


def write_lens_zmx_transform(self):

    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + "-json_to_zmx-" + random_string
    result_dir = f"./Transform-json/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    last_part = os.path.basename(self.filename)
    self.save_dir=f"{result_dir}/{last_part}.zmx"

    """Write the lens into .zmx file."""
    lens_zmx_str = ""
    enpd=self.surfaces[0].r * 2
    #     ENPD {enpd}
    # Head string
    head_str = f"""VERS 190513 80 123457 L123457
MODE SEQ
NAME 
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
ENVD 2.0E+1 1 0
GFAC 0 0
GCAT OSAKAGASCHEMICAL MISC
XFLN 0. 0. 0.
YFLN 0.0 2 5
WAVL 0.4861327 0.5875618 0.6562725
RAIM 0 0 1 1 0 0 0 0 0
PUSH 0 0 0 0 0 0
SDMA 0 1 0
FTYP 0 0 3 3 0 0 0
ROPD 1
PICB 1
PWAV 2
POLS 1 0 1 0 0 1 0
GLRS 1 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 1.0E-3 5 1.0E-6 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
TYPE STANDARD
CURV 0.0
DISZ INFINITY
"""
    lens_zmx_str += head_str

    # Surface string
    for i, s in enumerate(self.surfaces):
        d_next = (
            self.surfaces[i + 1].d - self.surfaces[i].d
            if i < len(self.surfaces) - 1
            else self.d_sensor - self.surfaces[i].d
        )
        surf_str = s.zmx_str(surf_idx=i + 1, d_next=d_next)
        lens_zmx_str += surf_str

    # Sensor string
    sensor_str = f"""SURF {i+2}
TYPE STANDARD
CURV 0.
DISZ 0.0
DIAM {self.r_sensor} 
"""
    lens_zmx_str += sensor_str

    # Write lens zmx string into file
    with open(self.save_dir, "w") as f:
        f.writelines(lens_zmx_str)
        f.close()

# 主函数
def main():
    lens = GeoLens(filename=r"D:\cursor_deeplens\DeepLens-main\lenses\cellphone\cellphone68deg.json")
    pupil_z, pupil_x = lens.entrance_pupil()
    pupil_D = pupil_x * 2
    lens.write_lens_zmx_transform(pupil_D)
if __name__ == "__main__":
    main()