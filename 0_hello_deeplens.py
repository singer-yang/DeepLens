"""
"Hello, world!" for DeepLens.

In this code, we will load a lens from a file. Then we will plot the lens setup and render a sample image.

Technical Paper:
    [1] Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
    [2] Congli Wang, Ni Chen, and Wolfgang Heidrich, "dO: A differentiable engine for Deep Lens design of computational imaging systems," IEEE TCI 2023.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

from deeplens import GeoLens


def main():
    lens = GeoLens(filename="./lenses/camera/iter17500.json")
    # lens = GeoLens(filename='./lenses/cellphone/cellphone80deg.json')
    # lens = GeoLens(filename='./lenses/zemax_double_gaussian.zmx')
    lens.analysis(
        f"./initial_lens",
        zmx_format=True,
        plot_invalid=True,
        multi_plot=False,
    )
    lens.optimize(
         lrs=[5e-4, 1e-4, 0.1, 1e-4],
         decay=0.02,
         iterations=5000,
         centroid=False,
         importance_sampling=True,
         optim_mat=True,
         match_mat=False,
         result_dir="./result",
    )

    # =====> 3. Analyze final result
    lens.prune_surf(expand_surf=0.02)
    lens.post_computation()

    logging.info(
        f"Actual: diagonal FOV {lens.hfov}, r sensor {lens.r_sensor}, F/{lens.fnum}."
    )
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens", zmx_format=True)

    # =====> 4. Create video
    create_video_from_images(f"{result_dir}", f"{result_dir}/autolens.mp4", fps=10)



if __name__ == "__main__":
    main()
