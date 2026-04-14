import rasterio
from pathlib import Path
from typing import List


class MaridaImageProcessor:
    """
    Procesare imaginilor din setul de date MARIDA: taiere in grid si salvare organizata pe disc
    """

    def __init__(self, root_dir: str, grid_size: int = 4):
        self.root_path = Path(root_dir)
        self.output_path = self.root_path / "Patches_Output"
        self.grid_size = grid_size



    def get_raw_image_paths(self) -> List[Path]:
        """Extrage doar caile catre imaginile Sentinel-2 """
        patches_src = self.root_path / "patches"


        return sorted([
            f for f in patches_src.rglob("*.tif")
            if not f.name.endswith("_cl.tif") and not f.name.endswith("_conf.tif")
        ])

    def process_and_save_all(self):
        """Metoda principala care ruleaza  pipeline-ul"""
        raw_images = self.get_raw_image_paths()
        total_images = len(raw_images)

        print(f"Total imagini {total_images} imagini")
        print(f"Configuratie: Grid {self.grid_size}x{self.grid_size} ({self.grid_size ** 2} patch-uri/imagine)\n")

        for idx, img_path in enumerate(raw_images):
            # incarca datele sentinel-2 (toate benzile)
            with rasterio.open(img_path) as src:
                data = src.read()
                meta = src.meta.copy()

            #calculeaza dimensiunile patch-urilor
            _, h, w = data.shape
            p_h, p_w = h // self.grid_size, w // self.grid_size

            #creaza folderul specific pt aceasta imagine
            img_folder_name = img_path.stem # numele fara extensie
            save_dir = self.output_path / img_folder_name
            save_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            # taiere grid si salvare
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    #  fereastra de taiere
                    y_start, x_start = i * p_h, j * p_w
                    patch = data[:, y_start:y_start + p_h, x_start:x_start + p_w]

                    # actualizare metadate pt noul fisier (dimensiuni noi)
                    patch_meta = meta.copy()
                    patch_meta.update({
                        "height": p_h,
                        "width": p_w,
                        "transform": rasterio.windows.transform(
                            rasterio.windows.Window(x_start, y_start, p_w, p_h), meta['transform']
                        )
                    })

                    patch_name = save_dir / f"patch_{count}.tif"

                    with rasterio.open(patch_name, 'w', **patch_meta) as dst:
                        dst.write(patch)

                    count += 1

            print(f"[{idx + 1}/{total_images}] Procesat: {img_folder_name}")
            print(f"  Generat: {count} patch-uri | Rezoluție patch: {p_w}x{p_h} pixeli")

        print(f"\npatch-urile au fost salvate in: {self.output_path} ---")


if __name__ == "__main__":
    BASE_DIR = r"D:\TAID\Disertatie\MARIDA"

    try:
        processor = MaridaImageProcessor(root_dir=BASE_DIR, grid_size=4)
        processor.process_and_save_all()
    except Exception as e:
        print(f"Eroare in pipeline: {e}")