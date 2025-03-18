
from deeplens import GeoLens

# main
def main():
    lens = GeoLens(filename=r"D:\cursor_deeplens\DeepLens-main\lenses\cellphone\cellphone68deg.json")
    lens.write_lens_zmx(filename="./test.zmx")
if __name__ == "__main__":
    main()