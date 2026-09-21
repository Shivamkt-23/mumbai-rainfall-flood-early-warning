from pathlib import Path
import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "GPM_IMERG"
    / "3B-HHR.MS.MRG.3IMERG.20240901-S000000-E002959.0000.V07B.HDF5"
)


def print_structure(name, obj):

    if isinstance(obj, h5py.Dataset):

        print(
            f"DATASET: {name} | "
            f"shape={obj.shape} | "
            f"dtype={obj.dtype}"
        )

    else:

        print(
            f"GROUP:   {name}"
        )


def main():

    print("Opening IMERG file...")
    print(FILE_PATH)

    with h5py.File(FILE_PATH, "r") as hdf:

        print("\n" + "=" * 70)
        print("IMERG HDF5 STRUCTURE")
        print("=" * 70)

        hdf.visititems(
            print_structure
        )

        print("\n" + "=" * 70)
        print("ROOT ATTRIBUTES")
        print("=" * 70)

        for key, value in hdf.attrs.items():

            print(
                f"{key}: {value}"
            )


if __name__ == "__main__":
    main()